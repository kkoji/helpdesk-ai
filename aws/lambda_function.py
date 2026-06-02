"""社内ヘルプデスクAI - AWS Lambda版

構成:
  [0] モジュールロード時のキャッシュ (コールドスタート対策)
  [1] ツール定義 (Jupyter版 Section 2 と対応)
  [2] LangGraph グラフ定義 (Jupyter版 Section 3 と対応)
  [3] 会話履歴の DynamoDB 永続化 (Jupyter版 Section 4 と対応)
  [4] Lambda ハンドラ (Jupyter版 Section 5 と対応)
"""

import json
import tempfile
import time
from datetime import date, datetime, timedelta
from typing import Annotated, Any, Literal, Optional, Sequence, TypedDict

import boto3
import faiss
import numpy as np
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

# ======================================================================
# [0] モジュールロード時のキャッシュ (コールドスタート対策)
# ======================================================================
# Lambda は同一コンテナ内でウォームリクエストを処理するため、
# モジュールスコープの変数は初回呼び出し後も再利用される。
# S3 アクセスを最小化するため、メタデータと本文をここにキャッシュする。
#
# 注: Udemy のラボ用 AWS 環境では KMS Encrypt が拒否されるため Lambda 環境変数が使えない。
# そのため、設定値はプレースホルダ (__XXX__) としてコードに埋め込み、
# deploy.sh が sed で実値に置換してから zip にパッケージする方式を採用している。

S3_BUCKET = "__S3_BUCKET__"
DOCS_PREFIX = "__DOCS_PREFIX__"
METADATA_KEY = "__METADATA_KEY__"
INDEX_KEY = "__INDEX_KEY__"
CHUNKS_KEY = "__CHUNKS_KEY__"
SESSION_HISTORY_TABLE = "__SESSION_HISTORY_TABLE__"
AWS_REGION = "__AWS_REGION__"
EMBED_MODEL_ID = "__EMBED_MODEL_ID__"

# チャット LLM は Bedrock の Anthropic Claude を Converse API 経由で呼ぶ。
# (埋め込みと同じく bedrock-runtime を使うため、追加の API キーは不要)
CHAT_MODEL_ID = "__CHAT_MODEL_ID__"

# RAG 検索の上位 K 件
RAG_TOP_K = 3

# グラフの再帰上限 (call_model ⇄ call_tool の往復が暴走しないようにする安全弁)
# LangGraph は 1 ノードの実行を 1 ステップとして数えるため、
# 「call_model → call_tool」の 1 往復で 2 ステップ消費する。
# 例: 6 なら最大 3 往復分。上限に達すると GraphRecursionError が送出される。
GRAPH_RECURSION_LIMIT = 6

_s3 = boto3.client("s3", region_name=AWS_REGION)
_bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
_dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
_table = _dynamodb.Table(SESSION_HISTORY_TABLE)

_DOC_METADATA_CACHE: Optional[dict] = None
_FAISS_INDEX = None  # faiss.Index (遅延ロード)
_CHUNKS_CACHE: Optional[list] = None  # FAISS の各ベクトルに対応するチャンク情報


def _load_metadata_from_s3() -> dict:
    """S3 からドキュメントのメタデータを読み込む"""
    obj = _s3.get_object(Bucket=S3_BUCKET, Key=METADATA_KEY)
    return json.loads(obj["Body"].read().decode("utf-8"))


def _get_doc_metadata() -> dict:
    """キャッシュ済みメタデータを返す (未取得なら S3 から取得)"""
    global _DOC_METADATA_CACHE
    if _DOC_METADATA_CACHE is None:
        _DOC_METADATA_CACHE = _load_metadata_from_s3()
    return _DOC_METADATA_CACHE


def _load_vector_index() -> None:
    """FAISS インデックスとチャンク情報を S3 からロードしキャッシュする

    Lambda コンテナの再利用中は 1 度だけ実行される
    """
    global _FAISS_INDEX, _CHUNKS_CACHE
    if _FAISS_INDEX is not None and _CHUNKS_CACHE is not None:
        return

    # FAISS は read_index() がファイルパスを要求するため、一旦 /tmp に書き出す。
    with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as fp:
        _s3.download_fileobj(S3_BUCKET, INDEX_KEY, fp)
        tmp_path = fp.name
    _FAISS_INDEX = faiss.read_index(tmp_path)

    obj = _s3.get_object(Bucket=S3_BUCKET, Key=CHUNKS_KEY)
    _CHUNKS_CACHE = json.loads(obj["Body"].read().decode("utf-8"))


def _embed_query(text: str) -> np.ndarray:
    """Bedrock Titan Embeddings v2 で query をベクトル化する

    `normalize=True` で L2 正規化された 1×D のベクトルが返るため、
    そのまま IndexFlatIP の内積でコサイン類似度として比較できる。
    """
    body = json.dumps({"inputText": text, "normalize": True})
    resp = _bedrock.invoke_model(
        modelId=EMBED_MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    payload = json.loads(resp["body"].read())
    return np.array([payload["embedding"]], dtype="float32")


# ======================================================================
# [1] ツール定義
# ======================================================================

@tool
def tool_rag_search(query: str) -> str:
    """ユーザーの質問に関連する社内文書のチャンクを取得する

    質問を Bedrock Titan Embeddings v2 でベクトル化し、FAISS で
    類似度上位 RAG_TOP_K 件のチャンクを返す。各チャンクには出典 (文書ID,
    セクション名) と類似度スコア (cosine similarity) を付与する。
    """
    _load_vector_index()
    qv = _embed_query(query)

    # search は複数クエリ対する検索結果をまとめて返すことが可能なため、
    # scores と indicies が配列で返ってくる（各要素がクエリに関連するドキュメントの配列）。
    # ここでは常に質問は 1 件のみなので、先頭 [0] が「1つめの質問」の結果。
    batch_scores, batch_indices = _FAISS_INDEX.search(qv, RAG_TOP_K)
    top_scores = batch_scores[0]   # 1つめの質問に対する上位 K 件の類似度 (降順)
    top_indices = batch_indices[0]  # 1つめの質問に対する上位 K 件のチャンク番号 (スコアに対応)

    results = []
    # スコア順にドキュメントを処理
    for rank, (idx, score) in enumerate(zip(top_indices, top_scores), start=1):
        # インデックス内のベクトルが RAG_TOP_K 件に満たない場合、
        # 埋められない枠は idx = -1 になるためスキップする。
        if idx < 0:
            continue
        chunk = _CHUNKS_CACHE[idx]
        results.append(
            f"【出典{rank}: {chunk['doc_id']} / {chunk['section']} "
            f"(類似度 {float(score):.3f})】\n{chunk['text']}"
        )

    # インデックスに1件もベクトルが入っていない場合のみチャンクなしになる
    if not results:
        return "関連する社内文書チャンクが見つかりませんでした。"

    header = "以下は質問に関連する社内文書の抜粋です (類似度順)。"
    return header + "\n\n" + "\n\n".join(results)


@tool
def tool_date_calc(
    operation: Literal["add_days", "subtract_days", "days_until"],
    base_date: Optional[str] = None,
    days: Optional[int] = None,
    target_date: Optional[str] = None,
) -> str:
    """日付計算を行う

    - add_days: base_date に days 日を加えた日付を返す (YYYY-MM-DD)
    - subtract_days: base_date から days 日を引いた日付を返す (YYYY-MM-DD)
    - days_until: 今日から target_date までの残り日数を返す (整数)

    base_date 省略時は今日の日付を使用する。
    """
    today = date.today()

    match operation:
        case "add_days":
            if days is None:
                return "エラー: add_days には days が必要です"
            base = datetime.strptime(base_date, "%Y-%m-%d").date() if base_date else today
            result = base + timedelta(days=days)
            return result.strftime("%Y-%m-%d")

        case "subtract_days":
            if days is None:
                return "エラー: subtract_days には days が必要です"
            base = datetime.strptime(base_date, "%Y-%m-%d").date() if base_date else today
            result = base - timedelta(days=days)
            return result.strftime("%Y-%m-%d")

        case "days_until":
            if target_date is None:
                return "エラー: days_until には target_date が必要です"
            target = datetime.strptime(target_date, "%Y-%m-%d").date()
            delta = (target - today).days
            return str(delta)

        # Literal で 3 値に限定しているため通常は到達しないが、
        # 検証を経ない直接呼び出しに備えたフォールバック
        case _:
            return f"エラー: 未知の operation '{operation}'"


@tool
def tool_escalation(question: str, department: str) -> str:
    """社内文書に情報がない質問を担当部署にエスカレーションする

    実際の通知送信は行わず、エスカレーション通知テキストを返す。
    """
    return (
        f"【エスカレーション通知】\n"
        f"以下の質問を {department} へエスカレーションしました:\n"
        f"質問内容: {question}\n"
        f"担当部署の連絡先については社内ポータルをご確認ください。"
    )


TOOLS = [tool_rag_search, tool_date_calc, tool_escalation]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


# ======================================================================
# [2] LangGraph グラフ定義
# ======================================================================
# Jupyter版と同じ構造: State / call_model / call_tool / 条件分岐エッジ

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def _build_system_prompt() -> str:
    """文書メタデータを含むシステムプロンプトを構築する"""
    metadata = _get_doc_metadata()
    today_str = date.today().strftime("%Y-%m-%d")

    doc_list_lines = []
    for doc_id, meta in metadata.items():
        doc_list_lines.append(
            f"- {doc_id} ({meta['department']}): {meta['title']} - {meta['summary']}"
        )
    doc_list = "\n".join(doc_list_lines)

    return f"""あなたは社内ヘルプデスクの担当AIです。社員からの質問に丁寧に回答してください。

本日の日付: {today_str}

利用可能な社内文書一覧:
{doc_list}

ツールの使い分け:
- 社内規程や手続きに関する質問 → tool_rag_search で該当文書を取得
- 日付計算 (締切までの日数、N日後の日付など) → tool_date_calc
- 文書に記載がない質問や専門判断が必要な場合 → 該当する担当部署に tool_escalation

回答は根拠となる文書を明示し、簡潔かつ正確に答えてください。"""


def _get_llm() -> ChatBedrockConverse:
    """LLM インスタンスを返す (ツールバインド済み)

    Bedrock の Anthropic Claude を Converse API 経由で呼ぶ。
    認証は Lambda 実行ロールの IAM 権限 (bedrock:InvokeModel) による。
    """
    llm = ChatBedrockConverse(
        model=CHAT_MODEL_ID,
        region_name=AWS_REGION,
        temperature=0,
        max_tokens=1024,
    )
    return llm.bind_tools(TOOLS)


def call_model(state: AgentState) -> dict:
    """call_model ノード: LLM を呼び出して AIMessage を返す

    LLM の判断により、ツールが必要なら tool_calls を含むメッセージを、
    不要ならそのまま最終回答のメッセージを返す。
    """
    llm = _get_llm()
    messages = list(state["messages"])

    # 先頭に SystemMessage がなければ付与
    # 補足: messages の履歴に SystemMessage は残さないため、
    # 基本的に毎回動的に生成されるシステムプロンプトが使用される。
    # これによってシステムプロンプトに含まれる現在日付や社内文書一覧が常に最新に保たれる。
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=_build_system_prompt())] + messages

    response = llm.invoke(messages)
    return {"messages": [response]}


def call_tool(state: AgentState) -> dict:
    """call_tool ノード: tool_call を実行して ToolMessage を返す"""
    last_message = state["messages"][-1]
    tool_messages = []

    for tc in last_message.tool_calls:
        tool_fn = TOOLS_BY_NAME.get(tc["name"])
        if tool_fn is None:
            result = f"エラー: ツール '{tc['name']}' は存在しません"
        else:
            try:
                result = tool_fn.invoke(tc["args"])
            except Exception as e:  # noqa: BLE001
                result = f"ツール実行エラー: {type(e).__name__}: {e}"

        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tc["id"])
        )

    return {"messages": tool_messages}


def should_continue(state: AgentState) -> str:
    """条件分岐: tool_call があれば call_tool へ、なければ END"""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "call_tool"
    return END


def _build_graph():
    """グラフをコンパイルして返す"""
    graph = StateGraph(AgentState)
    graph.add_node("call_model", call_model)
    graph.add_node("call_tool", call_tool)
    graph.set_entry_point("call_model")
    # call_model 後に条件に基づいて行き先を分岐
    # END の場合はグラフを終了して、最後の call_model が最終回答となる
    graph.add_conditional_edges(
        "call_model",
        should_continue,
        {"call_tool": "call_tool", END: END},
    )
    # call_tool の後は無条件で call_model に戻る
    graph.add_edge("call_tool", "call_model")
    return graph.compile()


# グラフはモジュールロード時に1度だけコンパイル
_GRAPH = _build_graph()


# ======================================================================
# [3] 会話履歴の DynamoDB 永続化
# ======================================================================

# DynamoDB には dict 形式で保存し、読み込み時に BaseMessage に変換する
# role: "human" | "ai" | "tool"  (SystemMessage はシステムプロンプトで毎回生成するので保存しない)

def _message_to_dict(msg: BaseMessage) -> dict:
    """BaseMessage を DynamoDB 保存用の dict に変換"""
    if isinstance(msg, HumanMessage):
        return {"role": "human", "content": msg.content}
    if isinstance(msg, AIMessage):
        # tool_calls も保存する
        return {
            "role": "ai",
            "content": msg.content,
            "tool_calls": msg.tool_calls or [],
        }
    if isinstance(msg, ToolMessage):
        return {
            "role": "tool",
            "content": msg.content,
            "tool_call_id": msg.tool_call_id,
        }
    # SystemMessage は保存しない
    return {}


def _dict_to_message(d: dict) -> Optional[BaseMessage]:
    """dict を BaseMessage に戻す"""
    role = d.get("role")
    if role == "human":
        return HumanMessage(content=d["content"])
    if role == "ai":
        return AIMessage(content=d["content"], tool_calls=d.get("tool_calls", []))
    if role == "tool":
        return ToolMessage(content=d["content"], tool_call_id=d["tool_call_id"])
    return None


def load_history(session_id: str) -> list[BaseMessage]:
    """DynamoDB からセッション履歴を読み込む"""
    resp = _table.get_item(Key={"session_id": session_id})
    item = resp.get("Item")
    if not item:
        return []
    messages = []
    for d in item.get("messages", []):
        msg = _dict_to_message(d)
        if msg is not None:
            messages.append(msg)
    return messages


def save_history(session_id: str, messages: Sequence[BaseMessage]) -> None:
    """DynamoDB にセッション履歴を保存 (TTL 24時間)"""
    ttl = int(time.time()) + 24 * 60 * 60  # 24時間後
    serialized = [
        _message_to_dict(m) for m in messages if not isinstance(m, SystemMessage)
    ]
    # 空 dict を除外
    serialized = [d for d in serialized if d]

    _table.put_item(
        Item={
            "session_id": session_id,
            "messages": serialized,
            "ttl": ttl,
        }
    )


# ======================================================================
# [4] Lambda ハンドラ
# ======================================================================

def _parse_event(event: dict) -> dict:
    """API Gateway Proxy 統合の event から body を取り出す"""
    if "body" in event and event["body"] is not None:
        body = event["body"]
        if isinstance(body, str):
            return json.loads(body)
        return body
    # 直接呼び出し (テスト用) は event 自体がペイロード
    return event


def _response(status: int, body: dict) -> dict:
    """API Gateway Proxy 統合用のレスポンスを返す"""
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json; charset=utf-8",
            # CORS: API Gateway HTTP API 側でも設定しているが、保険として Lambda 側でも返す
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body, ensure_ascii=False),
    }


def lambda_handler(event: dict, context: Any) -> dict:
    """Lambda エントリポイント

    期待する入力:
        {"session_id": "user-001", "message": "有給の申請期限は?"}
    """
    # ブラウザの CORS プリフライト (OPTIONS) は早期リターンで 200 を返す
    method = (
        event.get("requestContext", {}).get("http", {}).get("method")
        or event.get("httpMethod")
        or ""
    ).upper()
    if method == "OPTIONS":
        return _response(200, {})

    try:
        payload = _parse_event(event)
        session_id = payload.get("session_id")
        user_message = payload.get("message")

        if not session_id or not user_message:
            return _response(400, {"error": "session_id と message は必須です"})

        # 1. 履歴を復元
        history = load_history(session_id)

        # 2. 新しい user メッセージを追加
        history.append(HumanMessage(content=user_message))

        # 3. グラフ実行
        #    recursion_limit で call_model ⇄ call_tool の往復回数に上限を設ける。
        #    上限に達した場合は GraphRecursionError となり、暴走を防ぐ。
        try:
            result = _GRAPH.invoke(
                {"messages": history},
                config={"recursion_limit": GRAPH_RECURSION_LIMIT},
            )
        except GraphRecursionError:
            return _response(
                500,
                {"error": "応答の生成に時間がかかりすぎました。質問を分けて試してください。"},
            )
        all_messages = result["messages"]

        # 4. 履歴保存 (SystemMessage は除く)
        save_history(session_id, all_messages)

        # 5. 最後の AIMessage を返却
        #    途中の「ツール呼び出し用 AIMessage」もあるため、末尾から探して最終回答を取り出す
        reply_msg = next(
            (m for m in reversed(all_messages) if isinstance(m, AIMessage)),
            None,
        )
        if reply_msg is None:
            return _response(500, {"error": "AI 応答が生成されませんでした"})

        reply = reply_msg.content
        # Bedrock が content を list で返すケースに対応
        # （1つの回答が複数の文字列に分かれて配列として content に入っているケースがあり得る）
        if isinstance(reply, list):
            reply = "".join(
                c.get("text", "") if isinstance(c, dict) else str(c) for c in reply
            )

        return _response(200, {"session_id": session_id, "reply": reply})

    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        return _response(500, {"error": f"{type(e).__name__}: {e}"})
