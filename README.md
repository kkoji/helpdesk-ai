# 社内ヘルプデスクAI - AWS版

Udemyコース「ゼロから作る社内ヘルプデスクAI」の**最終成果物**。

## アーキテクチャ

```
クライアント (curl などの HTTP クライアント)
       |
       v
  API Gateway (HTTP API, POST /chat + OPTIONS /chat, CORS 有効)
       |
       v
     Lambda (Python 3.11)
       |
       +--> OpenAI gpt-4o-mini               (推論 / tool calling)
       +--> OpenAI text-embedding-3-small    (クエリのベクトル化)
       |
       +--> S3 (metadata.json + vector_index/index.faiss + vector_index/chunks.json)
       |       └─ FAISS インデックス と チャンク本体を取得
       |
       +--> DynamoDB (helpdesk-sessions) (会話履歴・TTL 24h)
```

## エージェント仕様

LangGraph で構築 (`create_react_agent` は**使わず** State/Node/Edge を手組み)。

**3つのツール:**

- `tool_rag_search(query)` - クエリを OpenAI text-embedding-3-small でベクトル化し、FAISS (`IndexFlatIP`) で類似度上位 K 件のチャンクを取得 (各結果に `doc_id` / `section` / コサイン類似度を付与)
- `tool_date_calc(operation, ...)` - `add_days` / `subtract_days` / `days_until`
- `tool_escalation(question, department)` - 担当部署へのエスカレーション通知

## 前提条件

- **AWS CLI 設定済み** (`aws configure` または CloudShell)
- **OpenAI API キー** (チャット / 埋め込みの両方に使用)
  - https://platform.openai.com/api-keys から発行
  - デプロイの前に環境変数として設定: `export OPENAI_API_KEY=sk-...`
- **Python 3.x と pip** (`deploy.sh` が Lambda Layer のクロスビルドと `build_index.py` によるベクトルインデックス構築で使用。CloudShell に標準搭載)

> **チャット LLM・埋め込みとも OpenAI API を直接使用**:
> チャットは gpt-4o-mini、埋め込みは text-embedding-3-small を OpenAI API 経由で呼ぶ。
> Bedrock は使用しないため、AWS 側のモデルアクセス設定は不要。
> API キーはデプロイ時に Lambda の環境変数 `OPENAI_API_KEY` として注入される
> (ソースコードや zip には埋め込まれないので、git にコミットしない限りキーは
> ローカルと Lambda の環境変数にのみ存在する)。
>
> 補足: AWS Academy / Vocareum 等のラボ環境では Bedrock のチャット系モデルが
> SCP やマーケットプレイス権限、Anthropic の Use case フォーム (起動毎にリセット) で
> ブロックされ実用できないため、外部の OpenAI API を直接利用する構成としている。

## デプロイ手順

`deploy.sh` がベクトルインデックスの構築（社内文書のチャンク分割 → OpenAI
text-embedding-3-small でベクトル化 → FAISS インデックス化）から S3 へのアップロード、
Lambda / API Gateway の作成までを一括で行う。
`build_index.py` を手動実行する必要はない。

```bash
cd helpdesk-ai/aws
chmod +x deploy.sh
export OPENAI_API_KEY=sk-...   # OpenAI で発行したキー (チャット・埋め込みに使用)
./deploy.sh
```

`deploy.sh` は内部で Step 3 として `build_index.py` を実行し、
`aws/.build/vector_index/{index.faiss, chunks.json}` を生成して S3 (`vector_index/`) に
アップロードする（実行に必要なホスト側依存も `requirements.txt` から自動で入れる）。

`OPENAI_API_KEY` が未設定だと deploy.sh は早期に終了する。キーは
Lambda の環境変数 `OPENAI_API_KEY` として注入される (ソースコードや zip には
埋め込まれないので、git にコミットしない限りキーはローカルと Lambda の環境変数に
のみ存在する)。Lambda の実行ロールには S3 読み取りと DynamoDB
読み書きのみを付与し、Bedrock 権限は不要。

最後に API エンドポイント URL が表示される。

Lambda Layer は常に `pip --platform manylinux2014_x86_64 --python-version 3.11`
で Lambda (x86_64 / Python 3.11) 互換のクロスビルドを行う。Docker は不要で、
CloudShell のようにディスクが小さくビルドイメージを pull できない環境でも動作する。

ラボ用ショートカット: 事前ビルド済みの Layer zip を S3 に置いてある場合は、
`PREBUILT_LAYER_S3` にその S3 URI を渡すと Layer のクロスビルドをスキップして
ダウンロードのみ行う (受講者がビルドせずに済むようにするための仕組み)。

```bash
PREBUILT_LAYER_S3=s3://udemy-helpdesk-public/layers/layer-v1.zip ./deploy.sh
```

### 制約環境 (AWS Academy / Udemy ラボ等) での注意点

これらの環境では以下の制約があり、スクリプトはそれに対応済み:

| 制約                                                                  | 対応                                                                                                                    |
| --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| CloudShell のディスクが 1GB 制限で Docker イメージが pull できない    | pip クロスビルド方式を採用                                                                                              |
| 設定値・API キーの受け渡し                                            | `deploy.sh` が `aws lambda --environment` で Lambda 環境変数として注入し、`lambda_function.py` は `os.environ` から読む |
| Lambda Layer の直接アップロード上限 (70MB) を超える                   | zip を一旦 S3 に上げてから `publish-layer-version --content S3...` で参照                                               |
| API Gateway HTTP API の OPTIONS リクエストがルート未登録で 400 を返す | `POST /chat` に加えて `OPTIONS /chat` ルートを明示的に作成 + Lambda で OPTIONS を 200 で早期リターン                    |

## 動作確認 (3つのデモシナリオ)

`curl` で API を直接呼び出して確認する。

```bash
# エンドポイントを変数に
ENDPOINT="$(aws apigatewayv2 get-apis --region us-east-1 \
  --query "Items[?Name=='helpdesk-ai-api'].ApiEndpoint | [0]" --output text)"
echo "${ENDPOINT}/chat"
```

`session_id` はリクエストボディで指定する。同じ `session_id` で送れば前の文脈を
引き継ぎ、会話履歴は DynamoDB に 24 時間保持される (別の値にすれば別セッション)。

#### シナリオ1: RAG のみ

```bash
curl -X POST "${ENDPOINT}/chat" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo-001", "message": "有給休暇は何日取れますか?"}'
```

期待動作: `tool_rag_search` が `01_paid_leave.txt` を取得し、付与日数を回答。

#### シナリオ2: RAG → 日付計算

```bash
curl -X POST "${ENDPOINT}/chat" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo-002", "message": "経費精算の締切まであと何日ですか?"}'
```

期待動作:

1. `tool_rag_search` で `02_expense_claim.txt` を取得 → 「毎月15日締切」
2. LLM が「次の15日」の日付を自力で組み立て
3. `tool_date_calc(operation="days_until", target_date="YYYY-MM-15")` で残日数を計算
4. 残日数を回答

#### シナリオ3: 文書なし → エスカレーション

```bash
curl -X POST "${ENDPOINT}/chat" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo-003", "message": "社有車の使用ルールについて教えてください"}'
```

期待動作: `tool_rag_search` が該当文書なしの旨を返却 → LLM が `tool_escalation` で総務部等へエスカレーション通知を生成。

## リソース削除

デモ終了後は `destroy.sh` で deploy.sh が作成したリソースを一括削除できる
(API Gateway / Lambda / Layer 全バージョン / IAM ロール / DynamoDB / S3 を
作成と逆順で削除。冪等なのでリソースが無くてもエラーにならない)。

```bash
cd helpdesk-ai/aws
chmod +x destroy.sh
./destroy.sh            # 確認プロンプトあり (yes と入力)
# FORCE=1 ./destroy.sh  # 確認プロンプトなしで即削除
```

> IAM ロールは AWS Academy 等の制約環境では削除できない場合がある (権限不足)。
> その場合もスクリプトは止まらず、残りのリソース削除を続行する。

<details>
<summary>個別に削除する場合 (手動コマンド)</summary>

```bash
REGION=us-east-1
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Lambda 削除
aws lambda delete-function --function-name helpdesk-ai-handler --region ${REGION}

# API Gateway 削除
API_ID=$(aws apigatewayv2 get-apis --region ${REGION} \
  --query "Items[?Name=='helpdesk-ai-api'].ApiId | [0]" --output text)
aws apigatewayv2 delete-api --api-id ${API_ID} --region ${REGION}

# Lambda Layer 削除 (全バージョン)
for v in $(aws lambda list-layer-versions --layer-name helpdesk-ai-deps \
  --region ${REGION} --query 'LayerVersions[*].Version' --output text); do
  aws lambda delete-layer-version --layer-name helpdesk-ai-deps \
    --version-number $v --region ${REGION}
done

# DynamoDB 削除
aws dynamodb delete-table --table-name helpdesk-sessions --region ${REGION}

# S3 削除 (オブジェクト削除後にバケット削除)
aws s3 rm s3://helpdesk-ai-${ACCOUNT_ID}-${REGION} --recursive
aws s3 rb s3://helpdesk-ai-${ACCOUNT_ID}-${REGION}

# IAM ロール削除 (AWS Academy 等の制約環境では削除不可の場合あり)
aws iam delete-role-policy --role-name helpdesk-ai-lambda-role \
  --policy-name helpdesk-ai-inline 2>/dev/null || true
aws iam detach-role-policy --role-name helpdesk-ai-lambda-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole 2>/dev/null || true
aws iam delete-role --role-name helpdesk-ai-lambda-role 2>/dev/null || true
```

</details>

## スコープ外

- 認証・認可 (誰でも API を呼べる状態)
- 本番向けエラーハンドリング・リトライ
- Gradio / Streamlit 等の WebUI
- 実際のメール送信 / Slack 通知
- 営業日カレンダーを考慮した日数計算
- ハイブリッド検索 (BM25 等の全文検索は併用していない、密ベクトルのみ)
- マネージドベクトル DB (OpenSearch Serverless / Bedrock Knowledge Bases 等) の利用 (FAISS ファイル方式で実装)

## ディレクトリ構成

```
helpdesk-ai/
├── README.md                        (このファイル)
├── data/
│   ├── docs/                        架空の社内規程 8件
│   └── metadata.json                文書メタデータ (タイトル + 要約 + 担当部署)
└── aws/
    ├── build_index.py               文書をチャンク分割し FAISS インデックスを構築 (deploy.sh から実行)
    ├── lambda_function.py           Lambda 本体 (設定値は環境変数 os.environ から取得)
    ├── requirements.txt             Lambda Layer 用依存 (faiss-cpu / langchain-openai / langgraph 等)
    ├── deploy.sh                    デプロイスクリプト (冪等、CORS + OPTIONS 対応)
    └── destroy.sh                   環境一括削除スクリプト (冪等、FORCE=1 で確認スキップ)
```
