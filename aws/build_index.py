"""社内ヘルプデスクAI ベクトルインデックス構築スクリプト

各ドキュメントがマークダウン形式で記述されていることを前提とし、
見出しレベル2（`##`）のセクション単位で分割し、長いセクションは MAX_CHUNK_CHARS 定数に設定した文字数内で再分割する。
Bedrock Titan Embeddings v2 でベクトル化し、FAISS (IndexFlatIP) に格納して保存する。

実行例:
    python build_index.py \\
        --docs-dir ../data/docs \\          # 元となる社内文書が入ったフォルダ（必須）
        --metadata ../data/metadata.json \\ # 各文書の付加情報（doc_id・タイトル等）を定義する JSON（必須）
        --output-dir .build/vector_index \\ # 生成物（index.faiss / chunks.json）の出力先フォルダ（必須）
        --region us-east-1                  # Bedrock(Titan) を呼び出す AWS リージョン（省略時は AWS_REGION → us-east-1）

出力:
    {output-dir}/index.faiss   ... FAISS バイナリインデックス
    {output-dir}/chunks.json   ... 各ベクトルに対応するチャンクのメタ情報と本文
"""

import argparse
import json
import os
from pathlib import Path

import boto3
import faiss
import numpy as np


EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
MAX_CHUNK_CHARS = 400
OVERLAP_CHARS = 50


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """見出し（`## `）単位でドキュメントを分割する

    最初の `## ` が現れるまではヘッダなので捨てる
    """
    sections: list[tuple[str, str]] = []
    current_title: str | None = None
    current_body: list[str] = []

    for line in text.splitlines():
        if line.startswith("## "):
            # 前セクションのデータが存在する場合は、タイトル・本文を sections に追加
            if current_title is not None:
                body = "\n".join(current_body).strip()
                if body:
                    sections.append((current_title, body))
            # 見出し記号 `## ` の3文字を除いたタイトル部分を取得
            current_title = line[3:].strip()
            current_body = []
        elif current_title is not None:
            current_body.append(line)

    # 最後に変数に残っているセクション情報を sections に追加
    if current_title is not None:
        body = "\n".join(current_body).strip()
        if body:
            sections.append((current_title, body))

    return sections


def split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    """長文を max_chars 単位でスライディング分割する."""
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        # チャンクの終端位置（index）
        # 最後のチャンクの場合は、必ずテキストの文字数が終端位置になる
        end = min(start + max_chars, len(text))
        # チャンクの範囲の文字列を切り出して chunks に追加
        chunks.append(text[start:end])
        # テキストの終端まで達したら break
        if end >= len(text):
            break
        # 次のチャンクの開始位置（overlap分だけ戻して文脈の途切れを防ぐ）
        start = end - overlap
    return chunks


def build_chunks_for_doc(doc_id: str, doc_text: str, doc_meta: dict) -> list[dict]:
    """ドキュメントを chunk のリストへ変換する"""
    chunks = []
    # ドキュメントごとに、セクション（見出しレベル2）単位で、タイトル・本文をループ処理
    for section_title, section_body in split_into_sections(doc_text):
        for piece in split_long(section_body, MAX_CHUNK_CHARS, OVERLAP_CHARS):
            # 埋め込み対象テキストには文書タイトル + セクション名も含めて
            # 「## 締切日」のような短い節題でも文脈を伴って検索できるようにする。
            embedding_text = (
                f"# {doc_meta['title']}\n## {section_title}\n{piece}"
            )
            chunks.append({
                "doc_id": doc_id,
                "title": doc_meta["title"],
                "department": doc_meta["department"],
                "section": section_title,
                "text": piece,
                "embedding_text": embedding_text,
            })
    return chunks


def embed_texts(client, texts: list[str]) -> np.ndarray:
    """Bedrock Titan Embeddings v2 でテキストをベクトル化する."""
    vectors = []
    for i, text in enumerate(texts):
        # Titan へのリクエスト本文を JSON 文字列に変換する。
        #   inputText : ベクトル化したいテキスト
        #   normalize : True で長さ1に揃えた(L2正規化済み)ベクトルを返させる。
        #               後段の IndexFlatIP(内積)でそのままコサイン類似度として扱えるようにするため。
        body = json.dumps({"inputText": text, "normalize": True})
        resp = client.invoke_model(
            modelId=EMBED_MODEL_ID,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        payload = json.loads(resp["body"].read())
        vectors.append(payload["embedding"])
        # 10件ずつ embeddings の進捗を表示
        if (i + 1) % 10 == 0 or i == len(texts) - 1:
            print(f"  {i + 1} / {len(texts)} 件埋め込み完了")
    return np.array(vectors, dtype="float32")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--region", default=os.environ.get("AWS_REGION", "us-east-1")
    )
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bedrock = boto3.client("bedrock-runtime", region_name=args.region)

    all_chunks: list[dict] = []
    for doc_id, doc_meta in metadata.items():
        doc_text = (docs_dir / doc_id).read_text(encoding="utf-8")
        all_chunks.extend(build_chunks_for_doc(doc_id, doc_text, doc_meta))
    print(f"チャンク総数: {len(all_chunks)}")

    print(f"Bedrock Titan Embeddings ({EMBED_MODEL_ID}) でベクトル化...")
    vectors = embed_texts(bedrock, [c["embedding_text"] for c in all_chunks])
    print(f"ベクトル形状: {vectors.shape}")

    # ベクトルの次元数を渡して index を初期化
    index = faiss.IndexFlatIP(vectors.shape[1])
    # index にベクトルを追加
    index.add(vectors)

    index_path = output_dir / "index.faiss"
    chunks_path = output_dir / "chunks.json"
    # ベクトルと検索構造を index.faiss として書き出す（本文・出典は別途 chunks.json に保存）
    faiss.write_index(index, str(index_path))

    # 各チャンクの辞書データを chunks.json ファイルとして保存。
    # chunks.json は検索時に「index.faiss が返したベクトル番号」から本文・出典(doc_id/title/
    # department/section)を引くための対応表として使用する。
    # embedding_text はベクトル化専用の内部テキストで、
    # 検索結果としてユーザーに提示する必要がないため出力から除外する。
    chunks_for_output = [
        {k: v for k, v in c.items() if k != "embedding_text"}
        for c in all_chunks
    ]
    chunks_path.write_text(
        json.dumps(chunks_for_output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"出力: {index_path}")
    print(f"出力: {chunks_path}")


if __name__ == "__main__":
    main()
