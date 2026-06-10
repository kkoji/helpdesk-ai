#!/usr/bin/env bash
#
# 社内ヘルプデスクAI デプロイスクリプト
#
# 実行手順:
#   chmod +x deploy.sh
#   ./deploy.sh
#
# 実行環境:
#   このスクリプトは AWS CloudShell で動かす前提。
#   - aws CLI が認証済み (CloudShell が起動時に自動で認証する)
#   - python3 / pip が標準搭載
#
# 前提:
#   - OpenAI API キーを環境変数に設定済み:
#       export OPENAI_API_KEY=sk-...
#     キーは https://platform.openai.com/api-keys から発行できる。
#   - チャット (gpt-4o-mini)・Embeddings (text-embedding-3-small) とも OpenAI API を直接呼ぶ。
#
# ラボ用ショートカット:
#   PREBUILT_LAYER_S3 環境変数に事前ビルド済み Layer zip の S3 URI を渡すと、
#   Step 7 のクロスビルドをスキップして zip をダウンロードのみ行う。
#   例) PREBUILT_LAYER_S3=s3://udemy-helpdesk-public/layers/layer-v1.zip ./deploy.sh
#
set -euo pipefail

# ======================================================================
# 設定
# ======================================================================

# Udemy の AWS ラボ環境は現状 us-east-1 でないと動作しない（権限が付与されていない）ので注意
REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
STACK_NAME="helpdesk-ai"

S3_BUCKET="${STACK_NAME}-${ACCOUNT_ID}-${REGION}"
DOCS_PREFIX="docs/"
METADATA_KEY="metadata.json"
INDEX_KEY="vector_index/index.faiss"
CHUNKS_KEY="vector_index/chunks.json"
DYNAMODB_TABLE="helpdesk-sessions"
LAMBDA_NAME="helpdesk-ai-handler"
LAMBDA_ROLE_NAME="helpdesk-ai-lambda-role"
LAYER_NAME="helpdesk-ai-deps"
API_NAME="helpdesk-ai-api"

CHAT_MODEL_ID="gpt-4o-mini"
EMBED_MODEL_ID="text-embedding-3-small"

# OpenAI API キーは環境変数経由でのみ受け取る (Lambda 環境変数として注入され、git には残さない)
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "エラー: 環境変数 OPENAI_API_KEY が設定されていません。" >&2
  echo "  発行: https://platform.openai.com/api-keys" >&2
  echo "  設定: export OPENAI_API_KEY=sk-..." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${SCRIPT_DIR}/.build"

echo "=================================================="
echo "社内ヘルプデスクAI デプロイ"
echo "  Region:       ${REGION}"
echo "  Account:      ${ACCOUNT_ID}"
echo "  S3 Bucket:    ${S3_BUCKET}"
echo "  DynamoDB:     ${DYNAMODB_TABLE}"
echo "  Lambda:       ${LAMBDA_NAME}"
echo "=================================================="

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

# ======================================================================
# Step 1: S3 バケット作成
# ======================================================================
echo ""
echo "[Step 1] S3 バケット作成"
if aws s3api head-bucket --bucket "${S3_BUCKET}" 2>/dev/null; then
  echo "  既存バケットを使用: ${S3_BUCKET}"
else
  if [ "${REGION}" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "${S3_BUCKET}" --region "${REGION}"
  else
    aws s3api create-bucket \
      --bucket "${S3_BUCKET}" \
      --region "${REGION}" \
      --create-bucket-configuration LocationConstraint="${REGION}"
  fi
  echo "  作成: ${S3_BUCKET}"
fi

# ======================================================================
# Step 2: ドキュメントとメタデータをアップロード
# ======================================================================
echo ""
echo "[Step 2] ドキュメント・メタデータを S3 にアップロード"
aws s3 cp "${PROJECT_ROOT}/data/docs/" "s3://${S3_BUCKET}/${DOCS_PREFIX}" \
  --recursive --region "${REGION}"
aws s3 cp "${PROJECT_ROOT}/data/metadata.json" "s3://${S3_BUCKET}/${METADATA_KEY}" \
  --region "${REGION}"

# ======================================================================
# Step 3: ベクトルインデックスを構築 (build_index.py)
# ======================================================================
# 社内文書をチャンク分割 → OpenAI Embeddings → FAISS インデックス化し、
# ${BUILD_DIR}/vector_index/{index.faiss, chunks.json} を生成する。
echo ""
echo "[Step 3] ベクトルインデックス構築"
INDEX_BUILD_DIR="${BUILD_DIR}/vector_index"
# build_index.py を実行する python3 と同じ環境へ依存パッケージを入れる (python3 -m pip)。
# CloudShell のデフォルト python3 で通常インストールする。
python3 -m pip install -r "${SCRIPT_DIR}/requirements.txt" --quiet
python3 "${SCRIPT_DIR}/build_index.py" \
  --docs-dir "${PROJECT_ROOT}/data/docs" \
  --metadata "${PROJECT_ROOT}/data/metadata.json" \
  --output-dir "${INDEX_BUILD_DIR}"

# ----------------------------------------------------------------------
# Step 4: ベクトルインデックスを S3 にアップロード
# ----------------------------------------------------------------------
# Lambda の tool_rag_search は S3 上の vector_index/{index.faiss, chunks.json}
# を読み込む。これらが無いと RAG 検索が S3 NoSuchKey で失敗し、文書が必要な
# 質問がすべてエスカレーションされてしまう。
#
# インデックスは直前の Step 3 で生成済み (${INDEX_BUILD_DIR})。
echo ""
echo "[Step 4] ベクトルインデックスを S3 にアップロード"
if [ ! -f "${INDEX_BUILD_DIR}/index.faiss" ] || [ ! -f "${INDEX_BUILD_DIR}/chunks.json" ]; then
  echo "エラー: ベクトルインデックスが見つかりません: ${INDEX_BUILD_DIR}/{index.faiss, chunks.json}" >&2
  echo "  Step 3 の build_index.py によるインデックス生成に失敗した可能性があります。" >&2
  exit 1
fi
aws s3 cp "${INDEX_BUILD_DIR}/index.faiss" "s3://${S3_BUCKET}/${INDEX_KEY}" \
  --region "${REGION}"
aws s3 cp "${INDEX_BUILD_DIR}/chunks.json" "s3://${S3_BUCKET}/${CHUNKS_KEY}" \
  --region "${REGION}"
echo "  アップロード完了: s3://${S3_BUCKET}/${INDEX_KEY}"
echo "                    s3://${S3_BUCKET}/${CHUNKS_KEY}"

# ======================================================================
# Step 5: DynamoDB テーブル作成 + TTL 有効化
# ======================================================================
echo ""
echo "[Step 5] DynamoDB テーブル作成"
if aws dynamodb describe-table --table-name "${DYNAMODB_TABLE}" --region "${REGION}" >/dev/null 2>&1; then
  echo "  既存テーブルを使用: ${DYNAMODB_TABLE}"
else
  aws dynamodb create-table \
    --table-name "${DYNAMODB_TABLE}" \
    --attribute-definitions AttributeName=session_id,AttributeType=S \
    --key-schema AttributeName=session_id,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    --region "${REGION}" >/dev/null
  echo "  テーブル作成完了、アクティブになるまで待機..."
  aws dynamodb wait table-exists --table-name "${DYNAMODB_TABLE}" --region "${REGION}"

  aws dynamodb update-time-to-live \
    --table-name "${DYNAMODB_TABLE}" \
    --time-to-live-specification "Enabled=true, AttributeName=ttl" \
    --region "${REGION}" >/dev/null
  echo "  TTL を有効化 (属性名: ttl)"
fi

# ======================================================================
# Step 6: Lambda 実行ロール作成
# ======================================================================
echo ""
echo "[Step 6] Lambda 実行ロール"
ROLE_NEWLY_CREATED=false
if aws iam get-role --role-name "${LAMBDA_ROLE_NAME}" >/dev/null 2>&1; then
  echo "  既存ロールを使用: ${LAMBDA_ROLE_NAME}"
else
  TRUST_POLICY='{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "lambda.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'
  aws iam create-role \
    --role-name "${LAMBDA_ROLE_NAME}" \
    --assume-role-policy-document "${TRUST_POLICY}" >/dev/null
  echo "  ロール作成: ${LAMBDA_ROLE_NAME}"
  ROLE_NEWLY_CREATED=true

  aws iam attach-role-policy \
    --role-name "${LAMBDA_ROLE_NAME}" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
fi

# インラインポリシーは毎回上書きする (権限変更時の追従と、
# 既存ロール時に新しい権限が反映されない問題を避けるため)
INLINE_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject"],
      "Resource": "arn:aws:s3:::${S3_BUCKET}/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:GetItem",
        "dynamodb:PutItem"
      ],
      "Resource": "arn:aws:dynamodb:${REGION}:${ACCOUNT_ID}:table/${DYNAMODB_TABLE}"
    }
  ]
}
EOF
)
aws iam put-role-policy \
  --role-name "${LAMBDA_ROLE_NAME}" \
  --policy-name helpdesk-ai-inline \
  --policy-document "${INLINE_POLICY}"
echo "  権限ポリシーを適用 (S3 読み取り + DynamoDB 読み書き)"

if [ "${ROLE_NEWLY_CREATED}" = "true" ]; then
  echo "  ロール伝播待ち (10秒)..."
  sleep 10
fi

LAMBDA_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${LAMBDA_ROLE_NAME}"

# ======================================================================
# Step 7: Lambda Layer 作成 (依存ライブラリ)
# ======================================================================
echo ""
echo "[Step 7] Lambda Layer 作成"
LAYER_BUILD="${BUILD_DIR}/layer"

PREBUILT_LAYER_S3="${PREBUILT_LAYER_S3:-}"

if [ -n "${PREBUILT_LAYER_S3}" ]; then
  # ラボ用ショートカット: 事前ビルド済み zip をダウンロードしてビルドを丸ごとスキップ
  # 講師側で一度ビルドして公開バケットに置いた zip を、受講者は取得するだけにする
  echo "  事前ビルド済み Layer を取得: ${PREBUILT_LAYER_S3}"
  # --no-sign-request: 公開バケット想定。クレデンシャル経由で 403 になるケースを避ける
  aws s3 cp "${PREBUILT_LAYER_S3}" "${BUILD_DIR}/layer.zip" \
    --region "${REGION}" --no-sign-request --only-show-errors
else
  mkdir -p "${LAYER_BUILD}/python"

  # pip --platform で Lambda 互換のクロスビルドを行う。
  # manylinux2014_x86_64 = Lambda x86_64 と互換のあるバイナリプラットフォーム
  # --python-version 3.11 で Lambda ランタイムに合わせる
  # --only-binary=:all: で wheel のみ使用 (sdist のローカルコンパイルを避ける)
  echo "  pip --platform で Lambda 互換クロスビルド中..."
  PIP_CMD="$(command -v pip3 || command -v pip)"
  "${PIP_CMD}" install \
    -r "${SCRIPT_DIR}/requirements.txt" \
    -t "${LAYER_BUILD}/python" \
    --platform manylinux2014_x86_64 \
    --python-version 3.11 \
    --implementation cp \
    --only-binary=:all: \
    --no-cache-dir \
    --upgrade

  # Layer から不要ファイル (キャッシュ/テスト/メタデータ) を削除してサイズを削減
  # Lambda Layer は解凍後 250MB の制限がある
  echo "  Layer から不要ファイルを削除中..."
  find "${LAYER_BUILD}/python" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
  find "${LAYER_BUILD}/python" -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true
  find "${LAYER_BUILD}/python" -type d -name "test" -exec rm -rf {} + 2>/dev/null || true
  find "${LAYER_BUILD}/python" -type d -name "*.dist-info" -exec rm -rf {} + 2>/dev/null || true
  find "${LAYER_BUILD}/python" -name "*.pyc" -delete 2>/dev/null || true
  find "${LAYER_BUILD}/python" -name "*.pyi" -delete 2>/dev/null || true
  # .so のシンボルを削除してサイズ削減 (失敗しても続行できるよう || true)
  find "${LAYER_BUILD}/python" -name "*.so" -exec strip --strip-unneeded {} \; 2>/dev/null || true

  # サイズチェック (Lambda Layer: 解凍後 250MB 制限)
  LAYER_SIZE_MB=$(du -sm "${LAYER_BUILD}" | cut -f1)
  echo "  Layer 解凍後サイズ: ${LAYER_SIZE_MB}MB"
  if [ "${LAYER_SIZE_MB}" -gt 250 ]; then
    echo "  警告: Layer サイズが 250MB を超えています。デプロイできない可能性があります。"
  fi

  ( cd "${LAYER_BUILD}" && zip -qr "${BUILD_DIR}/layer.zip" python )
fi

# Lambda へ Layer の zip を直接アップロードする方式にはサイズ上限 (zip 圧縮後 約50MB) があり、
# 依存ライブラリを詰めると超えやすい。
# S3 経由なら大きい zip でも扱え、小さくても問題なく動くため、常に S3 にして処理を統一する。
LAYER_S3_KEY="layers/layer-$(date +%s).zip"
echo "  Layer zip を S3 にアップロード: s3://${S3_BUCKET}/${LAYER_S3_KEY}"
aws s3 cp "${BUILD_DIR}/layer.zip" "s3://${S3_BUCKET}/${LAYER_S3_KEY}" \
  --region "${REGION}" --only-show-errors

LAYER_VERSION_ARN="$(aws lambda publish-layer-version \
  --layer-name "${LAYER_NAME}" \
  --content "S3Bucket=${S3_BUCKET},S3Key=${LAYER_S3_KEY}" \
  --compatible-runtimes python3.11 \
  --region "${REGION}" \
  --query LayerVersionArn --output text)"
echo "  Layer 作成: ${LAYER_VERSION_ARN}"

# ======================================================================
# Step 8: Lambda 関数デプロイ
# ======================================================================
echo ""
echo "[Step 8] Lambda 関数デプロイ"
FUNC_ZIP="${BUILD_DIR}/function.zip"

# 設定値は Lambda 環境変数で注入する。
# lambda_function.py は os.environ から読む。
cp "${SCRIPT_DIR}/lambda_function.py" "${BUILD_DIR}/lambda_function.py"
( cd "${BUILD_DIR}" && zip -qr "${FUNC_ZIP}" lambda_function.py )

# Lambda に渡す環境変数 (--environment の Variables) を jq で組み立てる。
# jq を使うことで、API キー等に含まれる特殊文字も安全にエスケープされる。
# 注: AWS_REGION は Lambda ランタイムが自動設定する予約環境変数のため、ここでは指定しない
#     (指定すると create/update が InvalidParameterValueException で失敗する)。
ENV_VARS_JSON="$(jq -n \
  --arg s3_bucket "${S3_BUCKET}" \
  --arg docs_prefix "${DOCS_PREFIX}" \
  --arg metadata_key "${METADATA_KEY}" \
  --arg index_key "${INDEX_KEY}" \
  --arg chunks_key "${CHUNKS_KEY}" \
  --arg session_table "${DYNAMODB_TABLE}" \
  --arg embed_model "${EMBED_MODEL_ID}" \
  --arg chat_model "${CHAT_MODEL_ID}" \
  --arg openai_key "${OPENAI_API_KEY}" \
  '{Variables: {
      S3_BUCKET: $s3_bucket,
      DOCS_PREFIX: $docs_prefix,
      METADATA_KEY: $metadata_key,
      INDEX_KEY: $index_key,
      CHUNKS_KEY: $chunks_key,
      SESSION_HISTORY_TABLE: $session_table,
      EMBED_MODEL_ID: $embed_model,
      CHAT_MODEL_ID: $chat_model,
      OPENAI_API_KEY: $openai_key
  }}')"

if aws lambda get-function --function-name "${LAMBDA_NAME}" --region "${REGION}" >/dev/null 2>&1; then
  echo "  既存関数を更新"
  aws lambda update-function-code \
    --function-name "${LAMBDA_NAME}" \
    --zip-file "fileb://${FUNC_ZIP}" \
    --region "${REGION}" >/dev/null
  aws lambda wait function-updated --function-name "${LAMBDA_NAME}" --region "${REGION}"

  # --role を毎回渡して、既存関数のロールも最新のロールに追従させる
  # --environment で設定値を毎回上書きし、値の変更をデプロイのたびに反映させる
  aws lambda update-function-configuration \
    --function-name "${LAMBDA_NAME}" \
    --role "${LAMBDA_ROLE_ARN}" \
    --layers "${LAYER_VERSION_ARN}" \
    --timeout 60 \
    --memory-size 512 \
    --environment "${ENV_VARS_JSON}" \
    --region "${REGION}" >/dev/null
  aws lambda wait function-updated --function-name "${LAMBDA_NAME}" --region "${REGION}"
else
  echo "  新規関数を作成"
  aws lambda create-function \
    --function-name "${LAMBDA_NAME}" \
    --runtime python3.11 \
    --role "${LAMBDA_ROLE_ARN}" \
    --handler lambda_function.lambda_handler \
    --zip-file "fileb://${FUNC_ZIP}" \
    --layers "${LAYER_VERSION_ARN}" \
    --timeout 60 \
    --memory-size 512 \
    --environment "${ENV_VARS_JSON}" \
    --region "${REGION}" >/dev/null
  aws lambda wait function-active --function-name "${LAMBDA_NAME}" --region "${REGION}"
fi
echo "  Lambda デプロイ完了"

# ======================================================================
# Step 9: API Gateway (HTTP API) 作成
# ======================================================================
echo ""
echo "[Step 9] API Gateway 作成"
API_ID="$(aws apigatewayv2 get-apis --region "${REGION}" \
  --query "Items[?Name=='${API_NAME}'].ApiId | [0]" --output text)"

if [ "${API_ID}" = "None" ] || [ -z "${API_ID}" ]; then
  LAMBDA_ARN="$(aws lambda get-function --function-name "${LAMBDA_NAME}" --region "${REGION}" \
    --query Configuration.FunctionArn --output text)"

  API_ID="$(aws apigatewayv2 create-api \
    --name "${API_NAME}" \
    --protocol-type HTTP \
    --target "${LAMBDA_ARN}" \
    --region "${REGION}" \
    --query ApiId --output text)"
  echo "  API 作成: ${API_ID}"

  aws lambda add-permission \
    --function-name "${LAMBDA_NAME}" \
    --statement-id apigateway-invoke \
    --action lambda:InvokeFunction \
    --principal apigateway.amazonaws.com \
    --source-arn "arn:aws:execute-api:${REGION}:${ACCOUNT_ID}:${API_ID}/*/*" \
    --region "${REGION}" >/dev/null || true
else
  echo "  既存 API を使用: ${API_ID}"
fi

# 統合IDを取得 (新規/既存どちらでも共通)
INTEGRATION_ID="$(aws apigatewayv2 get-integrations --api-id "${API_ID}" --region "${REGION}" \
  --query "Items[0].IntegrationId" --output text)"

# POST /chat と OPTIONS /chat ルートを確保する
# - POST  /chat: 実際のチャット呼び出し
# - OPTIONS /chat: ブラウザの CORS プリフライト用 (Lambda 側で 200 を返す)
#
# API Gateway HTTP API はルートがない HTTP メソッドに対して自動で 200 を返さないため、
# OPTIONS ルートを明示的に作成し、Lambda で 200 を返す。
_ensure_route() {
  local ROUTE_KEY="$1"
  local EXISTING
  EXISTING="$(aws apigatewayv2 get-routes --api-id "${API_ID}" --region "${REGION}" \
    --query "Items[?RouteKey=='${ROUTE_KEY}'].RouteId | [0]" --output text)"
  if [ "${EXISTING}" = "None" ] || [ -z "${EXISTING}" ]; then
    aws apigatewayv2 create-route \
      --api-id "${API_ID}" \
      --route-key "${ROUTE_KEY}" \
      --target "integrations/${INTEGRATION_ID}" \
      --region "${REGION}" >/dev/null
    echo "  ルート作成: ${ROUTE_KEY}"
  else
    echo "  ルート既存: ${ROUTE_KEY}"
  fi
}
_ensure_route "POST /chat"
_ensure_route "OPTIONS /chat"

# CORS 設定 (新規/既存どちらでも毎回適用する)
# デモ用途のため AllowOrigins="*" で公開するが、本番では特定オリジンに絞ること
echo "  CORS 設定を適用"
aws apigatewayv2 update-api \
  --api-id "${API_ID}" \
  --cors-configuration "AllowOrigins=*,AllowMethods=POST,OPTIONS,AllowHeaders=Content-Type" \
  --region "${REGION}" >/dev/null

API_ENDPOINT="$(aws apigatewayv2 get-api --api-id "${API_ID}" --region "${REGION}" \
  --query ApiEndpoint --output text)"

# ======================================================================
# 完了
# ======================================================================
echo ""
echo "=================================================="
echo "デプロイ完了"
echo ""
echo "API エンドポイント:"
echo "  ${API_ENDPOINT}/chat"
echo ""
echo "呼び出し例:"
echo "  curl -X POST ${API_ENDPOINT}/chat \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"session_id\": \"demo-001\", \"message\": \"有給は何日取れますか?\"}'"
echo "=================================================="
