#!/usr/bin/env bash
#
# 社内ヘルプデスクAI 環境削除スクリプト
#
# deploy.sh で作成した AWS リソースをまとめて削除する。
# 作成と逆の順序で消し、リソースが無い場合もエラーにせず進む (冪等)。
#
# 実行手順:
#   chmod +x destroy.sh
#   ./destroy.sh           # 確認プロンプトあり
#   FORCE=1 ./destroy.sh   # 確認プロンプトなしで即削除
#
# 前提:
#   - aws CLI が認証済み (deploy.sh と同じ環境で実行すること)
#       * Udemy ラボの CloudShell なら認証済みなので `aws configure` は不要。
#       * ローカル PC なら事前に `aws configure` が必要。
#
set -uo pipefail   # 削除は途中失敗しても続行したいので -e は付けない

# ======================================================================
# 設定 (deploy.sh と一致させること)
# ======================================================================
REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
STACK_NAME="helpdesk-ai"

S3_BUCKET="${STACK_NAME}-${ACCOUNT_ID}-${REGION}"
DYNAMODB_TABLE="helpdesk-sessions"
LAMBDA_NAME="helpdesk-ai-handler"
LAMBDA_ROLE_NAME="helpdesk-ai-lambda-role"
LAYER_NAME="helpdesk-ai-deps"
API_NAME="helpdesk-ai-api"

echo "=================================================="
echo "社内ヘルプデスクAI 環境削除"
echo "  Region:       ${REGION}"
echo "  Account:      ${ACCOUNT_ID}"
echo "  S3 Bucket:    ${S3_BUCKET}"
echo "  DynamoDB:     ${DYNAMODB_TABLE}"
echo "  Lambda:       ${LAMBDA_NAME}"
echo "  Layer:        ${LAYER_NAME}"
echo "  API:          ${API_NAME}"
echo "=================================================="

# 確認プロンプト (FORCE=1 でスキップ)
if [ "${FORCE:-0}" != "1" ]; then
  echo ""
  echo "上記リソースを削除します。この操作は取り消せません。"
  read -r -p "本当に削除しますか? (yes と入力): " CONFIRM
  if [ "${CONFIRM}" != "yes" ]; then
    echo "中止しました。"
    exit 0
  fi
fi

# ======================================================================
# Step 1: API Gateway 削除
# ======================================================================
echo ""
echo "[Step 1] API Gateway 削除"
API_ID="$(aws apigatewayv2 get-apis --region "${REGION}" \
  --query "Items[?Name=='${API_NAME}'].ApiId | [0]" --output text 2>/dev/null)"
if [ "${API_ID}" = "None" ] || [ -z "${API_ID}" ]; then
  echo "  対象なし: ${API_NAME}"
else
  aws apigatewayv2 delete-api --api-id "${API_ID}" --region "${REGION}" \
    && echo "  削除: ${API_NAME} (${API_ID})"
fi

# ======================================================================
# Step 2: Lambda 関数削除
# ======================================================================
echo ""
echo "[Step 2] Lambda 関数削除"
if aws lambda get-function --function-name "${LAMBDA_NAME}" --region "${REGION}" >/dev/null 2>&1; then
  aws lambda delete-function --function-name "${LAMBDA_NAME}" --region "${REGION}" \
    && echo "  削除: ${LAMBDA_NAME}"
else
  echo "  対象なし: ${LAMBDA_NAME}"
fi

# ======================================================================
# Step 3: CloudWatch Logs ロググループ削除
# ======================================================================
# Lambda は初回実行時に /aws/lambda/<関数名> のロググループを自動生成する。
# 関数を消しても残るため、明示的に削除する。
echo ""
echo "[Step 3] CloudWatch Logs ロググループ削除"
LOG_GROUP="/aws/lambda/${LAMBDA_NAME}"
if aws logs describe-log-groups --log-group-name-prefix "${LOG_GROUP}" --region "${REGION}" \
     --query "logGroups[?logGroupName=='${LOG_GROUP}']" --output text 2>/dev/null | grep -q .; then
  aws logs delete-log-group --log-group-name "${LOG_GROUP}" --region "${REGION}" \
    && echo "  削除: ${LOG_GROUP}"
else
  echo "  対象なし: ${LOG_GROUP}"
fi

# ======================================================================
# Step 4: Lambda Layer 削除 (全バージョン)
# ======================================================================
echo ""
echo "[Step 4] Lambda Layer 削除"
LAYER_VERSIONS="$(aws lambda list-layer-versions \
  --layer-name "${LAYER_NAME}" --region "${REGION}" \
  --query "LayerVersions[].Version" --output text 2>/dev/null)"
if [ -z "${LAYER_VERSIONS}" ] || [ "${LAYER_VERSIONS}" = "None" ]; then
  echo "  対象なし: ${LAYER_NAME}"
else
  for V in ${LAYER_VERSIONS}; do
    aws lambda delete-layer-version \
      --layer-name "${LAYER_NAME}" --version-number "${V}" --region "${REGION}" \
      && echo "  削除: ${LAYER_NAME}:${V}"
  done
fi

# ======================================================================
# Step 5: IAM ロール削除 (deploy.sh が作ったデフォルトロールのみ)
# ======================================================================
echo ""
echo "[Step 5] IAM ロール削除"
if aws iam get-role --role-name "${LAMBDA_ROLE_NAME}" >/dev/null 2>&1; then
  # ロール削除前に、インラインポリシーとアタッチ済みポリシーを外す必要がある
  aws iam delete-role-policy \
    --role-name "${LAMBDA_ROLE_NAME}" --policy-name helpdesk-ai-inline 2>/dev/null \
    && echo "  インラインポリシー削除: helpdesk-ai-inline"
  aws iam detach-role-policy \
    --role-name "${LAMBDA_ROLE_NAME}" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole 2>/dev/null \
    && echo "  ポリシーデタッチ: AWSLambdaBasicExecutionRole"
  aws iam delete-role --role-name "${LAMBDA_ROLE_NAME}" \
    && echo "  削除: ${LAMBDA_ROLE_NAME}"
else
  echo "  対象なし: ${LAMBDA_ROLE_NAME}"
fi

# ======================================================================
# Step 6: DynamoDB テーブル削除
# ======================================================================
echo ""
echo "[Step 6] DynamoDB テーブル削除"
if aws dynamodb describe-table --table-name "${DYNAMODB_TABLE}" --region "${REGION}" >/dev/null 2>&1; then
  aws dynamodb delete-table --table-name "${DYNAMODB_TABLE}" --region "${REGION}" >/dev/null \
    && echo "  削除: ${DYNAMODB_TABLE} (完全削除まで少し時間がかかる場合あり)"
else
  echo "  対象なし: ${DYNAMODB_TABLE}"
fi

# ======================================================================
# Step 7: S3 バケット削除 (中身を空にしてから削除)
# ======================================================================
echo ""
echo "[Step 7] S3 バケット削除"
if aws s3api head-bucket --bucket "${S3_BUCKET}" 2>/dev/null; then
  echo "  バケット内オブジェクトを削除中..."
  aws s3 rm "s3://${S3_BUCKET}" --recursive --region "${REGION}" --only-show-errors
  aws s3api delete-bucket --bucket "${S3_BUCKET}" --region "${REGION}" \
    && echo "  削除: ${S3_BUCKET}"
else
  echo "  対象なし: ${S3_BUCKET}"
fi

# ======================================================================
# 完了
# ======================================================================
echo ""
echo "=================================================="
echo "削除完了"
echo "=================================================="
