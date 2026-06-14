# AWS App Runner

AWS App Runner is the simplest AWS target for this router because it runs and scales an HTTP container from a source image, handles load balancing, and supports health checks and environment variables.

Official references:

- App Runner source image services: https://docs.aws.amazon.com/apprunner/latest/dg/service-source-image.html
- App Runner health checks: https://docs.aws.amazon.com/apprunner/latest/dg/manage-configure-healthcheck.html
- App Runner environment variables and secrets: https://docs.aws.amazon.com/apprunner/latest/dg/env-variable.html

## Image

Build and push the image to Amazon ECR:

```bash
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=123456789012
IMAGE="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/autohand-router:$(git rev-parse --short HEAD)"

aws ecr create-repository --repository-name autohand-router || true
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

docker build -t "$IMAGE" .
docker push "$IMAGE"
```

## App Runner Service

Create the service from the ECR image. Configure:

- Port: `8080`
- Health check protocol: `HTTP`
- Health check path: `/health`
- Environment variables:
  - `AUTOHAND_ROUTER_TOKEN`
  - provider keys such as `OPENROUTER_API_KEY`

The router config baked into the image should use:

```yaml
bind: 0.0.0.0:8080
auth:
  bearer_token_env: [AUTOHAND_ROUTER_TOKEN]
```

## Open-Weight Topology

For open-weight models on AWS, run vLLM on ECS, EKS, or EC2 GPU capacity in a private subnet. Point the router provider at the private service name:

```yaml
providers:
  - name: vllm
    kind: vllm
    base_url: http://vllm.service.local:8000
    chat_path: /v1/chat/completions
    responses_path: /v1/responses
    embeddings_path: /v1/embeddings
    health_path: /health
```

Keep App Runner public and the vLLM service private. If App Runner cannot reach the private target in your network layout, place the router itself on ECS/Fargate in the same VPC as the vLLM service.

## Post-Deploy Checks

```bash
ROUTER_URL=https://example.awsapprunner.com

curl -fsS "$ROUTER_URL/health"
curl -fsS "$ROUTER_URL/v1/router/providers" \
  -H "authorization: Bearer $AUTOHAND_ROUTER_TOKEN"
```

