# Azure Container Apps

Azure Container Apps is a managed container target for the router when you want HTTP ingress, revisions, scaling, and secret-backed environment variables without managing Kubernetes directly.

Official references:

- Azure Container Apps environment variables: https://learn.microsoft.com/azure/container-apps/environment-variables
- Azure Container Apps Azure Pipelines deploy task: https://learn.microsoft.com/azure/devops/pipelines/tasks/reference/azure-container-apps-v1
- Azure custom container workflow overview: https://learn.microsoft.com/azure/app-service/tutorial-custom-container

## Image

Build and push the image to Azure Container Registry:

```bash
RESOURCE_GROUP=autohand-prod
LOCATION=eastus
ACR_NAME=autohandrouteracr
IMAGE="$ACR_NAME.azurecr.io/autohand-router:$(git rev-parse --short HEAD)"

az group create --name "$RESOURCE_GROUP" --location "$LOCATION"
az acr create --resource-group "$RESOURCE_GROUP" --name "$ACR_NAME" --sku Standard
az acr login --name "$ACR_NAME"

docker build -t "$IMAGE" .
docker push "$IMAGE"
```

## Deploy

Create a Container Apps environment and deploy the router:

```bash
ENV_NAME=autohand-router-env
APP_NAME=autohand-router

az containerapp env create \
  --name "$ENV_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION"

az containerapp create \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --environment "$ENV_NAME" \
  --image "$IMAGE" \
  --target-port 8080 \
  --ingress external \
  --env-vars \
    AUTOHAND_ROUTER_TOKEN="$AUTOHAND_ROUTER_TOKEN" \
    OPENROUTER_API_KEY="$OPENROUTER_API_KEY"
```

The router config should include:

```yaml
bind: 0.0.0.0:8080
auth:
  bearer_token_env: [AUTOHAND_ROUTER_TOKEN]
```

## Open-Weight Topology

Run vLLM on Azure Kubernetes Service GPU node pools, Azure VMs with GPUs, or another private inference endpoint. Configure the router with the internal service URL:

```yaml
providers:
  - name: vllm
    kind: vllm
    base_url: http://vllm-router.internal:8000
    chat_path: /v1/chat/completions
    responses_path: /v1/responses
    embeddings_path: /v1/embeddings
    health_path: /health
```

Keep the router stateless. Use externalized logs and metrics for fleet-level analysis, and use provider `max_concurrency` plus platform scaling rules to keep local inference queues bounded.

## Post-Deploy Checks

```bash
ROUTER_URL="https://$(az containerapp show \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query properties.configuration.ingress.fqdn \
  --output tsv)"

curl -fsS "$ROUTER_URL/health"
curl -fsS "$ROUTER_URL/metrics/prometheus" \
  -H "authorization: Bearer $AUTOHAND_ROUTER_TOKEN"
```

