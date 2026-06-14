# Google Cloud Run

Cloud Run is a good target for the router when the upstream providers are hosted APIs or private services reachable through Google Cloud networking. It deploys container images as HTTP services and creates a new revision for each deploy.

Official references:

- Deploy container images to Cloud Run: https://cloud.google.com/run/docs/deploying
- Build source to containers: https://cloud.google.com/run/docs/building/containers
- Cloud Run container runtime contract: https://cloud.google.com/run/docs/container-contract

## Image

Build and push the image to Artifact Registry:

```bash
PROJECT_ID=autohand-prod
REGION=us-central1
REPOSITORY=router
IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/autohand-router:$(git rev-parse --short HEAD)"

gcloud artifacts repositories create "$REPOSITORY" \
  --repository-format=docker \
  --location="$REGION" || true

gcloud auth configure-docker "$REGION-docker.pkg.dev"
docker build -t "$IMAGE" .
docker push "$IMAGE"
```

## Deploy

Deploy the router with port `8080`, request auth enabled at the router layer, and provider keys as environment variables:

```bash
gcloud run deploy autohand-router \
  --image "$IMAGE" \
  --region "$REGION" \
  --port 8080 \
  --allow-unauthenticated \
  --set-env-vars AUTOHAND_ROUTER_TOKEN="$AUTOHAND_ROUTER_TOKEN" \
  --set-env-vars OPENROUTER_API_KEY="$OPENROUTER_API_KEY"
```

The router config should include:

```yaml
bind: 0.0.0.0:8080
auth:
  bearer_token_env: [AUTOHAND_ROUTER_TOKEN]
```

## Open-Weight Topology

Run vLLM on GKE GPU nodes, Compute Engine GPU VMs, or another private inference service. Configure the router with the private URL:

```yaml
providers:
  - name: vllm
    kind: vllm
    base_url: http://vllm.default.svc.cluster.local:8000
    chat_path: /v1/chat/completions
    responses_path: /v1/responses
    embeddings_path: /v1/embeddings
    audio_transcriptions_path: /v1/audio/transcriptions
    audio_translations_path: /v1/audio/translations
    health_path: /health
```

Use Cloud Run VPC connectivity only when the vLLM target is private. For purely hosted providers, keep the router stateless and scale it horizontally.

## Post-Deploy Checks

```bash
ROUTER_URL=$(gcloud run services describe autohand-router \
  --region "$REGION" \
  --format='value(status.url)')

curl -fsS "$ROUTER_URL/health"
curl -fsS "$ROUTER_URL/v1/router/multimodel" \
  -H "authorization: Bearer $AUTOHAND_ROUTER_TOKEN" \
  -H 'content-type: application/json' \
  -d '{"input":"Design an idempotent queue worker","policy":"capability_heavy"}'
```

