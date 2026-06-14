# Cloudflare Containers

Cloudflare Workers alone is not the right runtime for this Rust HTTP binary. Use Cloudflare Containers when you want Cloudflare edge routing to a containerized router, or use Cloudflare as an ingress/WAF layer in front of the router hosted on another cloud.

Official references:

- Cloudflare Containers overview: https://developers.cloudflare.com/containers/
- Cloudflare Containers getting started: https://developers.cloudflare.com/containers/get-started/
- Cloudflare Containers architecture: https://developers.cloudflare.com/containers/platform-details/architecture/
- Cloudflare Workers overview: https://developers.cloudflare.com/workers/

## Container Shape

Build the router image as `linux/amd64`, listen on `0.0.0.0:8080`, and keep the router stateless:

```bash
docker buildx build --platform linux/amd64 -t autohand-router:cloudflare .
```

Router config:

```yaml
bind: 0.0.0.0:8080
auth:
  bearer_token_env: [AUTOHAND_ROUTER_TOKEN]
```

## Worker Front Door

The Worker should terminate public traffic, enforce any Cloudflare Access/WAF policy you need, and forward only router API traffic to the Container. Keep provider keys inside the Container environment; do not put provider keys in the Worker.

Example routing sketch:

```ts
export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname === "/health") {
      return env.ROUTER_CONTAINER.fetch(request);
    }
    if (!request.headers.get("authorization")) {
      return new Response("missing authorization", { status: 401 });
    }
    return env.ROUTER_CONTAINER.fetch(request);
  },
};
```

## Open-Weight Topology

Use `kind: vllm` for vLLM servers reachable from the router container:

```yaml
providers:
  - name: vllm
    kind: vllm
    base_url: http://vllm.internal:8000
    chat_path: /v1/chat/completions
    responses_path: /v1/responses
    embeddings_path: /v1/embeddings
    health_path: /health
```

If the GPU inference fleet is not on Cloudflare Containers, host it in AWS, Google Cloud, Azure, CoreWeave, Lambda Labs, or another GPU provider and connect through a private network or authenticated internal endpoint.

## Post-Deploy Checks

```bash
ROUTER_URL=https://router.autohand.ai

curl -fsS "$ROUTER_URL/health"
curl -fsS "$ROUTER_URL/v1/router/providers" \
  -H "authorization: Bearer $AUTOHAND_ROUTER_TOKEN"
```

Before production cutover, export:

```bash
cargo run -- --config docs/examples/router.production.yaml provider-conformance-matrix --output router.provider-matrix.json
cargo run -- load-suite --url "$ROUTER_URL" --output router.load-suite.json
```
