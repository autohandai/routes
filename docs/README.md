# Autohand Router Deployment Docs

This folder contains implementation and hosting examples for running Autohand Router as a production container in front of hosted and open-weight LLM providers.

Start here:

- [Container packaging and runtime config](deployment/container.md)
- [AWS App Runner](deployment/aws-apprunner.md)
- [Google Cloud Run](deployment/google-cloud-run.md)
- [Azure Container Apps](deployment/azure-container-apps.md)
- [Cloudflare Containers](deployment/cloudflare-containers.md)

Production deployment rule: keep provider keys server-side, expose only the router API, and validate every configured provider with `provider-conformance-matrix` before routing live traffic.

