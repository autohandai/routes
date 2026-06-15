# Routes Docs

This folder contains usage, operations, monitoring, and deployment examples for Routes.

Start here:

- [Usage guide](usage.md)
- [Container packaging and runtime config](deployment/container.md)
- [AWS App Runner](deployment/aws-apprunner.md)
- [Google Cloud Run](deployment/google-cloud-run.md)
- [Azure Container Apps](deployment/azure-container-apps.md)
- [Cloudflare Containers](deployment/cloudflare-containers.md)
- [Monitoring dashboards and alerts](monitoring/README.md)

Production deployment rule: keep provider keys server-side, expose only the router API, and validate every configured provider with `provider-conformance-matrix` before routing live traffic.
