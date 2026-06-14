# Autohand Router Monitoring

The router exposes Prometheus text metrics at `/metrics/prometheus`. Scrape that endpoint from each router instance and import the dashboard/rules in this folder.

Recommended starting artifacts:

- `prometheus-alerts.yml`: alerting rules for provider failures, failover pressure, cache efficiency, judge fallback, budget rejection, and auth failures.
- `grafana-dashboard.json`: dashboard panels for traffic, routing health, cache behavior, token/cost usage, selected providers/models, budgets, and classifier adapter health.

Example Prometheus scrape job:

```yaml
scrape_configs:
  - job_name: autohand-router
    metrics_path: /metrics/prometheus
    static_configs:
      - targets:
          - router-1.internal:8080
          - router-2.internal:8080
```

The alert rules assume a Prometheus `job` label of `autohand-router`. If your scrape job uses a different label, update the selectors before loading the rules.

