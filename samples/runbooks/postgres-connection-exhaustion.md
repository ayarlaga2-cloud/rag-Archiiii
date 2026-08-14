# Postgres Connection Exhaustion

## Overview

The `checkout-api` service uses PgBouncer in transaction pooling mode in front
of the primary Postgres cluster. When the pool saturates, new connections are
refused and the service returns 503 to upstream callers.

## Symptoms

- `checkout-api` p99 latency above 3s, climbing
- HTTP 503 rate above 2% on the `/v1/orders` endpoint
- Application logs contain `org.postgresql.util.PSQLException: FATAL: sorry, too many clients already`
- PgBouncer `cl_waiting` metric non-zero and rising

## Impact

Customer-facing. Checkout fails outright for a fraction of requests proportional
to pool saturation. Payment capture is NOT affected — it runs on a separate pool.

## Diagnosis

1. Confirm the pool is actually saturated rather than slow:

   ```bash
   kubectl -n payments exec deploy/pgbouncer -- psql -p 6432 pgbouncer -c "SHOW POOLS;"
   ```

   Look at `cl_active`, `cl_waiting` and `sv_idle`. A non-zero `cl_waiting`
   with `sv_idle = 0` confirms exhaustion.

2. Identify long-running transactions holding connections:

   ```sql
   SELECT pid, now() - xact_start AS duration, state, left(query, 120) AS query
   FROM pg_stat_activity
   WHERE xact_start IS NOT NULL
   ORDER BY duration DESC
   LIMIT 20;
   ```

3. Check whether a deploy correlates with the onset. Compare the alert
   timestamp against the deployment history for `checkout-api`.

> **WARNING**
>
> Do not restart Postgres. Restarting the primary triggers a failover and turns
> a partial outage into a full one.

## Remediation

1. If a single runaway query is holding connections, terminate that backend
   only:

   ```sql
   SELECT pg_terminate_backend(12345);
   ```

2. If saturation is broad rather than one query, raise the pool ceiling
   temporarily:

   ```bash
   kubectl -n payments set env deploy/pgbouncer PGBOUNCER_MAX_CLIENT_CONN=400
   kubectl -n payments rollout status deploy/pgbouncer --timeout=120s
   ```

3. If the onset correlates with a deploy, roll back first and investigate after.
   See the Rollback section below.

4. Once `cl_waiting` returns to zero, revert the temporary ceiling within 24
   hours. Leaving it raised masks the underlying leak.

## Rollback

```bash
kubectl -n payments rollout undo deploy/checkout-api
kubectl -n payments rollout status deploy/checkout-api --timeout=300s
```

Verify the previous revision is serving before declaring the incident mitigated.

## Verification

| Check | Command | Expected |
| --- | --- | --- |
| Pool health | `SHOW POOLS;` | `cl_waiting = 0` |
| Error rate | Grafana `checkout-api 5xx` | below 0.1% for 10 min |
| Latency | Grafana `checkout-api p99` | below 800ms |

## Escalation

If pool saturation persists for more than 20 minutes after remediation, page
the Data Platform on-call via the `#data-platform-oncall` rotation. Include the
output of `SHOW POOLS;` and the top 20 rows from `pg_stat_activity`.
