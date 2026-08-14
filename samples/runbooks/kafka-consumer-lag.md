# Kafka Consumer Lag on order-events

## Overview

The `order-projector` consumer group reads from the `order-events` topic (12
partitions) and writes into the read model. Lag above 100k messages means the
order history page is serving stale data.

## Symptoms

- Alert `KafkaConsumerLagHigh` for group `order-projector`
- Order history page shows orders minutes behind reality
- Consumer logs show `CommitFailedException` or repeated rebalances

## Diagnosis

1. Measure current lag per partition:

   ```bash
   kafka-consumer-groups.sh --bootstrap-server $BROKERS \
     --describe --group order-projector
   ```

2. Decide which of the three usual causes applies:

   - Lag concentrated on **one partition** → a hot key or a poison message
   - Lag spread **evenly** across partitions → insufficient consumer throughput
   - Lag climbing with **frequent rebalances** → `max.poll.interval.ms` exceeded

3. If you suspect a poison message, inspect the head of the lagging partition:

   ```bash
   kafka-console-consumer.sh --bootstrap-server $BROKERS \
     --topic order-events --partition 7 --offset 884213 --max-messages 5
   ```

## Remediation

1. **Even lag, healthy consumers** — scale out. Consumer count must not exceed
   partition count (12):

   ```bash
   kubectl -n orders scale deploy/order-projector --replicas=8
   ```

2. **Frequent rebalances** — raise the poll interval and redeploy:

   ```bash
   kubectl -n orders set env deploy/order-projector \
     MAX_POLL_INTERVAL_MS=600000 MAX_POLL_RECORDS=200
   ```

3. **Poison message** — skip it only after capturing the payload for the
   post-incident review:

   ```bash
   kafka-consumer-groups.sh --bootstrap-server $BROKERS \
     --group order-projector --topic order-events:7 \
     --reset-offsets --shift-by 1 --execute
   ```

> **CAUTION**
>
> Offset resets are irreversible and skip data permanently. Capture the message
> first and record the offset in the incident channel.

## Verification

Lag should fall monotonically. If it plateaus after 10 minutes, the consumers
are throughput-bound and scaling out further will not help — escalate.

## Escalation

Page the Streaming Platform on-call if lag exceeds 500k or is still climbing 30
minutes after remediation.
