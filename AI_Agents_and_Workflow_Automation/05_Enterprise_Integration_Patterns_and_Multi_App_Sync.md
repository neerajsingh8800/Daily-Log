# 05: Enterprise Integration Patterns and Multi-App Synchronization

This module covers **Enterprise Integration Patterns (EIP)** in multi-app automation workflows. It details event-driven integration topologies, RAG pipeline automation, data transformation mechanics (Map-Reduce, dynamic pagination, JSONPath querying), and multi-system synchronization in Python.

---

## 1. Enterprise Integration Patterns (EIP) Topology

Multi-app automation involves integrating disparate SaaS platforms, legacy internal SQL databases, and message brokers while maintaining high system resilience and eventual consistency.

### Key Integration Architectural Patterns

1. **Content-Based Router:** Inspects incoming message payloads using JSONPath and dynamically routes the execution path based on payload contents without altering data structure.
2. **Splitter (Iterator):** Accepts a composite array of records (e.g., a batch of 50 orders fetched from Stripe) and breaks them into individual execution threads for downstream processing.
3. **Aggregator (Collector):** Collects individual processing execution threads back into a unified array batch before issuing a single bulk downstream network request.
4. **Message Translator:** Transforms vendor-proprietary JSON schemas (e.g., Salesforce Opportunity Object) into a normalized enterprise standard schema prior to database persistence.

---

## 2. Mathematical Modeling: Map-Reduce Parallel Processing & Throughput

When synchronizing large data sets across enterprise APIs, serial execution introduces severe latency bottlenecks. We model total execution latency ($L_{total}$) using parallel array partitioning (Map-Reduce).

Let $N$ be total records to process, $B$ be chunk batch size per parallel worker thread, $T_{api}$ be average network call latency in seconds, and $P$ be maximum worker concurrency:

$$L_{serial} = N \times T_{api}$$

$$L_{parallel} = \left\lceil \frac{N}{B \times P} \right\rceil \times T_{api}$$

$$\text{Latency Reduction Factor } \Delta L = \frac{L_{serial}}{L_{parallel}} = B \times P$$

* **Rule:** Employing a batch size $B=10$ with concurrency $P=5$ yields a **50x execution latency reduction** over standard serial looping.

---

## 3. Data Transformation Mechanics: JSONPath & Map-Reduce Operations

Handling nested payloads requires precise schema querying and data restructuring:

* **JSONPath Querying:** Extracts specific fields across deeply nested arrays.
  * Query: `$.items[*].price` $\rightarrow$ Returns array of all prices across line items.
* **Map-Reduce Operations:**
  * **Map Step:** Extracts, normalizes, or calculates derived fields across array items.
  * **Reduce Step:** Aggregates mapped values (e.g., summing order totals or computing average sentiment scores).

---

## 4. Production Implementation: Enterprise Multi-App Integration Pipeline in Python

This complete Python script implements an enterprise multi-app integration pipeline. It receives incoming bulk payloads, applies Content-Based Routing, executes array splitting and batching, performs schema transformation, and orchestrates multi-app synchronization (Database, Vector Search, and Notification Systems).

```python
import os
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, EmailStr

# Configure production logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("EnterpriseIntegrationEngine")


# -------------------------------------------------------------------
# 1. Standardized Enterprise Schema Models
# -------------------------------------------------------------------
class RawVendorOrder(BaseModel):
    """Raw payload model from external SaaS vendor (e.g., Stripe/Shopify)."""
    vendor_order_id: str
    customer_email: str
    line_items: List[Dict[str, Any]]
    status: str
    total_amount: float


class NormalizedOrderRecord(BaseModel):
    """Normalized internal enterprise schema."""
    order_id: str
    email: EmailStr
    total_items: int
    gross_value: float
    is_high_value: bool


# -------------------------------------------------------------------
# 2. Integration Pattern Implementations
# -------------------------------------------------------------------
class MessageTranslator:
    """Message Translator Pattern: Converts raw vendor payloads to internal enterprise format."""
    
    @staticmethod
    def translate_vendor_order(vendor_data: RawVendorOrder) -> NormalizedOrderRecord:
        logger.info(f"🔄 Translating payload for Vendor Order ID: {vendor_data.vendor_order_id}")
        
        # Calculate derived metrics (Map-Reduce concept)
        total_quantity = sum(item.get("quantity", 1) for item in vendor_data.line_items)
        high_value_flag = vendor_data.total_amount >= 500.00
        
        return NormalizedOrderRecord(
            order_id=f"INT-{vendor_data.vendor_order_id}",
            email=vendor_data.customer_email,
            total_items=total_quantity,
            gross_value=vendor_data.total_amount,
            is_high_value=high_value_flag
        )


class ContentBasedRouter:
    """Content-Based Router Pattern: Directs execution threads based on message metadata."""

    @staticmethod
    async def route_normalized_record(record: NormalizedOrderRecord) -> Dict[str, Any]:
        logger.info(f"🔀 Routing Order {record.order_id} | Value: ${record.gross_value}")
        
        routes_executed = []
        
        # Route 1: Persistent Enterprise Database Synchronization
        db_status = await ContentBasedRouter._sync_to_database(record)
        routes_executed.append(db_status)

        # Route 2: High-Value VIP Alert Branch
        if record.is_high_value:
            alert_status = await ContentBasedRouter._trigger_vip_slack_alert(record)
            routes_executed.append(alert_status)

        return {
            "order_id": record.order_id,
            "status": "COMPLETED",
            "execution_routes": routes_executed
        }

    @staticmethod
    async def _sync_to_database(record: NormalizedOrderRecord) -> str:
        await asyncio.sleep(0.05)  # Simulate Async DB Write Latency
        logger.info(f"💾 [DB SYNC] Persisted {record.order_id} to PostgreSQL database.")
        return "POSTGRES_UPSERT_SUCCESS"

    @staticmethod
    async def _trigger_vip_slack_alert(record: NormalizedOrderRecord) -> str:
        await asyncio.sleep(0.02)  # Simulate Webhook Network Latency
        logger.warning(f"🚨 [VIP ROUTE] Triggered High-Value Slack Notification for {record.order_id}")
        return "SLACK_VIP_ALERT_SENT"


class SplitterAggregatorEngine:
    """Splitter-Aggregator Pattern: Batches and parallelizes array payload execution."""

    @staticmethod
    async def process_bulk_vendor_ingestion(
        raw_orders: List[RawVendorOrder], batch_size: int = 5, max_concurrency: int = 2
    ) -> List[Dict[str, Any]]:
        logger.info(f"⚡ Starting Splitter-Aggregator Engine for {len(raw_orders)} records.")
        
        # Step 1: Message Translation
        translated_records = [MessageTranslator.translate_vendor_order(order) for order in raw_orders]

        # Step 2: Chunking into Batches (Splitter)
        batches = [
            translated_records[i : i + batch_size]
            for i in range(0, len(translated_records), batch_size)
        ]
        
        results = []
        semaphore = asyncio.Semaphore(max_concurrency)

        async def process_batch(batch: List[NormalizedOrderRecord]):
            async with semaphore:
                tasks = [ContentBasedRouter.route_normalized_record(record) for record in batch]
                return await asyncio.gather(*tasks)

        # Step 3: Parallel Batch Execution (Map)
        batch_tasks = [process_batch(b) for b in batches]
        batch_results = await asyncio.gather(*batch_tasks)

        # Step 4: Aggregation into final result array (Reduce)
        for batch_res in batch_results:
            results.extend(batch_res)

        logger.info(f"✅ Successfully aggregated and completed processing of {len(results)} records.")
        return results


# -------------------------------------------------------------------
# 3. Execution Pipeline Routine
# -------------------------------------------------------------------
async def main():
    # Mocking incoming bulk vendor payload from webhook
    mock_raw_vendor_payload = [
        RawVendorOrder(
            vendor_order_id="9901",
            customer_email="neeraj@example.com",
            line_items=[{"item": "Laptop", "quantity": 1}],
            status="paid",
            total_amount=1200.00  # High Value
        ),
        RawVendorOrder(
            vendor_order_id="9902",
            customer_email="alex@example.com",
            line_items=[{"item": "Mouse", "quantity": 2}],
            status="paid",
            total_amount=45.00  # Standard Value
        ),
        RawVendorOrder(
            vendor_order_id="9903",
            customer_email="sarah@example.com",
            line_items=[{"item": "Monitor", "quantity": 2}, {"item": "Cable", "quantity": 1}],
            status="paid",
            total_amount=650.00  # High Value
        )
    ]

    print("\n--- Starting Enterprise Integration Pipeline Execution ---\n")
    execution_summary = await SplitterAggregatorEngine.process_bulk_vendor_ingestion(
        raw_orders=mock_raw_vendor_payload,
        batch_size=2,
        max_concurrency=2
    )

    print("\n================ INTEGRATION PIPELINE SUMMARY ================")
    print(json.dumps(execution_summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
```
