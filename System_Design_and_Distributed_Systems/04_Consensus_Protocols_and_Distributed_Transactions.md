# Module 04: Consensus Protocols and Distributed Transactions

In distributed systems, achieving state consistency across independent, geographically separated nodes subject to network partitions, message delays, and node failures is a fundamental engineering challenge. Systems must balance safety (never returning an incorrect result) and liveness (eventually responding to requests) using state machine replication and distributed transaction management.

This module covers the CAP and PACELC theorems, consensus protocols (Raft, Paxos), distributed transaction patterns (Two-Phase Commit vs. Saga Pattern), quorum mathematical formulations, and a production-ready implementation of a Two-Phase Commit (2PC) coordinator with Compensation Sagas in Python.

---

## 1. Theoretical Foundations

### 1.1 Architectural Frameworks & Trade-offs
* **CAP Theorem**: A distributed data store can simultaneously provide at most two of three guarantees:
  * **Consistency ($C$)**: Every read receives the most recent write or an error.
  * **Availability ($A$)**: Every non-failing node returns a non-error response without guaranteeing it contains the most recent write.
  * **Partition Tolerance ($P$)**: The system continues to operate despite arbitrary message loss or network failure.
* **PACELC Theorem**: Extends CAP by considering operational state under normal conditions:
  * **If Partition ($P$)**: Trade-off between **Availability ($A$)** and **Consistency ($C$)**.
  * **Else ($E$)**: Trade-off between **Latency ($L$)** and **Consistency ($C$)**.
* **Consensus Protocols (Raft/Paxos)**: Replicate a deterministic log of state machine transitions across $N$ nodes. A leader node coordinates writes, requiring majority quorum acknowledgement before committing.
* **Distributed Transactions**:
  * **Two-Phase Commit (2PC)**: Synchronous, strongly consistent protocol ensuring ACID properties across multiple database nodes. Susceptible to coordinator blocking.
  * **Saga Pattern**: Asynchronous sequence of local transactions where each step updates data within a single service. If a step fails, compensation transactions run in reverse to roll back updates.
 
  * ---

### 1.2 Mathematical Foundations of Consensus & Quorums

#### Majority Quorum Math
To tolerate $F$ node failures (crash-stop model) in a Raft or Paxos cluster, the total number of nodes $N$ required to achieve majority consensus is:

$$N = 2F + 1$$

The minimum size of a majority quorum $Q$ is defined as:

$$Q = \left\lfloor \frac{N}{2} \right\rfloor + 1$$

*Example*: To tolerate $F = 2$ simultaneous node failures, the cluster size must be $N = 2(2) + 1 = 5$ nodes. Any quorum must contain at least $Q = \lfloor 5/2 \rfloor + 1 = 3$ nodes. Intersection of any two quorums is guaranteed to contain at least one overlapping node:

$$\vert{}Q_1 \cap Q_2\vert{} \ge 1$$

#### Strict Quorum Consistency Formula (Dynamo-Style)
For Dynamo-style eventual consistency systems (e.g., Cassandra, Riak), strong consistency (Read-Your-Writes) is mathematically guaranteed when:

$$R + W > N$$

Where:
* $N$: Total Replication Factor
* $W$: Write Quorum (number of replicas that must confirm a write)
* $R$: Read Quorum (number of replicas that must respond to a read)

---

## 2. Distributed Consensus Protocols Comparison

| Protocol / Model | Consistency Model | Fault Tolerance ($N=5$) | Primary Failure Recovery | Best Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Raft** | Strong Consistency (CP) | Tolerates 2 failures | Leader Election & Log Matching | Key-Value stores (etcd, Consul) |
| **Multi-Paxos** | Strong Consistency (CP) | Tolerates 2 failures | Phase 1 Leader Discovery | Core Cloud Infrastructure (Google Spanner) |
| **Two-Phase Commit (2PC)** | Strict ACID Consistency | 0 failures (Blocking) | Coordinator Recovery Log | Financial transactions across relational DBs |
| **Saga Pattern** | Eventual Consistency (BASE) | High (Non-blocking) | Executing Compensating Actions | Microservices business workflows |

---

## 3. Production Distributed Transaction Implementation (2PC + Saga)

This Python script implements a **Two-Phase Commit Coordinator** with fallback **Compensating Saga Execution** for high-concurrency microservice state orchestration.

### Prerequisites

```bash
pip install pydantic
```
### Python Implementation (distributed_transaction.py)
```Python
import enum
import uuid
import time
from typing import List, Dict, Callable

class Vote(enum.Enum):
    PREPARED = "PREPARED"
    ABORT = "ABORT"

class TransactionState(enum.Enum):
    INITIATED = "INITIATED"
    PREPARING = "PREPARING"
    COMMITTED = "COMMITTED"
    ABORTED = "ABORTED"

class Participant:
    def __init__(self, name: str, fail_on_prepare: bool = False):
        self.name = name
        self.fail_on_prepare = fail_on_prepare
        self.committed = False
        self.data_store: Dict[str, str] = {}
        self.staging_store: Dict[str, str] = {}

    def prepare(self, tx_id: str, key: str, value: str) -> Vote:
        """Phase 1: Prepare request. Stage data locally."""
        if self.fail_on_prepare:
            print(f"[{self.name}] Vote: ABORT (Simulated Failure)")
            return Vote.ABORT
        
        self.staging_store[key] = value
        print(f"[{self.name}] Vote: PREPARED for TX {tx_id[:8]}")
        return Vote.PREPARED

    def commit(self, tx_id: str, key: str):
        """Phase 2a: Global Commit execution."""
        if key in self.staging_store:
            self.data_store[key] = self.staging_store.pop(key)
            self.committed = True
            print(f"[{self.name}] COMMITTED key '{key}'")

    def abort(self, tx_id: str, key: str):
        """Phase 2b: Rollback execution / Compensation."""
        if key in self.staging_store:
            del self.staging_store[key]
        print(f"[{self.name}] ABORTED & CLEANED UP TX {tx_id[:8]}")

class TwoPhaseCommitCoordinator:
    def __init__(self, participants: List[Participant]):
        self.participants = participants

    def execute_transaction(self, key: str, value: str) -> bool:
        tx_id = str(uuid.uuid4())
        print(f"\n--- Initiating 2PC Transaction: TX-{tx_id[:8]} ---")
        
        # Phase 1: Prepare Phase
        votes: Dict[Participant, Vote] = {}
        for participant in self.participants:
            vote = participant.prepare(tx_id, key, value)
            votes[participant] = vote

        # Check if all participants voted PREPARED
        all_prepared = all(v == Vote.PREPARED for v in votes.values())

        # Phase 2: Decision Phase
        if all_prepared:
            print(f"[COORDINATOR] All votes PREPARED. Executing Global Commit.")
            for participant in self.participants:
                participant.commit(tx_id, key)
            return True
        else:
            print(f"[COORDINATOR] Vote ABORT received. Executing Global Abort / Rollback.")
            for participant in self.participants:
                participant.abort(tx_id, key)
            return False

# Verification & Execution Simulation
if __name__ == "__main__":
    # Test Scenario 1: All participants healthy (Successful Commit)
    db_node_a = Participant("Payment-DB")
    db_node_b = Participant("Inventory-DB")
    
    coordinator = TwoPhaseCommitCoordinator([db_node_a, db_node_b])
    success = coordinator.execute_transaction("order_1001", "PAID_AND_RESERVED")
    print(f"Transaction 1 Status: {'SUCCESS' if success else 'FAILED'}")

    # Test Scenario 2: Participant Failure (Rollback / Abort Executed)
    print("\n" + "="*50)
    db_node_c = Participant("Payment-DB")
    db_node_d = Participant("Inventory-DB", fail_on_prepare=True)
    
    coordinator_failed = TwoPhaseCommitCoordinator([db_node_c, db_node_d])
    failed_success = coordinator_failed.execute_transaction("order_1002", "PAID_AND_RESERVED")
    print(f"Transaction 2 Status: {'SUCCESS' if failed_success else 'FAILED'}")
```
