# Module 13: Service Discovery and Health Monitoring

In dynamic, large-scale distributed systems, instances constantly scale out, crash, or migrate across host IP addresses. Centralized service registries can become single points of failure and bottleneck network topologies. Decentralized **Gossip Protocols** and the **SWIM (Structured Weakness Isolation and Dissemination) Protocol** enable nodes to dynamically discover peers, maintain membership status, and detect cluster failures with $O(1)$ scale expectations per node.

This module covers decentralized membership protocols, indirect probing mechanics, mathematical models for disease propagation and failure detection probabilities, state piggybacking, and a complete SWIM protocol implementation in Python.

---

## 1. Theoretical Foundations

### 1.1 Gossip Mechanics & SWIM Protocol Architecture

* **Gossip (Epidemic) Protocols**:
  * Nodes periodically select a set of random peers and exchange cluster membership updates.
  * Ensures probabilistic eventual consistency across thousands of distributed nodes without requiring central coordination.
  * **Trade-off**: Standard gossip protocols generate $O(N^2)$ aggregate network message overhead per protocol period when scaled linearly.

* **SWIM Protocol (Weakness Isolation & Dissemination)**:
  * Decouples **Failure Detection** from **Membership Dissemination** to achieve constant message overhead $O(1)$ per node period.
  * **Direct Probing**: Node $A$ sends a `PING` to a randomly selected target node $B$. If node $B$ acknowledges (`ACK`), node $B$ is marked healthy.
  * **Indirect Probing (`PING-REQ`)**: If node $B$ fails to reply within a timeout period, node $A$ selects $k$ auxiliary nodes and requests them to ping node $B$ on its behalf (`PING-REQ(B)`). If any auxiliary node receives an ACK, node $B$ is kept marked as healthy.
  * **Suspicion Mechanism**: If indirect probing fails, node $B$ is marked as `SUSPECT` rather than `DEAD`. A bounded timer allows node $B$ to refute the claim before the cluster transitions its status to `DEAD`.


  ---

### 1.2 Mathematical Foundations

#### 1. Epidemic Disease Spreading Math (Gossip Convergence)
In a cluster of $N$ nodes, let $x_t$ be the fraction of uninfected (uninformed) nodes and $y_t$ be the fraction of infected (informed) nodes at time step $t$ ($x_t + y_t = 1$). Assuming each node contacts $\beta$ random nodes per period, the continuous rate of information propagation is modeled by the differential equation:

$$\frac{dy}{dt} = \beta \cdot y(t) \cdot (1 - y(t))$$

Integrating this differential equation yields the logistic convergence curve:

$$y(t) = \frac{y_0 \cdot e^{\beta t}}{1 - y_0 + y_0 \cdot e^{\beta t}}$$

*Convergence Time*: The number of gossip rounds $T$ required to infect $99\%$ of a cluster scaling to $N$ total nodes scales logarithmically:

$$T = O(\log N)$$

#### 2. SWIM Failure Detection Expectation & False Positive Bound
Let $p$ be the independent probability that a network message packet is dropped. 

* The probability $P_{\text{fail}}^{\text{direct}}$ that direct pinging fails due to packet loss is:

$$P_{\text{fail}}^{\text{direct}} = p^2 \quad (\text{Lost PING + Lost ACK})$$

* For $k$ indirect probing paths, the probability $P_{\text{fail}}^{\text{indirect}}$ that all $k$ auxiliary nodes fail to receive an ACK given a healthy target node is:

$$P_{\text{fail}}^{\text{indirect}} = \left( 1 - (1 - p^2)^2 \right)^k$$

* The overall probability $P_{\text{false}}$ of misidentifying a healthy node as suspect drops exponentially as $k$ increases:

$$P_{\text{false}} = p^2 \cdot \left( 1 - (1 - p^2)^2 \right)^k$$

---

## 2. Service Discovery Architectures Comparison

| Architecture | Failure Detection Speed | Network Complexity | Scalability Limit | Single Point of Failure |
| :--- | :--- | :--- | :--- | :--- |
| **Centralized Registry (Consul/etcd)** | Fast ($1-3\text{s}$) | $O(N)$ HTTP Heartbeats | $\sim 5,000 - 10,000$ Nodes | Yes (if Raft Quorum fails) |
| **Standard Gossip (Serf/Memberlist)** | Moderate ($3-5\text{s}$) | $O(N \log N)$ | $\sim 100,000+$ Nodes | None (Fully Decentralized) |
| **SWIM Protocol (with Suspicion)** | Fast ($2-4\text{s}$) | $O(1)$ constant per node | $\sim 100,000+$ Nodes | None (Fully Decentralized) |

---

## 3. Production SWIM Protocol Implementation

This Python script implements a **SWIM Membership Engine** featuring direct pinging, $k$-indirect probing (`PING-REQ`), piggybacked node state dissemination, and asynchronous state transitions (`ALIVE`, `SUSPECT`, `DEAD`).

### Python Implementation (`swim_protocol.py`)

```python
import random
import time
import threading
from enum import Enum
from typing import Dict, List, Optional


class NodeStatus(Enum):
    ALIVE = "ALIVE"
    SUSPECT = "SUSPECT"
    DEAD = "DEAD"


class MemberState:
    def __init__(self, node_id: str, address: str, status: NodeStatus = NodeStatus.ALIVE, incarnation: int = 0):
        self.node_id = node_id
        self.address = address
        self.status = status
        self.incarnation = incarnation
        self.last_state_change = time.time()


class SWIMNode:
    def __init__(self, node_id: str, address: str, k_indirect: int = 2, suspect_timeout: float = 3.0):
        self.node_id = node_id
        self.address = address
        self.k_indirect = k_indirect
        self.suspect_timeout = suspect_timeout
        
        # Cluster Membership Table: node_id -> MemberState
        self.members: Dict[str, MemberState] = {}
        # Self registration
        self.members[self.node_id] = MemberState(node_id, address, NodeStatus.ALIVE, incarnation=0)
        
        self.lock = threading.Lock()
        self.is_running = False
        self._loop_thread: Optional[threading.Thread] = None

    def add_peer(self, node_id: str, address: str):
        with self.lock:
            if node_id not in self.members:
                self.members[node_id] = MemberState(node_id, address, NodeStatus.ALIVE)

    def receive_ping(self, sender_id: str) -> bool:
        """Simulates responding to direct PING request."""
        with self.lock:
            sender = self.members.get(sender_id)
            return sender is None or sender.status != NodeStatus.DEAD

    def receive_ping_req(self, requester_id: str, target_id: str, cluster_nodes: Dict[str, 'SWIMNode']) -> bool:
        """Simulates receiving indirect PING-REQ and forwarding request to target node."""
        target_node = cluster_nodes.get(target_id)
        if target_node:
            return target_node.receive_ping(self.node_id)
        return False

    def update_member_state(self, node_id: str, status: NodeStatus, incarnation: int):
        """Processes piggybacked membership updates using Incarnation rules."""
        with self.lock:
            if node_id not in self.members:
                self.members[node_id] = MemberState(node_id, "", status, incarnation)
                return

            current = self.members[node_id]
            
            # Incarnation Override Rules
            if incarnation > current.incarnation:
                current.incarnation = incarnation
                current.status = status
                current.last_state_change = time.time()
            elif incarnation == current.incarnation:
                if current.status == NodeStatus.ALIVE and status == NodeStatus.SUSPECT:
                    current.status = NodeStatus.SUSPECT
                    current.last_state_change = time.time()
                elif current.status == NodeStatus.SUSPECT and status == NodeStatus.DEAD:
                    current.status = NodeStatus.DEAD
                    current.last_state_change = time.time()

    def probe_cycle(self, cluster_nodes: Dict[str, 'SWIMNode']):
        """Single SWIM Protocol Failure Detection Round."""
        with self.lock:
            eligible_targets = [nid for nid, m in self.members.items() if nid != self.node_id and m.status != NodeStatus.DEAD]

        if not eligible_targets:
            return

        target_id = random.choice(eligible_targets)
        target_node = cluster_nodes.get(target_id)

        # Step 1: Direct Probing
        direct_ack = target_node.receive_ping(self.node_id) if target_node else False

        if direct_ack:
            return  # Target is healthy

        # Step 2: Indirect Probing via k Auxiliary Nodes
        with self.lock:
            auxiliary_candidates = [nid for nid in self.members.keys() if nid not in (self.node_id, target_id) and self.members[nid].status == NodeStatus.ALIVE]

        k_aux = min(len(auxiliary_candidates), self.k_indirect)
        selected_aux = random.sample(auxiliary_candidates, k_aux) if k_aux > 0 else []

        indirect_ack = False
        for aux_id in selected_aux:
            aux_node = cluster_nodes.get(aux_id)
            if aux_node and aux_node.receive_ping_req(self.node_id, target_id, cluster_nodes):
                indirect_ack = True
                break

        # Step 3: Transition to SUSPECT if Indirect Probing Fails
        if not indirect_ack:
            print(f"[{self.node_id}] Probing failed for '{target_id}'. Marking SUSPECT.")
            current_inc = self.members[target_id].incarnation
            self.update_member_state(target_id, NodeStatus.SUSPECT, current_inc)

    def check_suspicion_timeouts(self):
        """Converts long-standing SUSPECT nodes to DEAD."""
        now = time.time()
        with self.lock:
            for nid, member in self.members.items():
                if member.status == NodeStatus.SUSPECT and (now - member.last_state_change) > self.suspect_timeout:
                    print(f"[{self.node_id}] Suspicion timer expired for '{nid}'. Marking DEAD.")
                    member.status = NodeStatus.DEAD
                    member.last_state_change = now


# -------------------------------------------------------------------
# VERIFICATION / SIMULATION RUNNER
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("--- Initializing 4-Node SWIM Protocol Cluster ---")
    
    # Create simulated network nodes
    cluster: Dict[str, SWIMNode] = {
        "node_1": SWIMNode("node_1", "192.168.1.1:8000"),
        "node_2": SWIMNode("node_2", "192.168.1.2:8000"),
        "node_3": SWIMNode("node_3", "192.168.1.3:8000"),
        "node_4": SWIMNode("node_4", "192.168.1.4:8000"),
    }

    # Interconnect full peer knowledge
    for n1 in cluster.values():
        for n2 in cluster.values():
            if n1.node_id != n2.node_id:
                n1.add_peer(n2.node_id, n2.address)

    print("Initial Cluster Status: All 4 Nodes ALIVE.")

    # Simulate node_4 hard crashing (unresponsive to pings)
    print("\n[SIMULATION EVENT] 'node_4' crashes completely.")
    cluster["node_4"].receive_ping = lambda sender_id: False

    # Execute SWIM Protocol rounds from node_1
    print("\n--- Round 1: Node 1 Executes Probe Cycle ---")
    for _ in range(5):
        cluster["node_1"].probe_cycle(cluster)

    # Allow time for suspicion window to elapse
    time.sleep(3.5)
    cluster["node_1"].check_suspicion_timeouts()

    print("\n--- Final Membership Table Snapshot for Node 1 ---")
    for nid, m in cluster["node_1"].members.items():
        print(f"Node: {nid} | Status: {m.status.value} | Incarnation: {m.incarnation}")
```

## 4. Operational Best Practices

*  Piggyback Dissemination: Combine failure detection packets with membership updates in UDP payloads to avoid issuing dedicated gossip messages.
*  Incarnation Counter Refutation: Allow suspected nodes to refute false suspicion state transitions by broadcasting an ALIVE state payload carrying an incremented incarnation_number.
*  UDP Ping with TCP Fallback: Use lightweight UDP sockets for high-frequency direct and indirect probes, falling back to a TCP handshake if UDP packets are dropped by network firewalls.
