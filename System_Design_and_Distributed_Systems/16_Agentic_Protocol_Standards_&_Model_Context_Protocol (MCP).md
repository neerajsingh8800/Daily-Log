# Module 16: Agentic Protocol Standards & Model Context Protocol (MCP)

As AI deployments transition from single-prompt interactions to multi-agent ecosystems, integrating Large Language Models (LLMs) with external databases, APIs, and tools via custom, proprietary wrappers becomes unsustainable. Anthropic's **Model Context Protocol (MCP)** establishes an open, standardized client-server protocol using **JSON-RPC 2.0** to decouple LLM runtime client applications from remote resource providers and tool execution environments.

This module covers the architecture of the Model Context Protocol (MCP), dynamic resource discovery, client-server handshake states, security isolation mechanisms, and a full Python implementation of an MCP-compliant Server and Client architecture.

---

## 1. Theoretical Foundations

### 1.1 MCP Protocol Architecture & State Machine

* **MCP Client-Server Decoupling**:
  * **MCP Host / Client**: The orchestrator (e.g., Claude Desktop, IDE extensions, custom AI Agent runtimes) that manages model execution and context windows.
  * **MCP Server**: A lightweight service exposing **Tools** (executable functions), **Resources** (readable data/file streams), and **Prompts** (pre-configured context templates).
  * Communication occurs bi-directionally over standard transports: **stdio** (standard input/output for local processes) or **SSE** (Server-Sent Events over HTTP for remote microservices).

* **Protocol Primitives**:
  1. **Prompts**: Server-defined prompt templates (`prompts/list`, `prompts/get`) with user argument schemas.
  2. **Resources**: Read-only context URIs (`resources/list`, `resources/read`, e.g., `file:///logs/app.log` or `db://users/profile`).
  3. **Tools**: Executable functions (`tools/list`, `tools/call`) carrying JSON Schema parameter specifications that the LLM invokes directly.

---

### 1.2 Mathematical Foundations & Message Formats

#### 1. JSON-RPC 2.0 Protocol Envelope
All MCP messages adhere to JSON-RPC 2.0 framing. An invocation request containing payload arguments is formatted as:

$$\text{Request}(M) = \{ \text{"jsonrpc"}: \text{"2.0"}, \; \text{"id"}: k, \; \text{"method"}: M, \; \text{"params"}: P \}$$

The server response maps back to request ID $k$:

$$\text{Response}(k) = \{ \text{"jsonrpc"}: \text{"2.0"}, \; \text{"id"}: k, \; \text{"result"}: R \}$$

Where error conditions yield $\text{"error"}: \{ \text{"code"}: c, \text{"message"}: m \}$.

#### 2. Entropy / Context Window Overhead Formula
Let $T_{\text{schema}}$ be the total token count of all tools exposed by an MCP Server $S$. If an MCP Client connects to $N$ servers, the base system prompt token budget $B_{\text{system}}$ scales additively:

$$B_{\text{system}} = B_{\text{base}} + \sum_{i=1}^{N} \sum_{t \in \text{Tools}(S_i)} \text{Tokens}(\text{JSONSchema}(t))$$

Dynamic tool discovery filters tools dynamically using embedding search to ensure $B_{\text{system}} \le B_{\text{max\_context}}$.

---

## 2. Agent Communication Architectures Comparison

| Dimension / Metric | Custom API Wrappers | OpenAPI / REST Schema | Model Context Protocol (MCP) |
| :--- | :--- | :--- | :--- |
| **Transport Protocol** | In-process Python / HTTP | HTTP / REST | `stdio` (Local) / SSE over HTTP (Remote) |
| **Standardization** | Proprietary per framework | HTTP Specification | Open Standard (JSON-RPC 2.0) |
| **Dynamic Discovery** | Hardcoded | Static OpenAPI spec | Dynamic capabilities negotiation (`tools/list`) |
| **Security Boundary** | High risk (direct code imports) | Token-based auth | Process boundary isolation via stdio/sandbox |
| **Interoperability** | Low (Framework lock-in) | Moderate | Universal across any MCP-compliant Client |

---

## 3. Production MCP Server & Client Engine Implementation

This Python script implements a **Model Context Protocol (MCP) Server** exposing tools and resources, alongside an **MCP Client** managing the JSON-RPC 2.0 handshake and execution loop over `stdio`.

### Prerequisites

```bash
pip install pydantic
```

### Python Implementation (mcp_agent_protocol.py)
```Python
import sys
import json
import asyncio
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


# -------------------------------------------------------------------
# 1. JSON-RPC 2.0 MESSAGING STRUCTURES
# -------------------------------------------------------------------
class JSONRPCRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: Optional[int] = None
    method: str
    params: Optional[Dict[str, Any]] = None


class JSONRPCResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: Optional[int] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None


# -------------------------------------------------------------------
# 2. MCP SERVER IMPLEMENTATION
# -------------------------------------------------------------------
class MCPServer:
    def __init__(self, server_name: str, server_version: str):
        self.server_name = server_name
        self.server_version = server_version
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.resources: Dict[str, Dict[str, Any]] = {}
        
        # Register default server features
        self._register_default_primitives()

    def _register_default_primitives(self):
        # Register a sample tool
        self.register_tool(
            name="query_inventory",
            description="Query product inventory stock levels by SKU ID.",
            input_schema={
                "type": "object",
                "properties": {
                    "sku_id": {"type": "string", "description": "The target product SKU ID"}
                },
                "required": ["sku_id"]
            },
            handler=self._handle_query_inventory
        )

    def register_tool(self, name: str, description: str, input_schema: Dict[str, Any], handler: Any):
        self.tools[name] = {
            "name": name,
            "description": description,
            "inputSchema": input_schema,
            "handler": handler
        }

    async def _handle_query_inventory(self, sku_id: str) -> str:
        mock_db = {"SKU-101": 150, "SKU-202": 0, "SKU-303": 42}
        stock = mock_db.get(sku_id.upper(), -1)
        if stock == -1:
            return f"SKU '{sku_id}' not found in inventory."
        return f"Stock level for {sku_id}: {stock} units available."

    async def handle_request(self, raw_message: str) -> str:
        try:
            req_data = json.loads(raw_message)
            req = JSONRPCRequest(**req_data)
        except Exception as e:
            return JSONRPCResponse(id=None, error={"code": -32700, "message": f"Parse error: {str(e)}"}).model_dump_json()

        # Route JSON-RPC Methods
        if req.method == "initialize":
            res = {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}, "resources": {}},
                "serverInfo": {"name": self.server_name, "version": self.server_version}
            }
            return JSONRPCResponse(id=req.id, result=res).model_dump_json()

        elif req.method == "tools/list":
            tool_list = [
                {"name": t["name"], "description": t["description"], "inputSchema": t["inputSchema"]}
                for t in self.tools.values()
            ]
            return JSONRPCResponse(id=req.id, result={"tools": tool_list}).model_dump_json()

        elif req.method == "tools/call":
            params = req.params or {}
            tool_name = params.get("name")
            args = params.get("arguments", {})

            if tool_name not in self.tools:
                return JSONRPCResponse(id=req.id, error={"code": -32601, "message": f"Tool '{tool_name}' not found."}).model_dump_json()

            handler = self.tools[tool_name]["handler"]
            output = await handler(**args)
            
            # MCP Tool Response Protocol
            content = [{"type": "text", "text": str(output)}]
            return JSONRPCResponse(id=req.id, result={"content": content}).model_dump_json()

        return JSONRPCResponse(id=req.id, error={"code": -32601, "message": "Method not found"}).model_dump_json()


# -------------------------------------------------------------------
# 3. MCP CLIENT (HOST SIMULATION)
# -------------------------------------------------------------------
class MCPClient:
    def __init__(self, server: MCPServer):
        self.server = server
        self.msg_id = 0

    def _next_id(self) -> int:
        self.msg_id += 1
        return self.msg_id

    async def send_rpc(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        req = JSONRPCRequest(id=self._next_id(), method=method, params=params)
        raw_req = req.model_dump_json()
        
        # Simulate transport over stdio / memory channel
        raw_resp = await self.server.handle_request(raw_req)
        resp = json.loads(raw_resp)
        
        if "error" in resp and resp["error"]:
            raise RuntimeError(f"MCP RPC Error ({resp['error']['code']}): {resp['error']['message']}")
        return resp.get("result", {})


# -------------------------------------------------------------------
# SIMULATION / VERIFICATION RUNNER
# -------------------------------------------------------------------
async def main():
    print("--- 1. Initializing MCP Server & Client Connection ---")
    server = MCPServer(server_name="InventoryMCPServer", server_version="1.0.0")
    client = MCPClient(server)

    # Handshake / Initialization Phase
    init_res = await client.send_rpc("initialize")
    print(f"Connected to MCP Server: {init_res['serverInfo']['name']} (v{init_res['serverInfo']['version']})")

    print("\n--- 2. Discovering Server Tools (`tools/list`) ---")
    tools_res = await client.send_rpc("tools/list")
    for tool in tools_res.get("tools", []):
        print(f"Tool Found: '{tool['name']}' | Description: {tool['description']}")

    print("\n--- 3. Invoking Tool (`tools/call`) via MCP Protocol ---")
    call_args = {"name": "query_inventory", "arguments": {"sku_id": "SKU-303"}}
    call_res = await client.send_rpc("tools/call", params=call_args)
    
    result_text = call_res["content"][0]["text"]
    print(f"Server Execution Result: '{result_text}'")


if __name__ == "__main__":
    asyncio.run(main())
```

## 4. Operational Best Practices

* Enforce Out-of-Process Isolation: Always run MCP Servers in separate subprocesses or isolated containers (e.g., Docker) communicating via stdio or SSE to prevent arbitrary code execution on the main host.
* Filter Schema Size: Implement tool retrieval filtering (tools/list) when connecting to servers with extensive tool libraries to avoid overflowing the model's system context window budget.
* Validate JSON Schemas: Rigorously validate both incoming arguments against the tool's inputSchema on the server and returned content types on the client before injecting responses into the LLM prompt stream.
