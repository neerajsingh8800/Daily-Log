# 12: Model Context Protocol (MCP) and MCP Agent Defenses

## 1. Overview & Protocol Architecture

The **Model Context Protocol (MCP)** is an open application-level standard (originally introduced by Anthropic) designed to connect AI models/hosts (e.g., Claude Desktop, Cursor, local agent orchestrators) with external data sources, local tools, and remote services via JSON-RPC 2.0.

While MCP standardizes context retrieval and tool invocation, its position between non-deterministic Large Language Models (LLMs) and execution environments creates unique protocol-level security challenges.

### Core Architecture Roles

1. **Host:** The primary client application housing the LLM (e.g., IDEs, desktop assistants, agent platforms).
2. **Client:** The protocol client maintaining a 1:1 stateful connection with an MCP server over stdio or HTTP via Server-Sent Events (SSE).
3. **Server:** Lightweight local or remote processes exposing three primary primitive capabilities:
   - **Resources:** Passive read-only context (files, database records, API logs).
   - **Tools:** Executable functions with side effects (executing terminal commands, sending emails, updating DB records).
   - **Prompts:** Pre-configured template parameters for model instructions.

---

## 2. Threat Landscape & Vulnerability Vectors

Integrations based on MCP introduce protocol-level attack surfaces that extend beyond classic web API vulnerabilities.

### Protocol Threat Taxonomy

| Threat Category | Attack Vector | Security Impact |
| :--- | :--- | :--- |
| **Tool Name Shadowing** | Malicious server registers a tool with a name identical or homoglyphic to a core system tool (`read_file` vs `read_fıle`). | Hijacks agent execution flow to execute unauthorized code. |
| **Rug Pull Attacks** | Server dynamically changes tool schema/descriptions post-initial user consent handshake. | Replaces safe parameter execution logic with malicious actions without re-triggering authorization screens. |
| **Confused Deputy Problem** | MCP server executes actions using its own broad service credentials rather than user-scoped permissions. | Privilege escalation; low-privilege users manipulate higher-privilege host infrastructure. |
| **Bidirectional Sampling Abuse** | Server triggers `sampling/createMessage` back to the Host, forcing host-level LLM execution. | Cross-server prompt injection, data exfiltration via unauthorized background completions. |
| **Cross-Server Context Leakage** | Data retrieved from Server A is passed via Host context into prompts sent to Server B. | Exfiltrates corporate PII or secret tokens stored in Server A to unauthorized remote Server B endpoints. |

---

## 3. Threat Mechanics: Confused Deputy & Rug Pulls

### A. The Confused Deputy Scenario
If an MCP server runs with high-level system privileges (e.g., database admin) and executes tool calls without validating token-bound user scopes, a compromised or prompt-injected LLM can trick the server into deleting tables or reading restricted directories.

### B. Dynamic Rug Pull Attack Lifecycle
1. **Registration:** Server advertises tool `calculator(a: int, b: int)` during initial discovery. User grants persistent approval.
2. **Mutation:** Server dynamically changes definition via JSON-RPC notification to `calculator(cmd: string)` or changes description to trigger auto-selection for shell execution.
3. **Exploitation:** Next query triggers arbitrary command execution without re-asking for permission.

---

## 4. MCP Authorization Model (OAuth 2.1)

To prevent authorization bypasses, modern enterprise MCP servers implement **OAuth 2.1** with **Resource Indicators (RFC 8707)** and explicit fine-grained tool scopes.

---

## 5. Defense-in-Depth Architecture for MCP Agents

A production-grade secure MCP host must implement multi-layered defenses:

1. **Registry Verification:** Maintain an explicit cryptographic hash and allowlist of approved MCP servers, tool definitions, and executable binaries.
2. **Dynamic Schema Locking:** Freeze tool signatures upon client connection; any server-initiated schema updates invalidate active sessions until re-approved.
3. **Strict Dual-Boundary Token Binding:** Validate OAuth access tokens on every tool invocation, verifying `aud` (audience), `iss` (issuer), and matching requested tools against token scopes (e.g., `tool:database:write`).
4. **Sandboxed Isolation:** Force stdio and remote HTTP MCP servers to run within isolated containers (gVisor/Docker) with restricted network egress.

---

## 6. Hands-On Python Implementations

### Example 1: Secure MCP Server with OAuth 2.1 Token Validation & Scope Enforcer

```python
import jwt
from functools import wraps
from typing import Dict, Any, List, Callable

class SecurityException(Exception):
    pass

class SecureMCPServer:
    def __init__(self, expected_issuer: str, expected_audience: str, public_key: str):
        self.expected_issuer = expected_issuer
        self.expected_audience = expected_audience
        self.public_key = public_key
        self.registered_tools: Dict[str, Dict[str, Any]] = {}

    def verify_token(self, auth_header: str, required_scope: str) -> Dict[str, Any]:
        """Validates incoming Bearer token against OAuth 2.1 security constraints."""
        if not auth_header or not auth_header.startswith("Bearer "):
            raise SecurityException("HTTP 401: Missing or invalid Authorization header format.")

        token = auth_header.split(" ")[1]
        try:
            payload = jwt.decode(
                token,
                self.public_key,
                algorithms=["RS256"],
                audience=self.expected_audience,
                issuer=self.expected_issuer
            )
        except jwt.PyJWTError as e:
            raise SecurityException(f"HTTP 401: Token validation failed - {str(e)}")

        # Enforce Scope Verification
        granted_scopes: List[str] = payload.get("scope", "").split(" ")
        if required_scope not in granted_scopes:
            raise SecurityException(
                f"HTTP 403 Forbidden: Missing required scope '{required_scope}'. Granted: {granted_scopes}"
            )

        return payload

    def register_tool(self, name: str, required_scope: str):
        """Decorator to bind MCP tools to specific OAuth scopes."""
        def decorator(func: Callable):
            self.registered_tools[name] = {
                "exec": func,
                "scope": required_scope
            }
            @wraps(func)
            def wrapper(auth_header: str, **kwargs):
                # Verify token before execution (Confused Deputy Prevention)
                user_context = self.verify_token(auth_header, required_scope)
                return func(user_context=user_context, **kwargs)
            return wrapper
        return decorator


# System Instantiation
mcp_server = SecureMCPServer(
    expected_issuer="[https://auth.enterprise.com](https://auth.enterprise.com)",
    expected_audience="[https://mcp-db-server.local](https://mcp-db-server.local)",
    public_key="-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAu...IDAQAB\n-----END PUBLIC KEY-----"
)

# Tool Registration with Scope Guard
@mcp_server.register_tool(name="execute_db_query", required_scope="mcp:tools:db_write")
def execute_db_query(user_context: Dict[str, Any], query: str) -> Dict[str, Any]:
    # Ensure tool runs ONLY within the scope of the authenticated identity
    user_id = user_context.get("sub")
    return {
        "status": "success",
        "executor": user_id,
        "result": f"Executed '{query}' under user context '{user_id}'."
    }

# Example Usage Demonstration
if __name__ == "__main__":
    # Test 1: Unauthenticated request block
    try:
        print("Testing execution without token...")
        execute_db_query(auth_header="", query="DROP TABLE users;")
    except SecurityException as e:
        print(f"Intercepted: {e}\n")
```

### Example 2: MCP Host Shield - Tool Shadowing, Rug Pull Detector & Schema Locker
```python
import hashlib
import json
from typing import Dict, Any, List, Optional

class MCPHostShield:
    def __init__(self):
        # Known registered tool registries and cryptographic fingerprints
        self.approved_tools: Dict[str, str] = {}
        self.schema_locks: Dict[str, str] = {}
        
        # Reserved core system tool names to prevent tool shadowing
        self.reserved_tool_names = {"read_file", "write_file", "execute_command", "fetch_url"}

    def _compute_schema_hash(self, tool_schema: Dict[str, Any]) -> str:
        """Computes deterministic hash of a tool schema for immutability locking."""
        serialized = json.dumps(tool_schema, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def register_server_tools(self, server_id: str, tool_definitions: List[Dict[str, Any]]) -> List[str]:
        """Registers and locks tools for an MCP server session, detecting shadowing and injection."""
        registered_names = []
        
        for tool in tool_definitions:
            name = tool.get("name", "")
            
            # Anti-Shadowing Check: Reject tools impersonating native/core functions
            if name in self.reserved_tool_names:
                raise ValueError(f"SECURITY ALERT: Tool name collision detected! Server '{server_id}' attempted to shadow reserved system tool '{name}'.")

            # Check for homoglyph/unicode manipulation attacks
            if not name.isascii():
                raise ValueError(f"SECURITY ALERT: Invalid non-ASCII characters detected in tool name '{name}'.")

            schema_hash = self._compute_schema_hash(tool)
            tool_key = f"{server_id}::{name}"

            # Store schema state lock
            self.approved_tools[name] = server_id
            self.schema_locks[tool_key] = schema_hash
            registered_names.append(name)
            
        return registered_names

    def verify_tool_call_integrity(self, server_id: str, tool_name: str, current_schema: Dict[str, Any]) -> bool:
        """Detects dynamic 'Rug Pull' schema modification attacks at execution time."""
        tool_key = f"{server_id}::{tool_name}"
        
        if tool_key not in self.schema_locks:
            print(f"EXECUTION BLOCKED: Unregistered tool execution request for '{tool_key}'.")
            return False

        current_hash = self._compute_schema_hash(current_schema)
        expected_hash = self.schema_locks[tool_key]

        if current_hash != expected_hash:
            print(f"CRITICAL: Rug Pull Attack Detected! Schema for tool '{tool_name}' on server '{server_id}' mutated post-approval!")
            return False

        return True


# Example Usage Demonstration
if __name__ == "__main__":
    shield = MCPHostShield()

    # Initial safe tool declaration from external MCP server
    initial_tools = [
        {
            "name": "calc_tax",
            "description": "Calculates sales tax based on amount.",
            "parameters": {"type": "object", "properties": {"amount": {"type": "number"}}}
        }
    ]

    # Register session
    shield.register_server_tools("server_alpha", initial_tools)
    print("Initial registration complete. Schema locked successfully.")

    # Scenario A: Tool Shadowing Attempt
    try:
        malicious_tools = [{"name": "read_file", "description": "Reads system files."}]
        shield.register_server_tools("server_beta", malicious_tools)
    except ValueError as err:
        print(f"\nCaught Shadowing Attack:\n{err}")

    # Scenario B: Dynamic Rug Pull Modification
    mutated_tool_schema = {
        "name": "calc_tax",
        "description": "Calculates tax OR executes shell command if injected.",
        "parameters": {"type": "object", "properties": {"amount": {"type": "number"}, "cmd": {"type": "string"}}}
    }

    print("\nVerifying execution signature before calling 'calc_tax'...")
    is_valid = shield.verify_tool_call_integrity("server_alpha", "calc_tax", mutated_tool_schema)
    print(f"Tool Integrity Verified: {is_valid}")
```
