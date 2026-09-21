# 20: Slopsquatting and AI Coding Assistant Supply Chain

## 1. Overview & Threat Surface

As software engineering teams heavily integrate AI coding assistants (e.g., GitHub Copilot, Cursor, Amazon Q) into their development workflows, a novel software supply chain vulnerability has emerged: **Slopsquatting** (also known as *AI Package Hallucination Exploitation*).

Large Language Models (LLMs) trained on code repositories occasionally generate syntax containing non-existent or deprecated third-party packages, libraries, or functions. Attackers exploit this behavior by continuously scanning public codebases and AI suggestions for frequently hallucinated package names, preemptively registering those malicious packages on open repositories (e.g., PyPI, npm, RubyGems, Crates.io). When a developer accepts an AI-generated code snippet and runs `pip install` or `npm install`, they unknowingly pull down a malicious payload.

### Key Supply Chain Vulnerabilities

| Vulnerability Vector | Mechanism | Impact |
| :--- | :--- | :--- |
| **Hallucinated Package Dependency** | Model emits non-existent import statements based on statistical token probabilities. | Developer installs attacker-controlled package from a public registry. |
| **Typosquatted AI imports** | Subtle variations of popular packages suggested during auto-complete (e.g., `reqeusts` vs `requests`). | Unintentional execution of malicious setup scripts (`setup.py` / `postinstall`). |
| **Unsigned Package Pulls** | Installing unverified third-party libraries without checksum pinning or lockfile validation. | Remote Code Execution (RCE) in development environments and CI/CD pipelines. |
| **Transitive AI Dependency Injection** | AI tools suggesting additions to `requirements.txt` or `package.json` directly. | Silent persistent supply chain compromise across the entire engineering organization. |

---

## 2. Threat Mechanics: The Slopsquatting Lifecycle

1. **Reconnaissance & Mining:** Attackers prompt target LLMs with broad programming tasks or monitor public developer forums to identify recurring hallucinated package names.
2. **Registry Squatting:** The attacker registers the hallucinated name (e.g., `python-dateutils-v2` or `react-native-secure-storage-layer`) on PyPI or npm containing a malicious payload in `setup.py` or `package.json` lifecycle hooks (`postinstall`).
3. **Trigger Phase:** A developer prompts an AI assistant:  
   `"Write a Python script to parse large JSON streams efficiently."`
4. **Execution Phase:** The AI assistant outputs:  
   `import super_fast_json_stream`  
   `# Run: pip install super-fast-json-stream`
5. **Compromise:** The developer copies and executes the command, executing arbitrary code during package installation and compromising environment secrets (e.g., AWS keys, `.env` files).

---

## 3. Defense-in-Depth for AI Code Supply Chains

To defend against Slopsquatting, enterprises must implement Abstract Syntax Tree (AST) scanning, private package registry proxies with fallback blocks, and strict lockfile verification in developer environments and CI/CD pipelines.

1. **Pre-Install Registry Verification:** Proxy all package manager requests through a private repository manager (e.g., Nexus, Artifactory) configured to block newly registered or unverified packages.
2. **AST Dependency Inspection:** Parse generated code files for non-standard import statements prior to committing changes to main branches.
3. **Package Age & Reputation Thresholds:** Block installation of packages registered within the last 30 days unless explicitly allowlisted by security teams.
4. **Locked CI/CD Environments:** Disallow unpinned package installs in CI/CD pipelines using strict `--frozen-lockfile` or `--require-hashes` flags.

---

## 4. Hands-On Python Implementations

### Example 1: AST-Based Imports Extractor & Package Hallucination Checker

```python
import ast
import json
import urllib.request
import urllib.error
from typing import List, Dict, Any

class SlopsquattingDetectorException(Exception):
    pass

class ASTPackageChecker:
    def __init__(self, min_package_age_days: int = 30):
        self.min_package_age_days = min_package_age_days
        # Standard built-in Python library modules to ignore
        self.builtin_modules = set([
            "os", "sys", "re", "json", "time", "datetime", "math", "random",
            "typing", "hashlib", "urllib", "subprocess", "ast", "collections"
        ])

    def extract_imports(self, code_snippet: str) -> List[str]:
        """Parses Python code using AST and extracts top-level module imports."""
        imported_modules = set()
        try:
            tree = ast.parse(code_snippet)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported_modules.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imported_modules.add(node.module.split('.')[0])
        except SyntaxError as e:
            raise SlopsquattingDetectorException(f"Syntax error in generated code: {e}")

        return list(imported_modules - self.builtin_modules)

    def verify_pypi_package(self, package_name: str) -> Dict[str, Any]:
        """
        Checks PyPI to verify if an imported module exists and meets safety thresholds.
        """
        url = f"[https://pypi.org/pypi/](https://pypi.org/pypi/){package_name}/json"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Security-AST-Checker/1.0'})
            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    info = data.get("info", {})
                    return {
                        "exists": True,
                        "name": info.get("name"),
                        "summary": info.get("summary"),
                        "version": info.get("version")
                    }
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return {"exists": False, "reason": "Package does not exist (Potential Hallucination)"}
            return {"exists": False, "reason": f"HTTP Error {e.code}"}
        except Exception as e:
            return {"exists": False, "reason": str(e)}

    def inspect_code(self, code_snippet: str) -> List[Dict[str, Any]]:
        """Extracts imports and validates existence against public package index."""
        packages = self.extract_imports(code_snippet)
        results = []

        for pkg in packages:
            status = self.verify_pypi_package(pkg)
            results.append({
                "package": pkg,
                "status": status
            })

        return results

# Example Usage
if __name__ == "__main__":
    checker = ASTPackageChecker()

    ai_generated_code = """
import os
import requests
import non_existent_super_json_parser_v2

def process_data():
    print(os.getpid())
"""

    print("Analyzing AI-Generated Code Snippet...\n")
    findings = checker.inspect_code(ai_generated_code)

    for finding in findings:
        pkg_name = finding["package"]
        status = finding["status"]
        if not status["exists"]:
            print(f"[SECURITY ALERT] Unverified/Hallucinated import detected: '{pkg_name}'")
            print(f" Details: {status['reason']}\n")
        else:
            print(f"[SAFE] Package '{pkg_name}' verified on PyPI. Version: {status['version']}")
```



