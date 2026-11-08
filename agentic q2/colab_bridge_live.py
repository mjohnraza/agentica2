"""Connects to Google Colab via official colab-mcp.

Extracts the WebSocket URL, displays it for the user, and maintains the connection.
"""

import json
import os
import re
import subprocess
import sys
import threading
import time
import webbrowser


def main():
    print("=" * 70, flush=True)
    print("  Official Google Colab MCP Server Bridge", flush=True)
    print("=" * 70, flush=True)

    proc = subprocess.Popen(
        ["uvx", "--python", "3.13", "git+https://github.com/googlecolab/colab-mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    port = None

    def read_stderr():
        nonlocal port
        for line in proc.stderr:
            # print("  [Server]:", line.strip(), flush=True)
            match = re.search(r"ws://(?:localhost|127\.0\.0\.1):(\d+)", line)
            if match:
                port = match.group(1)
                print(f"[OK] WebSocket Server running on localhost:{port}", flush=True)

    t = threading.Thread(target=read_stderr, daemon=True)
    t.start()

    # 1. Initialize MCP
    init_req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "antigravity-colab", "version": "1.0.0"},
        },
    }
    proc.stdin.write(json.dumps(init_req) + "\n")
    proc.stdin.flush()
    init_res = proc.stdout.readline()
    print("[OK] MCP Protocol Initialized", flush=True)

    # 2. Call open_colab_browser_connection
    print("\nTriggering open_colab_browser_connection...", flush=True)
    call_req = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "open_colab_browser_connection",
            "arguments": {},
        },
    }
    proc.stdin.write(json.dumps(call_req) + "\n")
    proc.stdin.flush()

    time.sleep(2)

    print("\n" + "*" * 70, flush=True)
    print("  GOOGLE COLAB BROWSER CONNECT:", flush=True)
    print("  A new Colab tab is being launched with your session token.", flush=True)
    print("  Please allow the tab to load in your browser.", flush=True)
    print("*" * 70 + "\n", flush=True)

    # Wait for result
    res = proc.stdout.readline()
    print("Connection Result:", res, flush=True)

    # List tools
    tools_req = {"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}}
    proc.stdin.write(json.dumps(tools_req) + "\n")
    proc.stdin.flush()
    tools_res = proc.stdout.readline()
    print("Unlocked Tools:", tools_res, flush=True)


if __name__ == "__main__":
    main()

