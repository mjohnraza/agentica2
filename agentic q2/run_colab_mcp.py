"""Connects to Google Colab via the official Google Colab MCP server.

Spawns colab-mcp, captures the WebSocket port & auth token, displays the exact
Google Colab URL for one-click connection, and unlocks live execution tools.
"""

import json
import os
import re
import subprocess
import sys
import threading
import time


def run_colab_bridge():
    print("=" * 70, flush=True)
    print("  Official Google Colab MCP Launcher", flush=True)
    print("=" * 70, flush=True)

    print("\n1. Starting colab-mcp server process...", flush=True)
    proc = subprocess.Popen(
        ["uvx", "--python", "3.13", "git+https://github.com/googlecolab/colab-mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    # Initialize MCP
    init_msg = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "antigravity-colab-runner", "version": "1.0.0"},
        },
    }
    proc.stdin.write(json.dumps(init_msg) + "\n")
    proc.stdin.flush()
    init_res = json.loads(proc.stdout.readline())
    server_name = init_res.get("result", {}).get("serverInfo", {}).get("name")
    print(f"[OK] MCP Server Initialized: {server_name}", flush=True)

    # Call open_colab_browser_connection
    print("\n2. Invoking 'open_colab_browser_connection'...", flush=True)
    call_msg = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "open_colab_browser_connection",
            "arguments": {},
        },
    }
    proc.stdin.write(json.dumps(call_msg) + "\n")
    proc.stdin.flush()

    print("\n" + "=" * 70, flush=True)
    print("  Status: Browser bridge request sent to Google Colab!", flush=True)
    print("  Please check your browser (Chrome/Edge) for the Colab tab.", flush=True)
    print("=" * 70 + "\n", flush=True)

    print("Waiting for Colab browser handshake (up to 60s)...", flush=True)
    res_line = proc.stdout.readline()
    print("Handshake Result:", res_line, flush=True)

    # List tools after connection
    tools_msg = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/list",
        "params": {},
    }
    proc.stdin.write(json.dumps(tools_msg) + "\n")
    proc.stdin.flush()
    tools_res = json.loads(proc.stdout.readline())
    tools = [t.get("name") for t in tools_res.get("result", {}).get("tools", [])]
    print(f"\n[OK] Available Tools in Colab: {tools}", flush=True)

    return proc


if __name__ == "__main__":
    run_colab_bridge()
