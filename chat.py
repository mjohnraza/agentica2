"""Root forwarder for SaafHawa Agentic Assistant."""

import os
import sys

# Ensure agentic q1 is in path and working directory
agentic_q1_dir = os.path.join(os.path.dirname(__file__), "agentic q1")
os.chdir(agentic_q1_dir)
if agentic_q1_dir not in sys.path:
    sys.path.insert(0, agentic_q1_dir)

import asyncio
from chat import interactive_cli

if __name__ == "__main__":
    asyncio.run(interactive_cli())

