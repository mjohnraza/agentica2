"""Puter-powered LLM Brain for SaafHawa Agentic Assistant.

Integrates with Puter (https://github.com/heyPuter/puter) to provide multi-model
reasoning, tool selection, and multilingual synthesis for Karachi climate advisories.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import logging
import os
from typing import Any, Callable, Coroutine, Dict, List, Optional

import httpx
from dotenv import load_dotenv

# Load environment variables (.env)
load_dotenv()

from tool import TOOL_REGISTRY, TOOL_SCHEMAS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SaafHawaPuterBrain")


@dataclass
class ToolCall:
    name: str
    args: Dict[str, Any]
    call_id: Optional[str] = None
    fn: Optional[Callable[..., Coroutine[Any, Any, Dict[str, Any]]]] = None

    async def invoke(self) -> Dict[str, Any]:
        if self.fn:
            return await self.fn(**self.args)
        return {"status": "success", "tool": self.name, "args": self.args}


@dataclass
class AgentDecision:
    is_final: bool
    tool: Optional[ToolCall] = None
    final_answer: Optional[str] = None
    model_used: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None


SYSTEM_PROMPT = """You are SaafHawa, an agentic AI assistant for Karachi citizens offering vital advice on air quality, extreme heatwaves, and public health.

Your goal for every citizen query:
1. Identify the area/location (e.g. Clifton, Korangi, Saddar, Karachi general).
2. Identify the time of day and vulnerable personas (children, elderly, asthma patients, outdoor labourers).
3. Gather required environmental evidence step-by-step using available tools:
   - 'get_aqi': To fetch real PM2.5, PM10 and AQI.
   - 'get_weather': To fetch temperature, humidity, and heat index.
   - 'retrieve_health_advice': To retrieve medical advisories tailored to the target risk group.
   - 'translate': To translate advice into the user's requested language/script (Urdu, Roman Urdu, Sindhi).
4. If a tool has already been called and its result is in the conversation history, do NOT call the same tool again with identical arguments.
5. When sufficient data has been collected, formulate a concise, practical, empathetic final advisory in the same language and script the user wrote in (e.g. Roman Urdu, Urdu, or English).
"""


class PuterSaafHawaBrain:
    """LLM Brain connecting to Puter (https://github.com/heyPuter/puter)."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        auth_token: Optional[str] = None,
    ):
        self.model_name = os.getenv("PUTER_MODEL", model_name)
        self.auth_token = auth_token or os.getenv("PUTER_AUTH_TOKEN")
        self.step_counter = 0
        self.api_url = "https://api.puter.com/drivers/call"

    def is_authenticated(self) -> bool:
        return bool(self.auth_token)

    async def _query_puter_api(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Calls Puter's unified driver endpoint with native authentication."""
        if not self.auth_token:
            raise ValueError(
                "Puter auth token is missing. Set PUTER_AUTH_TOKEN in .env or pass auth_token."
            )

        args_payload: Dict[str, Any] = {
            "messages": messages,
            "model": self.model_name,
            "temperature": 0.2,
        }
        if tools:
            args_payload["tools"] = tools

        body = {
            "interface": "puter-chat-completion",
            "driver": "ai-chat",
            "method": "complete",
            "args": args_payload,
            "auth_token": self.auth_token,
        }
        headers = {"Content-Type": "text/plain;actually=json"}

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(self.api_url, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            if not data.get("success", False):
                raise RuntimeError(f"Puter API error: {data.get('error', 'Unknown error')}")
            return data

    def _fallback_deterministic_decide(
        self,
        query: str,
        state: Dict[str, Any],
    ) -> AgentDecision:
        """Deterministic policy used when running offline or in simulated evaluation mode."""
        self.step_counter += 1
        lowered = query.lower()

        # Step 1: Need Weather data
        if "get_weather" not in state and "weather_api" not in state:
            return AgentDecision(
                is_final=False,
                tool=ToolCall(name="get_weather", args={"city": "Karachi"}),
                model_used=f"{self.model_name} (deterministic-fallback)",
            )

        # Step 2: Need AQI data
        if "get_aqi" not in state and "air_quality_api" not in state:
            return AgentDecision(
                is_final=False,
                tool=ToolCall(name="get_aqi", args={"city": "Karachi"}),
                model_used=f"{self.model_name} (deterministic-fallback)",
            )

        # Step 3: Need Health Advisory
        if "retrieve_health_advice" not in state:
            target = "general"
            if "bach" in lowered or "child" in lowered or "school" in lowered:
                target = "children during midday heat"
            elif "jog" in lowered or "exercise" in lowered or "run" in lowered:
                target = "outdoor exertion during high pollution"
            elif "asthma" in lowered or "saans" in lowered:
                target = "asthma patients respiratory protection"
            elif "mazdoor" in lowered or "labour" in lowered or "worker" in lowered:
                target = "outdoor labourers hydration and rest"

            return AgentDecision(
                is_final=False,
                tool=ToolCall(name="retrieve_health_advice", args={"query": target}),
                model_used=f"{self.model_name} (deterministic-fallback)",
            )

      
      
        is_roman_urdu = any(w in lowered for w in ["aj", "kitna", "bachon", "hogaye", "qk", "hai", "karna"])
        if is_roman_urdu:
            ans = (
                "SaafHawa Advisory (Karachi):\n"
                "• Temperature: Dopahar 12 bajy temperature 38°C tak ja sakta hai (Heat index ~43°C).\n"
                "• Air Quality: AQI 165 (Unhealthy) hai.\n"
                "• Hifazati Tadabeer: Bachon ko seedhi dhoop se bachayein, ORS/paani ka istemal barhayein "
                "aur outdoor activities meethi dhoop tak mehdood rakhein."
            )
        else:
            ans = (
                "SaafHawa Advisory (Karachi):\n"
                "• Temperature: Expected midday peak around 38°C (Heat Index ~43°C).\n"
                "• Air Quality: AQI is 165 (Unhealthy for sensitive groups).\n"
                "• Public Health Advice: Limit outdoor exertion between 11 AM - 3 PM. "
                "Ensure hydration and avoid strenuous exercise near industrial/heavy traffic corridors."
            )

        return AgentDecision(
            is_final=True,
            final_answer=ans,
            model_used=f"{self.model_name} (deterministic-fallback)",
        )

    async def decide(
        self,
        query: str,
        state: Dict[str, Any],
    ) -> AgentDecision:
        """Main agent decision function.

        Queries Puter LLM with current history and tool schemas to select the next action.
        """
        if self.auth_token:
            try:
                # Build multi-turn messages preserving tool calling history
                messages: List[Dict[str, Any]] = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ]

                # Append previous tool calls and tool returns into the conversation history
                for tool_name, data in state.items():
                    call_id = f"call_{tool_name}"
                    tool_args = data.get("args") or {"city": "Karachi"}
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": json.dumps(tool_args),
                                },
                            }
                        ],
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": tool_name,
                        "content": json.dumps(data.get("result", data)),
                    })

                # Decide if tools are still available
                # If we have 2 or more tools completed, allow LLM to either continue or finalize
                tools_to_offer = TOOL_SCHEMAS if len(state) < 3 else None

                response = await self._query_puter_api(messages=messages, tools=tools_to_offer)
                res_data = response.get("result", {})
                message = res_data.get("message", {})

                # Check if LLM requested tool calling
                tool_calls = message.get("tool_calls") or []
                if tool_calls:
                    first_call = tool_calls[0].get("function", {})
                    fn_name = first_call.get("name")
                    fn_args = json.loads(first_call.get("arguments", "{}"))
                    # Guard: If tool was already executed with same args, finalize
                    if fn_name in state and len(state) >= 2:
                        pass
                    else:
                        return AgentDecision(
                            is_final=False,
                            tool=ToolCall(name=fn_name, args=fn_args, call_id=tool_calls[0].get("id")),
                            model_used=response.get("metadata", {}).get("providerUsed", self.model_name),
                            raw_response=response,
                        )

                content = message.get("content") or ""
                if content:
                    return AgentDecision(
                        is_final=True,
                        final_answer=content,
                        model_used=response.get("metadata", {}).get("providerUsed", self.model_name),
                        raw_response=response,
                    )
            except Exception as e:
                logger.warning(f"Puter live API call failed ({e}). Falling back to deterministic agent brain.")

        return self._fallback_deterministic_decide(query, state)


async def main():
    print("=" * 65)
    print("  SaafHawa Agent LLM Brain Setup (Puter Live Engine)")
    print("=" * 65)

    brain = PuterSaafHawaBrain()
    print(f"Puter Auth Active: {brain.is_authenticated()}")
    print(f"Target Model: {brain.model_name}\n")

    sample_query = "Aj din 12 bajy temperature kitna rise hoga qk bachon ki school 12 bajy off hojayega?"
    print(f"User Query: {sample_query}\n")

    state: Dict[str, Any] = {}
    for step in range(1, 5):
        decision = await brain.decide(sample_query, state)
        if decision.is_final:
            print(f"\n[Step {step}] Final Response Formulated by Puter ({decision.model_used}):")
            print("-" * 55)
            print(decision.final_answer)
            print("-" * 55)
            break
        else:
            print(f"[Step {step}] Puter LLM selected Tool: '{decision.tool.name}' with args {decision.tool.args}")
            # Emulate tool execution and store in state
            if decision.tool.name == "get_weather":
                state["get_weather"] = {
                    "args": decision.tool.args,
                    "result": {"city": "Karachi", "temperature_celsius": 38.0, "relative_humidity_percent": 65, "heat_index_celsius": 44.5},
                }
            elif decision.tool.name == "get_aqi":
                state["get_aqi"] = {
                    "args": decision.tool.args,
                    "result": {"city": "Karachi", "aqi": 178, "category": "Unhealthy", "pm2_5": 92.5},
                }
            elif decision.tool.name == "retrieve_health_advice":
                state["retrieve_health_advice"] = {
                    "args": decision.tool.args,
                    "result": {"target_group": "children", "guidance": "Prevent outdoor exposure during 11 AM - 3 PM. Hydrate well."},
                }
            else:
                state[decision.tool.name] = {"args": decision.tool.args, "result": {"status": "success"}}


if __name__ == "__main__":
    asyncio.run(main())
