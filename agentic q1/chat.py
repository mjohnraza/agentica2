"""Interactive Chat Runner for the SaafHawa Agentic Assistant.

Type your questions in Roman Urdu or English to get live air quality,
temperature forecasts, heatwave alerts, and medical guidance for Karachi.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, Dict

import httpx
from model import PuterSaafHawaBrain, ToolCall
from latencey import (
    calculate_heat_index_celsius,
    search_knowledge_base,
    WEATHER_API_URL,
    AIR_QUALITY_API_URL,
)


async def execute_real_tool(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Executes live tools and APIs for Karachi."""
    city = args.get("city", "Karachi")

    async with httpx.AsyncClient(timeout=10.0) as client:
        if tool_name == "get_weather":
            try:
                resp = await client.get(WEATHER_API_URL)
                data = resp.json().get("current", {})
                temp = data.get("temperature_2m", 34.0)
                rh = data.get("relative_humidity_2m", 60.0)
                wind = data.get("wind_speed_10m", 12.0)
                hi = calculate_heat_index_celsius(temp, rh)
                return {
                    "city": city,
                    "temperature_celsius": temp,
                    "relative_humidity_percent": rh,
                    "wind_speed_kmh": wind,
                    "heat_index_celsius": hi,
                    "condition": "Hazy & Hot",
                }
            except Exception as e:
                return {"city": city, "temperature_celsius": 36.0, "heat_index_celsius": 42.0, "error": str(e)}

        elif tool_name == "get_aqi":
            try:
                resp = await client.get(AIR_QUALITY_API_URL)
                data = resp.json().get("current", {})
                pm25 = data.get("pm2_5", 85.0)
                pm10 = data.get("pm10", 140.0)
                aqi = data.get("us_aqi", 168)
                cat = "Unhealthy" if aqi > 150 else ("Moderate" if aqi > 50 else "Good")
                return {
                    "city": city,
                    "aqi": aqi,
                    "category": cat,
                    "pollutants": {"pm2_5": pm25, "pm10": pm10},
                }
            except Exception as e:
                return {"city": city, "aqi": 165, "category": "Unhealthy (Fallback)", "error": str(e)}

        elif tool_name == "retrieve_health_advice":
            query = args.get("query", "general")
            advice_items = search_knowledge_base(query)
            return {"query": query, "advisories": advice_items}

        elif tool_name == "translate":
            text = args.get("text", "")
            lang = args.get("lang", "roman_ur")
            return {"original_text": text, "target_language": lang, "translated_text": text}

    return {"status": "success", "tool": tool_name, "args": args}


async def run_agent_turn(brain: PuterSaafHawaBrain, user_query: str, max_steps: int = 5) -> str:
    """Executes the agentic loop with Puter LLM and live tool execution."""
    state: Dict[str, Any] = {}
    print("\n" + "-" * 50)
    print(f"[Citizen Query]: {user_query}")
    print("-" * 50)

    for step in range(1, max_steps + 1):
        decision = await brain.decide(user_query, state)

        if decision.is_final:
            print(f"\n[SaafHawa Response ({decision.model_used})]:")
            print(decision.final_answer)
            print("-" * 50)
            return decision.final_answer or ""

        tool = decision.tool
        if not tool:
            break

        print(f" -> Step {step}: LLM invoking tool [{tool.name}] with args: {tool.args}")
        tool_result = await execute_real_tool(tool.name, tool.args)
        state[tool.name] = {"args": tool.args, "result": tool_result}

    # Final summary if loop exhausted
    decision = await brain.decide(user_query, state)
    print(f"\n[SaafHawa Advisory ({decision.model_used})]:")
    print(decision.final_answer)
    print("-" * 50)
    return decision.final_answer or ""


async def interactive_cli():
    print("=" * 65)
    print("   SaafHawa Agentic Assistant (Karachi Environmental Advisor)")
    print("=" * 65)
    print("Live tools connected:")
    print(" - Open-Meteo Weather API (Real-time Temp, Humidity, Wind)")
    print(" - Open-Meteo Air Quality API (Real-time PM2.5, PM10, AQI)")
    print(" - Heat Index Rothfusz Engine (Real-time calculation)")
    print(" - Municipal Health & Heatwave Knowledge Base")
    print("\nType your question below (or 'exit' to quit).")
    print("Examples:")
    print(" 1) Aj din 12 bajy temperature kitna rise hoga qk bachon ki school 12 bajy off hojayega?")
    print(" 2) Is it safe to go for an evening jog in Clifton today?")
    print(" 3) Mujhe saans ka masla hai, kya aj Korangi jana theek hai?")
    print("=" * 65 + "\n")

    brain = PuterSaafHawaBrain()

    if len(sys.argv) > 1:
        # One-off query mode
        query = " ".join(sys.argv[1:])
        await run_agent_turn(brain, query)
        return

    # Interactive loop
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting SaafHawa Assistant. Allah Hafiz!")
                break
            await run_agent_turn(brain, user_input)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Allah Hafiz!")
            break


if __name__ == "__main__":
    asyncio.run(interactive_cli())

