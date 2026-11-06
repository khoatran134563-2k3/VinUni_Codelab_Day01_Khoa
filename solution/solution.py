"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
    "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = os.getenv("LAB_GEMINI_MODEL", "gemini-2.5-flash")
ANTHROPIC_MODEL = "claude-3-5-haiku"


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    start = time.perf_counter()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    latency = time.perf_counter() - start

    text = response.choices[0].message.content or ""

    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }

    return text, latency, usage


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5 (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )

    start = time.perf_counter()

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )

    latency = time.perf_counter() - start

    usage = {
        "input_tokens": response.usage_metadata.prompt_token_count or 0,
        "output_tokens": response.usage_metadata.candidates_token_count or 0,
    }

    return response.text or "", latency, usage


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    import anthropic

    client = anthropic.Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    start = time.perf_counter()

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    latency = time.perf_counter() - start

    text = "".join(
        block.text
        for block in response.content
        if isinstance(getattr(block, "text", None), str)
    )

    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }

    return text, latency, usage


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 2.5 Flash)
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    results = {}

    models = [
        ("gpt4o", call_openai, OPENAI_MODEL),
        ("gpt4o_mini", call_openai, OPENAI_MINI_MODEL),
        ("gemini_flash", call_gemini, GEMINI_MODEL),
    ]

    for name, call_function, model in models:
        text, latency, usage = call_function(
            prompt,
            model=model,
        )

        pricing = PRICING_1M_TOKENS[model]

        cost = (
            usage["input_tokens"] * pricing["input"]
            + usage["output_tokens"] * pricing["output"]
        ) / 1_000_000

        results[name] = {
            "response": text,
            "latency": latency,
            "cost": cost,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
        }

    return results


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    history = []

    print("Chatbot đã sẵn sàng! Gõ quit hoặc exit để thoát.")

    while True:
        question = input("\nBạn: ").strip()

        if question.lower() in ("quit", "exit"):
            print("Đã kết thúc trò chuyện.")
            break

        if not question:
            continue

        user_message = types.Content(
            role="user",
            parts=[types.Part(text=question)],
        )

        print("Gemini: ", end="", flush=True)
        answer = ""

        try:
            stream = client.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=history + [user_message],
            )

            for chunk in stream:
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    answer += chunk.text

            print()

            if answer:
                history.append(user_message)
                history.append(
                    types.Content(
                        role="model",
                        parts=[types.Part(text=answer)],
                    )
                )

                history = history[-6:]

        except Exception as error:
            print(f"\nChưa nhận được câu trả lời: {error}")


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception:
            if attempt == max_retries:
                raise

            delay = base_delay * (2 ** attempt)
            time.sleep(delay)


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    results = []

    for prompt in prompts:
        result = compare_models(prompt)

        result["prompt"] = prompt

        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    def clean(value):
        return str(value).replace("\n", " ").replace("\r", " ").replace("|", r"\|")

    lines = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for result in results:
        prompt = clean(result["prompt"])

        for model_name, stats in result.items():
            if model_name == "prompt":
                continue

            response = stats["response"]
            if len(response) > 50:
                response = response[:47] + "..."
            display_name = {
                "gpt4o": "GPT-4o",
                "gpt4o_mini": "GPT-4o-Mini",
                "gemini_flash": "Gemini-Flash",
            }.get(model_name, model_name)

            row = (
                f"| {prompt} "
                f"| {clean(display_name)} "
                f"| {clean(response)} "
                f"| {stats['latency']:.2f}s "
                f"| {stats['input_tokens']}/{stats['output_tokens']} "
                f"| ${stats['cost']:.6f} |"
            )

            lines.append(row)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        # Note: Requires valid API keys set in environment variables
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini 2.5 Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")
