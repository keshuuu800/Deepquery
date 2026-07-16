from openai import OpenAI
import os
import structlog

logger = structlog.get_logger()

groq_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

openrouter_client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

GROQ_MODEL = "llama-3.3-70b-versatile"
OPENROUTER_FREE_ROUTER = "openrouter/free"  # auto-picks an available free model


def build_prompt(question: str, context: str) -> str:
    return f"""Answer ONLY from the provided Wikipedia context.

Context:
{context}

Question:
{question}

If the answer is not present, say:
Information not found in Wikipedia article.
"""


def _call(client: OpenAI, model: str, prompt: str, temperature: float = 0.2) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        timeout=30,
    )
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError("Empty response from model")
    return content, getattr(response, "model", model)


def ask_llm(question: str, context: str) -> str:
    prompt = build_prompt(question, context)

    # 1. Groq first
    try:
        result, used_model = _call(groq_client, GROQ_MODEL, prompt)
        logger.info("llm_success", provider="groq", model=used_model)
        return result
    except Exception as e:
        logger.warning("llm_failed", provider="groq", model=GROQ_MODEL, error=str(e))

    # 2. OpenRouter free-model auto-router as fallback
    try:
        result, used_model = _call(openrouter_client, OPENROUTER_FREE_ROUTER, prompt)
        # used_model will show which underlying free model actually handled it
        logger.info("llm_success", provider="openrouter", model=used_model)
        return result
    except Exception as e:
        logger.error("llm_failed", provider="openrouter", model=OPENROUTER_FREE_ROUTER, error=str(e))
        raise RuntimeError(f"All LLM providers failed. Last error: {e}")