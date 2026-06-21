from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

def ask_qwen(question, context):

    prompt = f"""
Answer ONLY from the provided Wikipedia context.

Context:
{context}

Question:
{question}

If the answer is not present, say:
Information not found in Wikipedia article.
"""

    response = client.chat.completions.create(
        model="qwen/qwen-2.5-7b-instruct",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content