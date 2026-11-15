import os
import time

from dotenv import load_dotenv
from groq import Groq


# Carrega as variáveis do arquivo .env
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY não encontrada. Verifique seu arquivo .env."
    )


print("Inicializando cliente Groq...", flush=True)

client = Groq(api_key=api_key)

# Modelo escolhido para o teste
model = "openai/gpt-oss-120b"

print("Modelo:", model, flush=True)
print("Enviando solicitação...", flush=True)

start = time.perf_counter()

try:

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Responda somente com a palavra OK."
            }
        ],
        temperature=0.2,
        max_tokens=50
    )

    elapsed = time.perf_counter() - start

    answer = response.choices[0].message.content

    print("\nSUCESSO!", flush=True)
    print("Resposta:", answer, flush=True)
    print(f"Tempo total: {elapsed:.2f} segundos", flush=True)

    if response.usage:
        print("\nUtilização de tokens:")
        print("Prompt:", response.usage.prompt_tokens)
        print("Resposta:", response.usage.completion_tokens)
        print("Total:", response.usage.total_tokens)

except Exception as error:

    elapsed = time.perf_counter() - start

    print("\nFALHA NA CHAMADA", flush=True)
    print("Tipo:", type(error).__name__, flush=True)
    print("Erro:", error, flush=True)
    print(f"Tempo até a falha: {elapsed:.2f} segundos", flush=True)