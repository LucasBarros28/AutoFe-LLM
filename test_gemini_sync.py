
import time

from concurrent.futures import ThreadPoolExecutor, TimeoutError

from src.llm.gemini_client import GeminiClient


def testar_gemini():

    print("Inicializando cliente Gemini...", flush=True)

    client = GeminiClient()

    print("Modelo:", client.model, flush=True)
    print("Enviando solicitação...", flush=True)

    start = time.perf_counter()

    response = client.client.models.generate_content(
        model=client.model,
        contents="Responda somente com a palavra OK."
    )

    elapsed = time.perf_counter() - start

    print("\nSUCESSO!", flush=True)
    print("Resposta:", response.text, flush=True)
    print(f"Tempo: {elapsed:.2f}s", flush=True)


if __name__ == "__main__":

    executor = ThreadPoolExecutor(max_workers=1)

    future = executor.submit(testar_gemini)

    try:

        future.result(timeout=30)

    except TimeoutError:

        print("\nTIMEOUT: Gemini demorou mais de 30 segundos.")

        future.cancel()

    except Exception as error:

        print("\nFALHA NA CHAMADA")
        print("Tipo:", type(error).__name__)
        print("Erro:", error)

    finally:

        executor.shutdown(wait=False, cancel_futures=True)
