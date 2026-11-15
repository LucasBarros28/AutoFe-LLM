
import time

from src.llm.gemini_client import GeminiClient


print("Inicializando cliente Gemini...")

client = GeminiClient()

print("Modelo:", client.model)
print("Iniciando interação em segundo plano...")

start = time.perf_counter()

try:

    interaction = client.client.interactions.create(
        model=client.model,
        input="Responda somente com a palavra OK.",
        background=True
    )

    print("\nInteração iniciada!")
    print("ID:", interaction.id)
    print("Status:", interaction.status)

    print("\nAguardando resposta do Gemini...")

    max_wait = 600
    elapsed = 0

    while elapsed < max_wait:

        time.sleep(5)
        elapsed += 5

        result = client.client.interactions.get(
            id=interaction.id
        )

        print(
            f"Status: {result.status} "
            f"| Tempo: {elapsed}s"
        )

        if result.status == "completed":

            total = time.perf_counter() - start

            print("\nResposta:", result.output_text)
            print(f"Tempo total: {total:.2f} segundos")

            break

        elif result.status in ("failed", "cancelled"):

            print("\nInteração não concluída.")
            print("Status final:", result.status)

            # Verifica se a API retornou erros
            errors = getattr(result, "errors", None)

            if errors:

                print("\nErros retornados pela API:")

                for error in errors:

                    print("Código:", getattr(
                        error, "code", "Não informado"
                    ))

                    print("Mensagem:", getattr(
                        error, "message", "Não informada"
                    ))

            else:

                print("\nNenhum erro encontrado no atributo 'errors'.")

                # Tenta exibir todos os dados da interação
                if hasattr(result, "model_dump"):

                    print("\nDados completos da interação:")
                    print(result.model_dump(exclude_none=True))

                else:

                    print("\nAtributos disponíveis:")
                    print([
                        attr for attr in dir(result)
                        if not attr.startswith("_")
                    ])

            break

    else:

        print("\nTempo máximo de espera atingido.")

except Exception as error:

    print("\nERRO:", type(error).__name__)
    print("Detalhes:", error)

    total = time.perf_counter() - start
    print(f"Tempo até o erro: {total:.2f} segundos")
