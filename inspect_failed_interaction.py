from src.llm.gemini_client import GeminiClient

client = GeminiClient()

interaction_id = (
    "v1_ChdLemI1YW9yTkI3NmRvX1VQZ282NjJRRRIX"
    "S3piNWFvck5CNzZkb19VUGdvNjYyUUU"
)

print("Consultando interação anterior...")

try:
    result = client.client.interactions.get(
        id=interaction_id
    )

    print("\nSTATUS:", result.status)
    print("MODELO:", result.model)

    print("\nERRO (error):")
    print(getattr(result, "error", None))

    print("\nERROS (errors):")
    print(getattr(result, "errors", None))

    print("\nATRIBUTOS DISPONÍVEIS:")

    if hasattr(result, "model_dump"):
        data = result.model_dump(exclude_none=False)

        for key, value in data.items():
            if key not in ("input", "steps"):
                print(f"{key}: {value}")

        print("\nETAPAS:")

        for step in data.get("steps") or []:
            print(step)

    else:
        print(result)

except Exception as error:
    print("\nErro ao recuperar interação:")
    print(type(error).__name__, error)