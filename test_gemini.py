
import pandas as pd

from sklearn.datasets import load_breast_cancer

from src.llm.gemini_client import GeminiClient


# ==========================================
# 1. CARREGAR DATASET
# ==========================================

dataset = load_breast_cancer(as_frame=True)

X = dataset.data

print("=" * 60)
print("DATASET")
print("=" * 60)

print("Dimensoes:", X.shape)
print("\nColunas:")
print(X.columns.tolist())


# ==========================================
# 2. CONSTRUIR CONTEXTO
# ==========================================

numerical_columns = X.select_dtypes(
    include="number"
).columns.tolist()

column_information = {}

for column in X.columns:

    column_information[column] = {
        "missing_values": int(X[column].isna().sum()),
        "missing_percentage": round(
            X[column].isna().mean() * 100, 2
        ),
        "mean": float(X[column].mean()),
        "std": float(X[column].std()),
        "min": float(X[column].min()),
        "max": float(X[column].max())
    }


context = {
    "dataset_description": (
        "Dataset de características numéricas "
        "de tumores mamários."
    ),
    "problem_type": "classification",
    "number_of_rows": len(X),
    "number_of_columns": len(X.columns),
    "columns": X.columns.tolist(),
    "numerical_columns": numerical_columns,
    "column_information": column_information
}


# ==========================================
# 3. INICIALIZAR CLIENTE
# ==========================================

print("\n" + "=" * 60)
print("INICIALIZANDO GEMINI")
print("=" * 60)

try:

    gemini = GeminiClient()

    print("Cliente inicializado com sucesso!")
    print("Modelo:", gemini.model)


    # ==========================================
    # 4. SOLICITAR SUGESTOES
    # ==========================================

    print("\nConsultando o Gemini...")

    suggestions = gemini.suggest_features(context)


    # ==========================================
    # 5. EXIBIR RESULTADOS
    # ==========================================

    print("\n" + "=" * 60)
    print("SUGESTOES VALIDAS")
    print("=" * 60)

    if suggestions:

        for i, suggestion in enumerate(
            suggestions, start=1
        ):

            print(f"\nSugestao {i}:")
            print("Operacao:", suggestion["operation"])

            if "column" in suggestion:
                print("Coluna:", suggestion["column"])

            if "columns" in suggestion:
                print("Colunas:", suggestion["columns"])

            print("Justificativa:", suggestion["reason"])

    else:

        print("O Gemini nao recomendou transformacoes.")


    # ==========================================
    # 6. EXIBIR SUGESTOES REJEITADAS
    # ==========================================

    print("\n" + "=" * 60)
    print("SUGESTOES REJEITADAS")
    print("=" * 60)

    rejected = gemini.last_rejected_suggestions_

    if rejected:

        for item in rejected:
            print("\nSugestao:", item["suggestion"])
            print("Motivo:", item["reason"])

    else:

        print("Nenhuma sugestao foi rejeitada.")


    print("\nTeste finalizado!")


except Exception as error:

    print("\nERRO DURANTE O TESTE:")
    print(type(error).__name__)
    print(error)
