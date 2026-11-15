
import pandas as pd

from src.analysis.autofe_transformer import AutoFETransformer


df = pd.DataFrame({
    "idade": [20, 25, 30, 35, 40, 45, 50, 55, 60, 65],
    "renda": [1000, 1200, 1500, 1800, 2200,
              2500, 3000, 3500, 4000, 4500],
    "horas_estudo": [1, 2, 2, 3, 4, 5, 6, 7, 8, 9],
    "compras": [1, 1, 2, 2, 3, 4, 5, 6, 7, 8]
})

y = pd.Series([
    0, 0, 0, 0, 0, 1, 1, 1, 1, 1
])

transformer = AutoFETransformer(
    max_interactions=3,
    include_interactions=True,
    problem_type="classification"
)

# Ajusta usando os dados de treinamento.
transformer.fit(df, y)

# Transforma os mesmos dados.
df_transformed = transformer.transform(df)

print("\n===== INTERACOES SELECIONADAS =====")

for item in transformer.interaction_opportunities_:
    print(
        item["feature"],
        "Score:",
        round(item["score"], 5)
    )

print("\n===== FEATURES GERADAS =====")
print(transformer.generated_features_)

print("\n===== DATASET TRANSFORMADO =====")
print(df_transformed.head())

print("\nDimensoes originais:", df.shape)
print("Dimensoes transformadas:", df_transformed.shape)
