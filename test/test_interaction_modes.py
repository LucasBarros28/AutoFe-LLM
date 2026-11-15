
import pandas as pd

from src.analysis.autofe_transformer import AutoFETransformer


df = pd.DataFrame({
    "age": [20, 30, 40, 50, 60],
    "bmi": [22, 25, 28, 31, 35],
    "fare": [10, 20, 30, 100, 200]
})

print("===== AUTOFE SEM INTERACOES =====")

autofe_traditional = AutoFETransformer(
    include_interactions=False
)

autofe_traditional.fit(df)

print("Features geradas:")
print(autofe_traditional.generated_features_)

print("Interacoes:")
print(autofe_traditional.interaction_opportunities_)


print("\n===== AUTOFE COM INTERACOES =====")

autofe_interactions = AutoFETransformer(
    include_interactions=True,
    max_interactions=3
)

autofe_interactions.fit(df)

print("Features geradas:")
print(autofe_interactions.generated_features_)

print("Interacoes:")
print(autofe_interactions.interaction_opportunities_)
