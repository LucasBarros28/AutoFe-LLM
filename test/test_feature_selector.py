
import pandas as pd
from sklearn.base import clone
from src.analysis.feature_selector import FeatureSelector


X = pd.DataFrame({
    "Age": [20, 30, 40, 50, 60, 70, 80, 90],
    "Fare": [10, 20, 30, 40, 50, 60, 70, 80],
    "Age_is_missing": [0, 0, 1, 0, 0, 1, 0, 0],
    "Fare_log1p": [2, 3, 4, 5, 6, 7, 8, 9],
    "Random_feature": [1, 0, 1, 0, 1, 0, 1, 0]
})

y = pd.Series([0, 0, 1, 1, 1, 1, 0, 0])


selector = FeatureSelector(
    original_columns=["Age", "Fare"],
    k=2,
    problem_type="classification"
)

X_selected = selector.fit_transform(X, y)


print("\n===== PONTUACAO DAS FEATURES =====")

for feature, score in selector.get_feature_scores().items():
    print(f"{feature}: {score:.4f}")


print("\n===== FEATURES SELECIONADAS =====")
print(selector.selected_features_)


print("\n===== DATASET APOS SELECAO =====")
print(X_selected.head())


assert "Age" in X_selected.columns
assert "Fare" in X_selected.columns

assert len(selector.selected_features_) <= 2

assert all(
    feature in X_selected.columns
    for feature in selector.selected_features_
)

print("\nTeste concluido com sucesso!")

print("\n===== TESTE DE CLONAGEM =====")

selector_clone = clone(selector)

print("Clonagem realizada com sucesso!")
print("Problem type:", selector_clone.problem_type)
print("K:", selector_clone.k)

assert selector_clone.problem_type == "classification"
assert selector_clone.k == 2

print("Teste de clonagem concluido com sucesso!")