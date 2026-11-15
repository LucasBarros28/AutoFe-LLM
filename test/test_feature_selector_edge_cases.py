
import pandas as pd
import numpy as np

from src.analysis.feature_selector import FeatureSelector


def expect_error(function, expected_exception=ValueError):

    try:
        function()

    except expected_exception:
        return

    raise AssertionError(
        f"Era esperado um erro do tipo {expected_exception.__name__}."
    )


# ==========================================
# TESTE 1: VALORES INFINITOS NAS CANDIDATAS
# ==========================================

X = pd.DataFrame({
    "Age": [20, 30, 40, 50, 60, 70],
    "Age_feature": [1, 2, np.inf, 4, 5, 6]
})

y = [0, 0, 0, 1, 1, 1]

selector = FeatureSelector(
    original_columns=["Age"],
    k=1
)

selector.fit(X, y)

assert np.isfinite(
    list(selector.feature_scores_.values())
).all()

print("Teste 1 aprovado: infinitos tratados.")


# ==========================================
# TESTE 2: FEATURES CONSTANTES
# ==========================================

X = pd.DataFrame({
    "Age": [20, 30, 40, 50, 60, 70],
    "Constant_feature": [1, 1, 1, 1, 1, 1],
    "Useful_feature": [0, 0, 0, 1, 1, 1]
})

y = [0, 0, 0, 1, 1, 1]

selector = FeatureSelector(
    original_columns=["Age"],
    k=2
)

selector.fit(X, y)

assert selector.feature_scores_["Constant_feature"] == 0.0
assert "Constant_feature" not in selector.selected_features_

print("Teste 2 aprovado: features constantes identificadas.")


# ==========================================
# TESTE 3: CANDIDATAS NAO NUMERICAS
# ==========================================

X = pd.DataFrame({
    "Age": [20, 30, 40, 50, 60, 70],
    "Text_feature": ["a", "b", "c", "d", "e", "f"],
    "Useful_feature": [0, 0, 0, 1, 1, 1]
})

y = [0, 0, 0, 1, 1, 1]

selector = FeatureSelector(
    original_columns=["Age"],
    k=2
)

selector.fit(X, y)

assert np.isfinite(
    list(selector.feature_scores_.values())
).all()

print("Teste 3 aprovado: candidatas nao numericas tratadas.")


# ==========================================
# TESTE 4: SEM FEATURES CANDIDATAS
# ==========================================

X = pd.DataFrame({
    "Age": [20, 30, 40, 50, 60, 70],
    "Fare": [10, 20, 30, 40, 50, 60]
})

y = [0, 0, 0, 1, 1, 1]

selector = FeatureSelector(
    original_columns=["Age", "Fare"],
    k=2
)

selector.fit(X, y)

assert selector.selected_features_ == []

result = selector.transform(X)

assert list(result.columns) == ["Age", "Fare"]

print("Teste 4 aprovado: ausencia de candidatas.")


# ==========================================
# TESTE 5: REGRESSAO
# ==========================================

X = pd.DataFrame({
    "Age": [20, 30, 40, 50, 60, 70],
    "Useful_feature": [1, 2, 3, 4, 5, 6],
    "Random_feature": [3, 1, 6, 2, 5, 4]
})

y = np.array([10, 20, 30, 40, 50, 60])

selector = FeatureSelector(
    original_columns=["Age"],
    k=2,
    problem_type="regression"
)

selector.fit(X, y)

assert len(selector.feature_scores_) == 2
assert np.isfinite(
    list(selector.feature_scores_.values())
).all()

print("Teste 5 aprovado: selecao para regressao.")


# ==========================================
# TESTE 6: TARGET COM VALORES INFINITOS
# ==========================================

X = pd.DataFrame({
    "Age": [20, 30, 40, 50],
    "Feature": [1, 2, 3, 4]
})

y = [0, 1, np.inf, 1]

selector = FeatureSelector(
    original_columns=["Age"]
)

expect_error(lambda: selector.fit(X, y))

print("Teste 6 aprovado: target infinito rejeitado.")


# ==========================================
# TESTE 7: TAMANHOS DIFERENTES
# ==========================================

X = pd.DataFrame({
    "Age": [20, 30, 40, 50],
    "Feature": [1, 2, 3, 4]
})

y = [0, 1, 0]

selector = FeatureSelector(
    original_columns=["Age"]
)

expect_error(lambda: selector.fit(X, y))

print("Teste 7 aprovado: dimensoes inconsistentes rejeitadas.")


# ==========================================
# TESTE 8: COLUNA NECESSARIA AUSENTE
# ==========================================

X_train = pd.DataFrame({
    "Age": [20, 30, 40, 50, 60, 70],
    "Feature": [1, 2, 3, 4, 5, 6]
})

y = [0, 0, 0, 1, 1, 1]

selector = FeatureSelector(
    original_columns=["Age"],
    k=1
)

selector.fit(X_train, y)

X_test = pd.DataFrame({
    "Feature": [1, 2, 3]
})

expect_error(lambda: selector.transform(X_test))

print("Teste 8 aprovado: colunas obrigatorias verificadas.")


# ==========================================
# RESULTADO FINAL
# ==========================================

print("\nTodos os testes de robustez foram aprovados!")
