
import pandas as pd
import numpy as np

from src.analysis.feature_generator import FeatureGenerator


def expect_error(function, expected_exception=ValueError):

    try:
        function()

    except expected_exception:
        return

    raise AssertionError(
        f"Era esperado um erro do tipo {expected_exception.__name__}."
    )


# ==========================================
# TESTE 1: VALORES INFINITOS
# ==========================================

df = pd.DataFrame({
    "Age": [20, 30, np.inf, 40]
})

generator = FeatureGenerator(df)

expect_error(
    lambda: generator.generate_log_transform("Age")
)

print("Teste 1 aprovado: valores infinitos rejeitados.")


# ==========================================
# TESTE 2: VALORES NEGATIVOS
# ==========================================

df = pd.DataFrame({
    "Fare": [10, 20, -5, 40]
})

generator = FeatureGenerator(df)

expect_error(
    lambda: generator.generate_log_transform("Fare")
)

print("Teste 2 aprovado: valores negativos rejeitados.")


# ==========================================
# TESTE 3: VALORES AUSENTES
# ==========================================

df = pd.DataFrame({
    "Fare": [10, None, 30, 40]
})

generator = FeatureGenerator(df)

result = generator.generate_log_transform("Fare")

assert result["Fare_log1p"].isna().sum() == 1

print("Teste 3 aprovado: valores ausentes preservados.")


# ==========================================
# TESTE 4: COLUNA JA EXISTENTE
# ==========================================

df = pd.DataFrame({
    "Fare": [10, 20, 30],
    "Fare_log1p": [100, 200, 300]
})

generator = FeatureGenerator(df)

expect_error(
    lambda: generator.generate_log_transform("Fare")
)

assert generator.df["Fare_log1p"].tolist() == [100, 200, 300]

print("Teste 4 aprovado: colunas existentes protegidas.")


# ==========================================
# TESTE 5: OVERFLOW EM INTERACAO
# ==========================================

df = pd.DataFrame({
    "A": [1e308, 2.0],
    "B": [1e308, 3.0]
})

generator = FeatureGenerator(df)

with np.errstate(over="ignore", invalid="ignore"):
    expect_error(
        lambda: generator.generate_interaction("A", "B")
    )

assert "A_x_B" not in generator.df.columns

print("Teste 5 aprovado: overflow rejeitado.")


# ==========================================
# TESTE 6: COLUNAS DUPLICADAS
# ==========================================

df = pd.DataFrame(
    [[10, 20]],
    columns=["Age", "Age"]
)

expect_error(
    lambda: FeatureGenerator(df)
)

print("Teste 6 aprovado: colunas duplicadas rejeitadas.")


# ==========================================
# TESTE 7: OPORTUNIDADE MALFORMADA
# ==========================================

df = pd.DataFrame({
    "Age": [20, 30, 40]
})

generator = FeatureGenerator(df)

expect_error(
    lambda: generator.generate_candidates([
        {
            "type": "interaction_product",
            "columns": ["Age"]
        }
    ])
)

print("Teste 7 aprovado: oportunidades malformadas rejeitadas.")


# ==========================================
# RESULTADO FINAL
# ==========================================

print("\nTodos os testes de robustez foram aprovados!")
