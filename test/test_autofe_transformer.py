import pandas as pd

from src.analysis.autofe_transformer import AutoFETransformer


# ==========================================
# DADOS DE TREINAMENTO
# ==========================================

X_train = pd.DataFrame({
    "Age": [20, None, 30, 40, 50, 60],
    "Fare": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]
})


# ==========================================
# DADOS DE VALIDACAO
# ==========================================

X_test = pd.DataFrame({
    "Age": [25, 35, None],
    "Fare": [2.0, 8.0, 50.0]
})


# ==========================================
# CONFIGURACAO DO AUTOFE
# ==========================================

transformer = AutoFETransformer(
    include_interactions=False
)


# ==========================================
# APRENDIZADO NO TREINAMENTO
# ==========================================

transformer.fit(X_train)


# ==========================================
# APLICACAO DAS TRANSFORMACOES
# ==========================================

train_transformed = transformer.transform(X_train)

test_transformed = transformer.transform(X_test)


# ==========================================
# RESULTADOS
# ==========================================

print("\n===== OPORTUNIDADES DETECTADAS NO TREINAMENTO =====")

for opportunity in transformer.opportunities_:
    print(opportunity)


print("\n===== FEATURES GERADAS =====")

print(transformer.generated_features_)


print("\n===== TREINAMENTO TRANSFORMADO =====")

print(train_transformed)


print("\n===== VALIDACAO TRANSFORMADA =====")

print(test_transformed)


# ==========================================
# VERIFICACOES
# ==========================================

# Indicador de valores ausentes
assert "Age_is_missing" in train_transformed.columns
assert "Age_is_missing" in test_transformed.columns

# Transformacao logaritmica
assert "Fare_log1p" in train_transformed.columns
assert "Fare_log1p" in test_transformed.columns

# Mesmo conjunto e ordem de colunas
assert list(train_transformed.columns) == list(test_transformed.columns)

# Mesmo numero de features
assert train_transformed.shape[1] == test_transformed.shape[1]

# O transformer deve preservar o numero de linhas
assert len(train_transformed) == len(X_train)
assert len(test_transformed) == len(X_test)

print("\nTeste concluido com sucesso!")