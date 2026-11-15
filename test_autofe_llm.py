
from sklearn.datasets import load_breast_cancer

from src.analysis.autofe_transformer import AutoFETransformer


# Carrega dataset
dataset = load_breast_cancer(as_frame=True)

X = dataset.data
y = dataset.target


# Inicializa AutoFE com LLM
autofe = AutoFETransformer(
    include_interactions=True,
    problem_type="classification",
    use_llm=True
)


# Ajusta o transformer
autofe.fit(X, y)


# Aplica as transformações
X_transformed = autofe.transform(X)


# Resultados
print("\n" + "=" * 60)
print("RESULTADO DO AUTOFE COM LLM")
print("=" * 60)

print("\nDimensoes originais:", X.shape)
print("Dimensoes transformadas:", X_transformed.shape)

print("\nSugestoes do Gemini:")

for suggestion in autofe.llm_suggestions_:
    print(suggestion)

print("\nFeatures geradas:")
print(autofe.generated_features_)

print("\nSugestoes rejeitadas:")
print(autofe.llm_rejected_suggestions_)
