
import time

from sklearn.datasets import load_breast_cancer

from src.analysis.autofe_transformer import AutoFETransformer


# ==========================================
# CARREGAMENTO DO DATASET
# ==========================================

print("Carregando dataset...", flush=True)

dataset = load_breast_cancer(as_frame=True)

X = dataset.data
y = dataset.target

print("Dataset:", X.shape, flush=True)


# ==========================================
# TESTE 1: AUTOFE TRADICIONAL
# SEM INTERACOES E SEM LLM
# ==========================================

print("\n" + "=" * 50, flush=True)
print("TESTE 1: AutoFE tradicional", flush=True)
print("=" * 50, flush=True)

autofe = AutoFETransformer(
    include_interactions=False,
    problem_type="classification",
    use_llm=False
)

# FIT

start = time.perf_counter()

autofe.fit(X, y)

fit_time = time.perf_counter() - start

print(
    f"\nFit concluido em {fit_time:.2f} segundos",
    flush=True
)


# VERIFICACAO DOS ATRIBUTOS

print("\nVerificando atributos do AutoFE...", flush=True)

print(
    "Possui columns_in_:",
    hasattr(autofe, "columns_in_"),
    flush=True
)

print(
    "Possui generation_log_:",
    hasattr(autofe, "generation_log_"),
    flush=True
)

print(
    "Possui generated_features_:",
    hasattr(autofe, "generated_features_"),
    flush=True
)

print(
    "Quantidade de oportunidades:",
    len(autofe.opportunities_),
    flush=True
)

print(
    "Features geradas:",
    autofe.generated_features_,
    flush=True
)

if not hasattr(autofe, "generation_log_"):
    raise RuntimeError(
        "O fit() nao criou generation_log_. "
        "Verifique o AutoFETransformer."
    )


# TRANSFORM

print("\nExecutando transform...", flush=True)

start = time.perf_counter()

X_transformed = autofe.transform(X)

transform_time = time.perf_counter() - start

print(
    f"Transform concluido em {transform_time:.2f} segundos",
    flush=True
)

print(
    "Formato original:",
    X.shape,
    flush=True
)

print(
    "Formato transformado:",
    X_transformed.shape,
    flush=True
)


# ==========================================
# TESTE 2: AUTOFE COM GEMINI
# SEM INTERACOES TRADICIONAIS
# ==========================================

print("\n" + "=" * 50, flush=True)
print("TESTE 2: AutoFE com Gemini", flush=True)
print("=" * 50, flush=True)

autofe_llm = AutoFETransformer(
    include_interactions=False,
    problem_type="classification",
    use_llm=True
)

# FIT COM LLM

start = time.perf_counter()

autofe_llm.fit(X, y)

fit_llm_time = time.perf_counter() - start

print(
    f"\nFit com LLM concluido em "
    f"{fit_llm_time:.2f} segundos",
    flush=True
)


# VERIFICACAO DAS SUGESTOES

print("\nVerificando resultados da LLM...", flush=True)

print(
    "Sugestoes recebidas:",
    len(autofe_llm.llm_suggestions_),
    flush=True
)

print(
    "Sugestoes rejeitadas:",
    len(autofe_llm.llm_rejected_suggestions_),
    flush=True
)

print(
    "Oportunidades tradicionais:",
    len(autofe_llm.opportunities_),
    flush=True
)

print(
    "Oportunidades sugeridas pela LLM:",
    len(autofe_llm.llm_opportunities_),
    flush=True
)

print(
    "Features geradas:",
    autofe_llm.generated_features_,
    flush=True
)


# TRANSFORM COM LLM

print("\nExecutando transform com LLM...", flush=True)

start = time.perf_counter()

X_transformed_llm = autofe_llm.transform(X)

transform_llm_time = time.perf_counter() - start

print(
    f"Transform com LLM concluido em "
    f"{transform_llm_time:.2f} segundos",
    flush=True
)

print(
    "Formato original:",
    X.shape,
    flush=True
)

print(
    "Formato transformado com LLM:",
    X_transformed_llm.shape,
    flush=True
)


# ==========================================
# RESUMO
# ==========================================

print("\n" + "=" * 50, flush=True)
print("RESUMO DOS TESTES", flush=True)
print("=" * 50, flush=True)

print(
    f"AutoFE tradicional - Fit: {fit_time:.2f}s",
    flush=True
)

print(
    f"AutoFE tradicional - Transform: {transform_time:.2f}s",
    flush=True
)

print(
    f"AutoFE com Gemini - Fit: {fit_llm_time:.2f}s",
    flush=True
)

print(
    f"AutoFE com Gemini - Transform: "
    f"{transform_llm_time:.2f}s",
    flush=True
)

print("\nTESTES FINALIZADOS!", flush=True)
