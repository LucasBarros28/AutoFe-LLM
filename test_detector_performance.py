import time

from sklearn.datasets import load_breast_cancer

from src.analysis.opportunity_detector import OpportunityDetector


# ==========================================
# CARREGAMENTO
# ==========================================

print("Carregando dataset...", flush=True)

dataset = load_breast_cancer(as_frame=True)

X = dataset.data
y = dataset.target

print("Dataset carregado:", X.shape, flush=True)


detector = OpportunityDetector(X)


# ==========================================
# TESTE 1: CLASSIFICACAO
# ==========================================

print("\nTESTE 1: classify_columns()", flush=True)

start = time.perf_counter()

roles = detector.classify_columns()

print(
    f"Concluido em {time.perf_counter() - start:.2f} segundos",
    flush=True
)


# ==========================================
# TESTE 2: DETECCAO ESTATISTICA
# ==========================================

print("\nTESTE 2: detect()", flush=True)

start = time.perf_counter()

opportunities = detector.detect()

print(
    f"Concluido em {time.perf_counter() - start:.2f} segundos",
    flush=True
)

print("Oportunidades:", len(opportunities), flush=True)


# ==========================================
# TESTE 3: DETECCAO DE INTERACOES
# ==========================================

print("\nTESTE 3: detect_interactions()", flush=True)

start = time.perf_counter()

interactions = detector.detect_interactions(
    max_interactions=10,
    y=y,
    problem_type="classification"
)

print(
    f"Concluido em {time.perf_counter() - start:.2f} segundos",
    flush=True
)

print("Interacoes retornadas:", len(interactions), flush=True)


print("\nTESTES FINALIZADOS!", flush=True)