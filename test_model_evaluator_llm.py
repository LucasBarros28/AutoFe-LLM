
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold

from src.evaluation.model_evaluator import ModelEvaluator


# =====================================================
# CARREGAMENTO DOS DADOS
# =====================================================

print("Carregando dataset...")

dataset = load_breast_cancer(as_frame=True)

X = dataset.data
y = dataset.target

print("Dataset:", X.shape)


# =====================================================
# CONFIGURACAO
# =====================================================

evaluator = ModelEvaluator(
    problem_type="classification",
    model_name="decision_tree",
    random_state=42
)

cv = StratifiedKFold(
    n_splits=2,
    shuffle=True,
    random_state=42
)


# =====================================================
# AVALIACAO COM AUTOFE + LLM
# =====================================================

print("\nIniciando avaliacao com AutoFE + Gemini...")

results = evaluator.evaluate(
    X,
    y,
    cv=cv,
    use_autofe=True,
    include_interactions=False,
    use_llm=True,
    llm_model="gemini-3.8-flash"
)


# =====================================================
# RESULTADOS
# =====================================================

print("\n" + "=" * 60)
print("RESULTADOS DO MODEL EVALUATOR + LLM")
print("=" * 60)

for metric, values in results.items():

    print(
        f"{metric}: "
        f"{values['mean']:.4f} "
        f"(+/- {values['std']:.4f})"
    )


# =====================================================
# AUDITORIA
# =====================================================

print("\n" + "=" * 60)
print("AUDITORIA DAS FEATURES")
print("=" * 60)

for fold in evaluator.last_feature_audit_:

    print(f"\nFOLD {fold['fold']}")

    print(
        "Features geradas:",
        len(fold["generated_features"])
    )

    print(
        "Sugestoes Gemini:",
        len(fold["llm_suggestions"])
    )

    print(
        "Oportunidades LLM:",
        len(fold["llm_opportunities"])
    )

    print(
        "Sugestoes rejeitadas:",
        len(fold["llm_rejected_suggestions"])
    )

    print(
        "Features:",
        fold["generated_features"]
    )


print("\nTeste concluido!")
