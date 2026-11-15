
from sklearn.datasets import load_diabetes, load_breast_cancer

from src.evaluation.model_evaluator import ModelEvaluator


# =====================================================
# TESTE DE REGRESSAO
# =====================================================

print("\n===== TESTANDO MODELOS DE REGRESSAO =====")

diabetes = load_diabetes(as_frame=True)

X_reg = diabetes.data
y_reg = diabetes.target

regression_models = [
    "decision_tree",
    "random_forest",
    "gradient_boosting",
    "dummy"
]

for model_name in regression_models:

    print(f"\nModelo: {model_name}")

    evaluator = ModelEvaluator(
        problem_type="regression",
        random_state=42,
        model_name=model_name
    )

    results = evaluator.evaluate(
        X_reg,
        y_reg
    )

    for metric, values in results.items():

        print(
            f"{metric.upper()}: "
            f"{values['mean']:.4f} "
            f"+/- {values['std']:.4f}"
        )


# =====================================================
# TESTE DE CLASSIFICACAO
# =====================================================

print("\n===== TESTANDO MODELOS DE CLASSIFICACAO =====")

cancer = load_breast_cancer(as_frame=True)

X_clf = cancer.data
y_clf = cancer.target

classification_models = [
    "decision_tree",
    "random_forest",
    "gradient_boosting",
    "dummy"
]

for model_name in classification_models:

    print(f"\nModelo: {model_name}")

    evaluator = ModelEvaluator(
        problem_type="classification",
        random_state=42,
        model_name=model_name
    )

    results = evaluator.evaluate(
        X_clf,
        y_clf
    )

    for metric, values in results.items():

        print(
            f"{metric.upper()}: "
            f"{values['mean']:.4f} "
            f"+/- {values['std']:.4f}"
        )


print("\n===== TESTES FINALIZADOS =====")
