from sklearn.datasets import load_breast_cancer, load_diabetes

from src.evaluation.model_evaluator import ModelEvaluator


# ==========================================
# TESTE 1: CLASSIFICAÇÃO
# ==========================================

dataset_classification = load_breast_cancer(as_frame=True)

X_classification = dataset_classification.data
y_classification = dataset_classification.target

evaluator_classification = ModelEvaluator(
    problem_type="classification"
)

results_classification = evaluator_classification.evaluate(
    X_classification,
    y_classification
)

print("\n===== TESTE DE CLASSIFICAÇÃO =====")

for metric, values in results_classification.items():

    print(f"\n{metric}")
    print(f"Média: {values['mean']:.4f}")
    print(f"Desvio: {values['std']:.4f}")
    print(f"Folds: {values['folds']}")


# ==========================================
# TESTE 2: REGRESSÃO
# ==========================================

dataset_regression = load_diabetes(as_frame=True)

X_regression = dataset_regression.data
y_regression = dataset_regression.target

evaluator_regression = ModelEvaluator(
    problem_type="regression"
)

results_regression = evaluator_regression.evaluate(
    X_regression,
    y_regression
)

print("\n===== TESTE DE REGRESSÃO =====")

for metric, values in results_regression.items():

    print(f"\n{metric}")
    print(f"Média: {values['mean']:.4f}")
    print(f"Desvio: {values['std']:.4f}")
    print(f"Folds: {values['folds']}")