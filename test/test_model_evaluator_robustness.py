
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_validate

from src.evaluation.model_evaluator import ModelEvaluator


# ==========================================
# DADOS DE TESTE
# ==========================================

def create_classification_data():

    np.random.seed(42)

    X = pd.DataFrame({
        "Age": np.random.randint(18, 70, 40),
        "Fare": np.random.uniform(10, 100, 40),
        "Sex": ["male", "female"] * 20,
        "Embarked": ["S", "C", "Q", "S"] * 10
    })

    X.loc[0:3, "Age"] = np.nan
    X.loc[4:6, "Fare"] = np.nan
    X.loc[7, "Sex"] = np.nan

    y = np.array([0, 1] * 20)

    return X, y


def create_regression_data():

    np.random.seed(42)

    X = pd.DataFrame({
        "Age": np.random.uniform(18, 70, 40),
        "Income": np.random.uniform(1000, 10000, 40),
        "Category": ["A", "B"] * 20
    })

    y = (
        X["Age"] * 2
        + X["Income"] * 0.01
        + np.random.normal(0, 2, 40)
    )

    return X, y


# ==========================================
# TESTE 1: CLASSIFICACAO
# ==========================================

def test_classification():

    X, y = create_classification_data()

    evaluator = ModelEvaluator(
        problem_type="classification"
    )

    results = evaluator.evaluate(X, y)

    expected_metrics = {
        "accuracy",
        "precision",
        "recall",
        "f1"
    }

    assert set(results.keys()) == expected_metrics

    for metric, values in results.items():

        assert np.isfinite(values["mean"])
        assert np.isfinite(values["std"])
        assert len(values["folds"]) == 5

    print("Teste 1 aprovado: classificacao.")


# ==========================================
# TESTE 2: REGRESSAO
# ==========================================

def test_regression():

    X, y = create_regression_data()

    evaluator = ModelEvaluator(
        problem_type="regression"
    )

    results = evaluator.evaluate(X, y)

    expected_metrics = {
        "mae",
        "rmse",
        "r2"
    }

    assert set(results.keys()) == expected_metrics

    assert results["mae"]["mean"] >= 0
    assert results["rmse"]["mean"] >= 0

    for metric, values in results.items():

        assert np.isfinite(values["mean"])
        assert np.isfinite(values["std"])
        assert len(values["folds"]) == 5

    print("Teste 2 aprovado: regressao.")


# ==========================================
# TESTE 3: VALORES AUSENTES
# ==========================================

def test_missing_values():

    X, y = create_classification_data()

    evaluator = ModelEvaluator(
        problem_type="classification"
    )

    results = evaluator.evaluate(X, y)

    assert np.isfinite(results["accuracy"]["mean"])

    print("Teste 3 aprovado: valores ausentes.")


# ==========================================
# TESTE 4: TAMANHOS DIFERENTES
# ==========================================

def test_inconsistent_dimensions():

    X, y = create_classification_data()

    evaluator = ModelEvaluator()

    try:

        evaluator.evaluate(X, y[:-1])

        raise AssertionError(
            "Deveria rejeitar dimensoes inconsistentes."
        )

    except ValueError:
        pass

    print("Teste 4 aprovado: dimensoes inconsistentes.")


# ==========================================
# TESTE 5: TARGET AUSENTE
# ==========================================

def test_missing_target():

    X, y = create_classification_data()

    y = y.astype(float)
    y[0] = np.nan

    evaluator = ModelEvaluator()

    try:

        evaluator.evaluate(X, y)

        raise AssertionError(
            "Deveria rejeitar target com valores ausentes."
        )

    except ValueError:
        pass

    print("Teste 5 aprovado: target ausente.")


# ==========================================
# TESTE 6: TARGET COM UMA CLASSE
# ==========================================

def test_single_class():

    X, y = create_classification_data()

    y = np.zeros(len(y))

    evaluator = ModelEvaluator()

    try:

        evaluator.evaluate(X, y)

        raise AssertionError(
            "Deveria rejeitar classificacao com uma classe."
        )

    except ValueError:
        pass

    print("Teste 6 aprovado: target com uma classe.")


# ==========================================
# TESTE 7: TARGET NAO NUMERICO EM REGRESSAO
# ==========================================

def test_invalid_regression_target():

    X, y = create_regression_data()

    y = np.array(["A", "B"] * 20)

    evaluator = ModelEvaluator(
        problem_type="regression"
    )

    try:

        evaluator.evaluate(X, y)

        raise AssertionError(
            "Deveria rejeitar target nao numerico."
        )

    except ValueError:
        pass

    print("Teste 7 aprovado: target invalido para regressao.")


# ==========================================
# TESTE 8: COLUNAS DUPLICADAS
# ==========================================

def test_duplicate_columns():

    X, y = create_classification_data()

    X["Age_copy"] = X["Age"]

    X.columns = [
        "Age",
        "Fare",
        "Sex",
        "Embarked",
        "Age"
    ]

    evaluator = ModelEvaluator()

    try:

        evaluator.evaluate(X, y)

        raise AssertionError(
            "Deveria rejeitar colunas duplicadas."
        )

    except ValueError:
        pass

    print("Teste 8 aprovado: colunas duplicadas.")


# ==========================================
# TESTE 9: SELECAO SEM AUTOFE
# ==========================================

def test_selection_without_autofe():

    X, y = create_classification_data()

    evaluator = ModelEvaluator()

    try:

        evaluator.evaluate(
            X,
            y,
            use_autofe=False,
            use_feature_selection=True
        )

        raise AssertionError(
            "Deveria exigir AutoFE para selecao."
        )

    except ValueError:
        pass

    print("Teste 9 aprovado: selecao exige AutoFE.")


# ==========================================
# TESTE 10: AUDITORIA DOS FOLDS
# ==========================================

def test_fold_audit():

    X, y = create_classification_data()

    evaluator = ModelEvaluator(
        problem_type="classification"
    )

    evaluator.evaluate(
        X,
        y,
        use_autofe=True,
        use_feature_selection=True,
        selector_k=3,
        include_interactions=False
    )

    audit = evaluator.last_feature_audit_

    assert len(audit) == 5

    for fold in audit:

        assert "opportunities" in fold
        assert "generated_features" in fold
        assert "feature_scores" in fold
        assert "selected_features" in fold

    print("Teste 10 aprovado: auditoria dos cinco folds.")


# ==========================================
# TESTE 11: VERIFICACAO DE LEAKAGE
# ==========================================

def test_preprocessing_without_leakage():

    X = pd.DataFrame({
        "Value": [1, 2, 3, 1000, 5, 6]
    })

    y = np.array([0, 1, 0, 1, 0, 1])

    evaluator = ModelEvaluator()

    pipeline = evaluator.build_model_pipeline(
        original_columns=list(X.columns)
    )

    # O valor 1000 pertence exclusivamente ao teste.
    train_indices = np.array([0, 1, 2, 4, 5])
    test_indices = np.array([3])

    cv = [
        (train_indices, test_indices)
    ]

    results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="accuracy",
        return_estimator=True,
        error_score="raise"
    )

    fitted_pipeline = results["estimator"][0]

    preprocessor = fitted_pipeline.named_steps[
        "preprocessor"
    ]

    numerical_pipeline = (
        preprocessor.named_transformers_["numerical"]
    )

    imputer = numerical_pipeline.named_steps["imputer"]

    learned_median = imputer.statistics_[0]

    expected_median = np.median(
        X.iloc[train_indices]["Value"]
    )

    assert learned_median == expected_median

    assert learned_median != np.median(
        X["Value"]
    )

    print(
        "Teste 11 aprovado: preprocessamento "
        "ajustado somente no treinamento."
    )


# ==========================================
# EXECUCAO DOS TESTES
# ==========================================

if __name__ == "__main__":

    print("\n===== TESTES DE ROBUSTEZ DO MODELEVALUATOR =====\n")

    test_classification()
    test_regression()
    test_missing_values()
    test_inconsistent_dimensions()
    test_missing_target()
    test_single_class()
    test_invalid_regression_target()
    test_duplicate_columns()
    test_selection_without_autofe()
    test_fold_audit()
    test_preprocessing_without_leakage()

    print(
        "\nTodos os testes do ModelEvaluator foram aprovados!"
    )
