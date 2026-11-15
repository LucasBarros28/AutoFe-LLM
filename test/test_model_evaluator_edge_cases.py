
import numpy as np
import pandas as pd

from src.evaluation.model_evaluator import ModelEvaluator


def test_colunas_completamente_vazias():

    print("\n===== TESTE 1: COLUNAS COMPLETAMENTE VAZIAS =====")

    X = pd.DataFrame({
        "Age": [22, 25, 30, 35, 40, 45, 50, 55, 60, 65],
        "Fare": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "Sex": [
            "male", "female", "male", "female", "male",
            "female", "male", "female", "male", "female"
        ],
        "Empty_Numeric": [np.nan] * 10,
        "Empty_Categorical": [np.nan] * 10
    })

    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    evaluator = ModelEvaluator(problem_type="classification")

    pipeline = evaluator.build_model_pipeline(
        original_columns=list(X.columns),
        use_autofe=False
    )

    try:
        pipeline.fit(X, y)

        predictions = pipeline.predict(X)

        assert len(predictions) == len(X)
        assert not pd.isna(predictions).any()

        print("Teste 1 aprovado: colunas completamente vazias.")

    except Exception as error:
        print("Teste 1 falhou.")
        print("Erro:", repr(error))
        raise


def test_categoria_inedita_na_validacao():

    print("\n===== TESTE 2: CATEGORIA INEDITA NA VALIDACAO =====")

    X = pd.DataFrame({
        "Age": [20, 25, 30, 35, 40, 45, 50, 55, 60, 65],
        "Fare": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "Sex": [
            "male", "female", "male", "female", "male",
            "female", "male", "female", "male", "female"
        ],
        "Embarked": [
            "S", "C", "S", "C", "S",
            "C", "S", "C", "NEW", "NEW"
        ]
    })

    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    # Treinamento: somente categorias S e C.
    X_train = X.iloc[:8].copy()
    y_train = y.iloc[:8].copy()

    # Validacao: categoria NEW nunca vista no treinamento.
    X_test = X.iloc[8:].copy()

    evaluator = ModelEvaluator(problem_type="classification")

    pipeline = evaluator.build_model_pipeline(
        original_columns=list(X.columns),
        use_autofe=False
    )

    try:
        pipeline.fit(X_train, y_train)

        preprocessor = pipeline.named_steps["preprocessor"]

        transformed_train = preprocessor.transform(X_train)
        transformed_test = preprocessor.transform(X_test)

        predictions = pipeline.predict(X_test)

        # A quantidade de features deve permanecer igual.
        assert transformed_train.shape[1] == transformed_test.shape[1]

        # Todas as linhas de validacao devem gerar previsoes.
        assert len(predictions) == len(X_test)

        assert not pd.isna(predictions).any()

        print("Categorias no treinamento:", sorted(X_train["Embarked"].unique()))
        print("Categoria nova na validacao:", "NEW")
        print("Features apos preprocessamento:", transformed_test.shape[1])

        print("Teste 2 aprovado: categoria inedita tratada corretamente.")

    except Exception as error:
        print("Teste 2 falhou.")
        print("Erro:", repr(error))
        raise


def main():

    print("\n==============================================")
    print("TESTES DE CASOS EXTREMOS DO MODELEVALUATOR")
    print("==============================================")

    test_colunas_completamente_vazias()

    test_categoria_inedita_na_validacao()

    print("\nTodos os testes de casos extremos foram aprovados!")


if __name__ == "__main__":
    main()
