
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression
)


class FeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        original_columns=None,
        k=3,
        min_score=0.0,
        random_state=42,
        problem_type="classification"
    ):

        self.original_columns = original_columns
        self.k = k
        self.min_score = min_score
        self.random_state = random_state
        self.problem_type = problem_type

    # ==========================================
    # VALIDACOES
    # ==========================================

    def _validate_parameters(self):

        if (
            isinstance(self.k, bool)
            or not isinstance(self.k, (int, np.integer))
            or self.k < 1
        ):
            raise ValueError(
                "k deve ser um numero inteiro maior ou igual a 1."
            )

        if (
            isinstance(self.min_score, bool)
            or not isinstance(self.min_score, (int, float, np.number))
            or not np.isfinite(self.min_score)
            or self.min_score < 0
        ):
            raise ValueError(
                "min_score deve ser um numero finito maior ou igual a zero."
            )

        if (
            self.random_state is not None
            and (
                isinstance(self.random_state, bool)
                or not isinstance(
                    self.random_state,
                    (int, np.integer)
                )
                or self.random_state < 0
            )
        ):
            raise ValueError(
                "random_state deve ser um inteiro nao negativo ou None."
            )

        if self.problem_type not in (
            "classification",
            "regression"
        ):
            raise ValueError(
                "problem_type deve ser 'classification' "
                "ou 'regression'."
            )

    def _validate_X(self, X):

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X deve ser um DataFrame.")

        if X.empty:
            raise ValueError("X esta vazio.")

        if X.columns.duplicated().any():
            raise ValueError(
                "X possui nomes de colunas duplicados."
            )

    def _validate_original_columns(self, X):

        if self.original_columns is None:
            raise ValueError(
                "Informe as colunas originais para identificar "
                "as features candidatas."
            )

        if isinstance(self.original_columns, str):
            raise TypeError(
                "original_columns deve ser uma lista ou tupla "
                "de nomes de colunas."
            )

        try:
            original_columns = list(self.original_columns)

        except TypeError:
            raise TypeError(
                "original_columns deve ser uma lista ou tupla."
            )

        if not original_columns:
            raise ValueError(
                "A lista de colunas originais nao pode estar vazia."
            )

        if len(original_columns) != len(set(original_columns)):
            raise ValueError(
                "original_columns possui colunas duplicadas."
            )

        missing_columns = [
            column
            for column in original_columns
            if column not in X.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Colunas originais ausentes: {missing_columns}"
            )

        return original_columns

    def _prepare_target(self, y, n_samples):

        if y is None:
            raise ValueError("A variavel alvo y e obrigatoria.")

        if isinstance(y, pd.DataFrame):
            raise ValueError(
                "y deve possuir apenas uma coluna."
            )

        try:
            y_array = np.asarray(y)

        except (TypeError, ValueError):
            raise ValueError(
                "Nao foi possivel converter y para um vetor."
            )

        if y_array.ndim != 1:
            raise ValueError(
                "y deve ser unidimensional."
            )

        if len(y_array) != n_samples:
            raise ValueError(
                "X e y possuem quantidades diferentes de linhas."
            )

        if pd.isna(y_array).any():
            raise ValueError(
                "A variavel alvo possui valores ausentes."
            )

        # ----------------------------------
        # CLASSIFICACAO
        # ----------------------------------

        if self.problem_type == "classification":

            if len(np.unique(y_array)) < 2:
                raise ValueError(
                    "Classificacao exige pelo menos duas classes."
                )

            if np.issubdtype(y_array.dtype, np.number):

                if not np.isfinite(y_array.astype(float)).all():
                    raise ValueError(
                        "O target possui valores infinitos."
                    )

            return y_array

        # ----------------------------------
        # REGRESSAO
        # ----------------------------------

        try:
            y_numeric = pd.to_numeric(
                pd.Series(y_array),
                errors="raise"
            ).to_numpy(dtype=float)

        except (ValueError, TypeError):
            raise ValueError(
                "Para regressao, y deve possuir valores numericos."
            )

        if not np.isfinite(y_numeric).all():
            raise ValueError(
                "O target de regressao possui valores "
                "infinitos ou nao finitos."
            )

        if np.unique(y_numeric).size < 2:
            raise ValueError(
                "Regressao exige variacao na variavel alvo."
            )

        return y_numeric

    # ==========================================
    # PREPARACAO DAS FEATURES CANDIDATAS
    # ==========================================

    def _prepare_candidates(self, X):

        candidate_data = X[self.candidate_columns_].copy()

        for column in self.candidate_columns_:

            # Converte valores numericos.
            candidate_data[column] = pd.to_numeric(
                candidate_data[column],
                errors="coerce"
            )

            values = candidate_data[column].to_numpy(
                dtype=float,
                na_value=np.nan
            )

            # Substitui infinitos por valores ausentes.
            values[~np.isfinite(values)] = np.nan

            candidate_data[column] = values

            # Verifica se existe pelo menos um valor valido.
            if candidate_data[column].notna().any():

                median = candidate_data[column].median()

            else:

                # Se todos os valores forem ausentes,
                # utiliza zero como valor de preenchimento.
                median = 0.0

            # Imputacao pela mediana.
            candidate_data[column] = (
                candidate_data[column].fillna(median)
            )

        return candidate_data

    # ==========================================
    # FIT
    # ==========================================

    def fit(self, X, y):

        self._validate_parameters()
        self._validate_X(X)

        original_columns = self._validate_original_columns(X)

        y_array = self._prepare_target(y, len(X))

        self.original_columns_ = original_columns

        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(
            X.columns,
            dtype=object
        )

        # ----------------------------------
        # IDENTIFICACAO DAS CANDIDATAS
        # ----------------------------------

        self.candidate_columns_ = [
            column
            for column in X.columns
            if column not in self.original_columns_
        ]

        self.feature_scores_ = {}

        # ----------------------------------
        # SEM FEATURES CANDIDATAS
        # ----------------------------------

        if not self.candidate_columns_:

            self.selected_features_ = []

            return self

        # ----------------------------------
        # PREPARACAO DOS DADOS
        # ----------------------------------

        candidate_data = self._prepare_candidates(X)

        # ----------------------------------
        # IDENTIFICACAO DE FEATURES VALIDAS
        # ----------------------------------

        valid_columns = []

        for column in self.candidate_columns_:

            values = candidate_data[column].to_numpy(
                dtype=float
            )

            # Features constantes nao possuem informacao util
            # para o calculo de informacao mutua.
            if np.unique(values).size <= 1:

                self.feature_scores_[column] = 0.0

            else:
                valid_columns.append(column)

        # ----------------------------------
        # CALCULO DA INFORMACAO MUTUA
        # ----------------------------------

        if valid_columns:

            valid_data = candidate_data[
                valid_columns
            ].to_numpy(dtype=float)

            try:

                if self.problem_type == "classification":

                    scores = mutual_info_classif(
                        valid_data,
                        y_array,
                        random_state=self.random_state
                    )

                else:

                    scores = mutual_info_regression(
                        valid_data,
                        y_array,
                        random_state=self.random_state
                    )

            except (ValueError, TypeError, FloatingPointError) as error:

                raise ValueError(
                    "Falha no calculo da informacao mutua: "
                    f"{error}"
                ) from error

            # ----------------------------------
            # ARMAZENAMENTO DAS PONTUACOES
            # ----------------------------------

            for column, score in zip(valid_columns, scores):

                score = float(score)

                if not np.isfinite(score):
                    score = 0.0

                self.feature_scores_[column] = score

        # ----------------------------------
        # RANKING DAS FEATURES
        # ----------------------------------

        ranked_features = sorted(
            self.feature_scores_,
            key=self.feature_scores_.get,
            reverse=True
        )

        # ----------------------------------
        # SELECAO FINAL
        # ----------------------------------

        self.selected_features_ = [
            column
            for column in ranked_features
            if self.feature_scores_[column] > self.min_score
        ][:self.k]

        return self

    # ==========================================
    # TRANSFORM
    # ==========================================

    def transform(self, X):

        if not hasattr(self, "selected_features_"):
            raise RuntimeError(
                "O selector precisa executar fit antes de transform."
            )

        self._validate_X(X)

        required_columns = (
            self.original_columns_ + self.selected_features_
        )

        missing_columns = [
            column
            for column in required_columns
            if column not in X.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Colunas necessarias ausentes: {missing_columns}"
            )

        return X[required_columns].copy()

    # ==========================================
    # FIT_TRANSFORM
    # ==========================================

    def fit_transform(self, X, y):

        return self.fit(X, y).transform(X)

    # ==========================================
    # PONTUACAO DAS FEATURES
    # ==========================================

    def get_feature_scores(self):

        if not hasattr(self, "feature_scores_"):
            raise RuntimeError(
                "O selector precisa executar fit primeiro."
            )

        return dict(
            sorted(
                self.feature_scores_.items(),
                key=lambda item: item[1],
                reverse=True
            )
        )

    # ==========================================
    # PROPRIEDADE: TIPO DO PROBLEMA
    # ==========================================

    @property
    def problem_type(self):

        return self._problem_type

    @problem_type.setter
    def problem_type(self, value):

        if value not in (
            "classification",
            "regression"
        ):
            raise ValueError(
                "problem_type deve ser 'classification' "
                "ou 'regression'."
            )

        self._problem_type = value
