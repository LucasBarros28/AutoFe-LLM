
import re
import pandas as pd
import numpy as np

from itertools import combinations

from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression
)


class OpportunityDetector:

    VALID_ROLES = [
        "target",
        "identifier",
        "binary",
        "numerical_discrete",
        "numerical_continuous",
        "categorical_nominal",
        "categorical_ordinal",
        "high_cardinality_categorical"
    ]

    def __init__(self, df, target=None, column_roles=None):

        if not isinstance(df, pd.DataFrame):
            raise TypeError("A entrada deve ser um DataFrame.")

        if df.empty:
            raise ValueError("O DataFrame está vazio.")

        if not df.columns.is_unique:
            raise ValueError("O DataFrame possui nomes de colunas duplicados.")

        if target is not None and target not in df.columns:
            raise ValueError("Variável alvo não encontrada.")

        if column_roles is not None and not isinstance(column_roles, dict):
            raise TypeError("column_roles deve ser um dicionário.")

        self.df = df.copy()
        self.target = target
        self.column_roles = column_roles.copy() if column_roles else {}

        # Valida as colunas e os papéis informados manualmente.
        for column, role in self.column_roles.items():

            if column not in self.df.columns:
                raise ValueError(
                    f"A coluna '{column}' informada em column_roles "
                    "não existe no DataFrame."
                )

            if role not in self.VALID_ROLES:
                raise ValueError(
                    f"Papel inválido '{role}' para a coluna '{column}'."
                )

            if role == "target" and column != self.target:
                raise ValueError(
                    "Somente a coluna indicada em target pode possuir "
                    "o papel 'target'."
                )

    # ==========================================
    # IDENTIFICACAO DE COLUNAS
    # ==========================================

    def is_identifier(self, column):

        if not isinstance(column, str):
            return False

        normalized = re.sub(
            r"([a-z0-9])([A-Z])",
            r"\1_\2",
            column
        ).lower()

        parts = normalized.split("_")

        identifier_names = [
            "id",
            "uuid",
            "identifier"
        ]

        return any(
            part in identifier_names
            for part in parts
        )

    # ==========================================
    # CLASSIFICACAO DAS COLUNAS
    # ==========================================

    def classify_columns(self):

        result = {}

        for column in self.df.columns:

            series = self.df[column]

            unique = series.nunique()
            unique_ratio = unique / len(self.df)

            if column == self.target:
                role = "target"

            elif column in self.column_roles:
                role = self.column_roles[column]

            elif self.is_identifier(column):
                role = "identifier"

            elif pd.api.types.is_bool_dtype(series):
                role = "binary"

            elif pd.api.types.is_numeric_dtype(series):

                if unique <= 2:
                    role = "binary"

                elif pd.api.types.is_integer_dtype(series):
                    role = "numerical_discrete"

                else:
                    role = "numerical_continuous"

            else:

                if unique_ratio > 0.5:
                    role = "high_cardinality_categorical"

                else:
                    role = "categorical_nominal"

            result[column] = {
                "role": role,
                "unique_values": int(unique),
                "unique_ratio": round(unique_ratio, 4)
            }

        return result

    # ==========================================
    # VALIDACAO DO TARGET
    # ==========================================

    def _prepare_target(self, y, problem_type):

        if y is None:
            return None, None

        y = pd.Series(y).reset_index(drop=True)

        if len(y) != len(self.df):
            raise ValueError(
                "O tamanho de y deve ser igual ao número "
                "de linhas de df."
            )

        # Remove valores ausentes.
        valid_target = y.notna().to_numpy()

        y_valid = y.to_numpy()[valid_target]

        if problem_type == "regression":

            try:
                y_valid = pd.to_numeric(
                    pd.Series(y_valid),
                    errors="raise"
                ).to_numpy(dtype=float)

            except (ValueError, TypeError):
                raise ValueError(
                    "O target de regressão precisa ser numérico."
                )

            # Remove valores infinitos.
            finite_target = np.isfinite(y_valid)

            y_valid = y_valid[finite_target]

            original_positions = np.flatnonzero(valid_target)

            valid_positions = original_positions[finite_target]

            valid_target = np.zeros(len(y), dtype=bool)
            valid_target[valid_positions] = True

            if len(y_valid) < 4:
                raise ValueError(
                    "São necessárias pelo menos 4 observações "
                    "válidas para calcular a informação mútua."
                )

            if np.unique(y_valid).size < 2:
                raise ValueError(
                    "O target de regressão precisa possuir variação."
                )

        else:

            if len(y_valid) < 4:
                raise ValueError(
                    "São necessárias pelo menos 4 observações "
                    "válidas para calcular a informação mútua."
                )

            if pd.Series(y_valid).nunique() < 2:
                raise ValueError(
                    "O target precisa possuir pelo menos duas classes."
                )

        return y_valid, valid_target

    # ==========================================
    # DETECCAO DE INTERACOES
    # ==========================================

    def detect_interactions(
        self,
        column_roles=None,
        max_interactions=10,
        min_unique=3,
        y=None,
        problem_type="classification",
        random_state=42
    ):

        if (
            not isinstance(max_interactions, (int, np.integer))
            or isinstance(max_interactions, (bool, np.bool_))
            or max_interactions < 1
        ):
            raise ValueError(
                "max_interactions deve ser um inteiro maior que zero."
            )

        if (
            not isinstance(min_unique, (int, np.integer))
            or isinstance(min_unique, (bool, np.bool_))
            or min_unique < 2
        ):
            raise ValueError(
                "min_unique deve ser um inteiro maior ou igual a 2."
            )

        if problem_type not in ["classification", "regression"]:
            raise ValueError(
                "problem_type deve ser 'classification' ou 'regression'."
            )

        if column_roles is None:
            column_roles = self.classify_columns()

        if not isinstance(column_roles, dict):
            raise TypeError("column_roles deve ser um dicionário.")

        # Valida os papéis utilizados na detecção.
        for column, information in column_roles.items():

            if column not in self.df.columns:
                raise ValueError(
                    f"Coluna '{column}' não encontrada no DataFrame."
                )

            if not isinstance(information, dict):
                raise ValueError(
                    f"As informações da coluna '{column}' devem "
                    "ser um dicionário."
                )

            if "role" not in information:
                raise ValueError(
                    f"O papel da coluna '{column}' não foi informado."
                )

        # ==========================================
        # PREPARACAO DO TARGET
        # ==========================================

        y_valid, valid_target = self._prepare_target(
            y,
            problem_type
        )

        # ==========================================
        # SELECAO DAS COLUNAS ELEGIVEIS
        # ==========================================

        eligible_columns = []

        for column, information in column_roles.items():

            role = information["role"]

            if role not in [
                "numerical_continuous",
                "numerical_discrete"
            ]:
                continue

            if column == self.target:
                continue

            if self.is_identifier(column):
                continue

            if not pd.api.types.is_numeric_dtype(self.df[column]):
                continue

            # Converte os dados para uma representação numérica segura.
            values = pd.to_numeric(
                self.df[column],
                errors="coerce"
            ).replace(
                [np.inf, -np.inf],
                np.nan
            )

            if values.nunique() < min_unique:
                continue

            eligible_columns.append(column)

        # ==========================================
        # GERACAO E AVALIACAO DOS PARES
        # ==========================================

        candidates = []

        # Não cria uma lista com todas as combinações.
        # Isso reduz o consumo de memória em datasets grandes.
        for column_a, column_b in combinations(
            eligible_columns,
            2
        ):

            values_a = pd.to_numeric(
                self.df[column_a],
                errors="coerce"
            ).replace(
                [np.inf, -np.inf],
                np.nan
            )

            values_b = pd.to_numeric(
                self.df[column_b],
                errors="coerce"
            ).replace(
                [np.inf, -np.inf],
                np.nan
            )

            interaction = values_a * values_b

            interaction = interaction.replace(
                [np.inf, -np.inf],
                np.nan
            )

            if interaction.nunique() < 2:
                continue

            # Imputação da interação usando somente dados do fold
            # recebido pelo detector.
            median = interaction.median()

            if pd.isna(median):
                continue

            interaction = interaction.fillna(median)

            # ==========================================
            # CALCULO DA PONTUACAO
            # ==========================================

            if y_valid is not None:

                interaction_values = interaction.to_numpy(
                    dtype=float
                )

                interaction_valid = interaction_values[valid_target]

                # Segurança adicional para evitar valores inválidos.
                if not np.isfinite(interaction_valid).all():
                    continue

                try:

                    if problem_type == "classification":

                        score = mutual_info_classif(
                            interaction_valid.reshape(-1, 1),
                            y_valid,
                            discrete_features=False,
                            random_state=random_state
                        )[0]

                    else:

                        score = mutual_info_regression(
                            interaction_valid.reshape(-1, 1),
                            y_valid,
                            discrete_features=False,
                            random_state=random_state
                        )[0]

                except (ValueError, TypeError, FloatingPointError):
                    continue

            else:

                score = interaction.var()

            if not np.isfinite(score):
                continue

            candidates.append({
                "columns": [column_a, column_b],
                "feature": f"{column_a}_x_{column_b}",
                "type": "interaction_product",
                "severity": "experimental",
                "score": float(score),
                "recommendation": "Evaluate multiplication interaction",
                "reason": (
                    f"Interaction candidate between {column_a} "
                    f"and {column_b}. Score = {score:.5f}"
                )
            })

        # ==========================================
        # RANQUEAMENTO
        # ==========================================

        candidates.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return candidates[:max_interactions]

    # ==========================================
    # DETECCAO DE OPORTUNIDADES ESTATISTICAS
    # ==========================================

    def detect(self):

        opportunities = []

        column_roles = self.classify_columns()

        for column in self.df.columns:

            series = self.df[column]

            role = column_roles[column]["role"]

            # Não analisar target e identificadores.
            if role in ["target", "identifier"]:
                continue

            # ==========================================
            # VALORES AUSENTES
            # ==========================================

            missing_percentage = series.isna().mean() * 100

            if missing_percentage > 5:

                opportunities.append({
                    "column": column,
                    "type": "missing_values",
                    "severity": "high",
                    "recommendation": "Evaluate missing value treatment",
                    "reason": (
                        f"{missing_percentage:.2f}% "
                        "of values are missing."
                    )
                })

            # ==========================================
            # COLUNAS CONSTANTES
            # ==========================================

            if series.nunique() <= 1:

                opportunities.append({
                    "column": column,
                    "type": "constant_column",
                    "severity": "medium",
                    "recommendation": "Evaluate column removal",
                    "reason": "Column has only one unique value."
                })

                continue

            # ==========================================
            # ANALISE NUMERICA
            # ==========================================

            if role not in [
                "numerical_continuous",
                "numerical_discrete"
            ]:
                continue

            values = pd.to_numeric(
                series,
                errors="coerce"
            )

            values = values.replace(
                [np.inf, -np.inf],
                np.nan
            ).dropna()

            if len(values) < 2:
                continue

            if values.nunique() <= 1:
                continue

            # ==========================================
            # ASSIMETRIA
            # ==========================================

            skewness = values.skew()

            if pd.notna(skewness) and abs(skewness) >= 1:

                if values.min() >= 0:

                    recommendation = (
                        "Evaluate log1p transformation"
                    )

                else:

                    recommendation = (
                        "Evaluate Yeo-Johnson transformation"
                    )

                opportunities.append({
                    "column": column,
                    "type": "high_skewness",
                    "severity": "medium",
                    "recommendation": recommendation,
                    "reason": f"Skewness = {skewness:.2f}"
                })

            # ==========================================
            # OUTLIERS UTILIZANDO IQR
            # ==========================================

            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)

            iqr = q3 - q1

            if iqr == 0:
                continue

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outliers = (
                (values < lower) | (values > upper)
            ).sum()

            outlier_percentage = (
                outliers / len(values)
            ) * 100

            if outlier_percentage > 5:

                opportunities.append({
                    "column": column,
                    "type": "possible_outliers",
                    "severity": "medium",
                    "recommendation": "Evaluate outlier treatment",
                    "reason": (
                        f"{outlier_percentage:.2f}% "
                        "of values outside IQR limits."
                    )
                })

        return opportunities
