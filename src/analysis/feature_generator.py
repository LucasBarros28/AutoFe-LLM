
import pandas as pd
import numpy as np


class FeatureGenerator:

    def __init__(self, df):

        if not isinstance(df, pd.DataFrame):
            raise TypeError("A entrada deve ser um DataFrame.")

        if df.empty:
            raise ValueError("O DataFrame está vazio.")

        if df.columns.duplicated().any():
            raise ValueError("O DataFrame possui nomes de colunas duplicados.")

        self.df = df.copy()

    # ==========================================
    # VALIDACOES AUXILIARES
    # ==========================================

    def _validate_column(self, column):

        if not isinstance(column, str) or not column:
            raise ValueError("O nome da coluna deve ser uma string valida.")

        if column not in self.df.columns:
            raise ValueError(f"Coluna '{column}' nao encontrada.")

    def _validate_numeric_column(self, column):

        self._validate_column(column)

        series = self.df[column]

        if not pd.api.types.is_numeric_dtype(series):
            raise ValueError(f"Coluna '{column}' nao e numerica.")

        if pd.api.types.is_bool_dtype(series):
            raise ValueError(f"Coluna '{column}' possui tipo booleano.")

        values = series.to_numpy(dtype=float, na_value=np.nan)

        if np.isinf(values).any():
            raise ValueError(
                f"A coluna '{column}' possui valores infinitos."
            )

    def _validate_new_column(self, new_column):

        if new_column in self.df.columns:
            raise ValueError(
                f"A coluna '{new_column}' ja existe. "
                "A transformacao nao sera realizada."
            )

    # ==========================================
    # TRANSFORMACAO LOGARITMICA
    # ==========================================

    def generate_log_transform(self, column):

        self._validate_numeric_column(column)

        new_column = f"{column}_log1p"

        self._validate_new_column(new_column)

        # Ignora valores ausentes na verificacao.
        valid_values = self.df[column].dropna()

        if (valid_values < 0).any():
            raise ValueError(
                "A transformacao log1p exige valores maiores "
                "ou iguais a zero."
            )

        transformed = np.log1p(self.df[column])

        if np.isinf(
            transformed.to_numpy(dtype=float, na_value=np.nan)
        ).any():
            raise ValueError(
                f"A transformacao de '{column}' produziu "
                "valores infinitos."
            )

        self.df[new_column] = transformed

        return self.df.copy()

    # ==========================================
    # INDICADOR DE VALORES AUSENTES
    # ==========================================

    def generate_missing_indicator(self, column):

        self._validate_column(column)

        new_column = f"{column}_is_missing"

        self._validate_new_column(new_column)

        self.df[new_column] = self.df[column].isna().astype(int)

        return self.df.copy()

    # ==========================================
    # INTERACAO ENTRE VARIAVEIS
    # ==========================================

    def generate_interaction(self, column_a, column_b):

        self._validate_numeric_column(column_a)
        self._validate_numeric_column(column_b)

        if column_a == column_b:
            raise ValueError(
                "Nao e permitido gerar interacao de uma coluna com ela mesma."
            )

        new_column = f"{column_a}_x_{column_b}"

        self._validate_new_column(new_column)

        interaction = self.df[column_a] * self.df[column_b]

        # Verifica se a multiplicacao produziu valores infinitos.
        values = interaction.to_numpy(dtype=float, na_value=np.nan)

        if np.isinf(values).any():
            raise ValueError(
                f"A interacao '{new_column}' produziu valores infinitos."
            )

        self.df[new_column] = interaction

        return self.df.copy()

    # ==========================================
    # VALIDACAO DAS OPORTUNIDADES
    # ==========================================

    def _validate_opportunity(self, opportunity):

        if not isinstance(opportunity, dict):
            raise TypeError(
                "Cada oportunidade deve ser um dicionario."
            )

        if "type" not in opportunity:
            raise ValueError(
                "Oportunidade sem o campo obrigatorio 'type'."
            )

        opportunity_type = opportunity["type"]

        if not isinstance(opportunity_type, str):
            raise ValueError(
                "O tipo da oportunidade deve ser uma string."
            )

        if opportunity_type in (
            "missing_values",
            "high_skewness"
        ):

            column = opportunity.get("column")

            if not isinstance(column, str) or not column:
                raise ValueError(
                    "Oportunidade sem um nome de coluna valido."
                )

        elif opportunity_type == "interaction_product":

            columns = opportunity.get("columns")

            if (
                not isinstance(columns, (list, tuple))
                or len(columns) != 2
                or not all(
                    isinstance(column, str) and column
                    for column in columns
                )
            ):
                raise ValueError(
                    "Uma interacao deve possuir exatamente "
                    "duas colunas validas."
                )

        return opportunity_type

    # ==========================================
    # GERACAO AUTOMATICA DE FEATURES
    # ==========================================

    def generate_candidates(self, opportunities):

        if not isinstance(opportunities, (list, tuple)):
            raise TypeError(
                "As oportunidades devem ser fornecidas "
                "em uma lista ou tupla."
            )

        generated_features = []
        generation_log = []

        for opportunity in opportunities:

            opportunity_type = self._validate_opportunity(
                opportunity
            )

            reason = opportunity.get(
                "reason",
                "Oportunidade identificada pelo detector."
            )

            # ----------------------------------
            # VALORES AUSENTES
            # ----------------------------------

            if opportunity_type == "missing_values":

                column = opportunity["column"]

                if column not in self.df.columns:
                    continue

                new_column = f"{column}_is_missing"

                if new_column in self.df.columns:
                    continue

                try:

                    self.generate_missing_indicator(column)

                    generated_features.append(new_column)

                    generation_log.append({
                        "column": column,
                        "feature": new_column,
                        "transformation": "missing_indicator",
                        "reason": reason
                    })

                except ValueError:
                    continue

            # ----------------------------------
            # ALTA ASSIMETRIA
            # ----------------------------------

            elif opportunity_type == "high_skewness":

                column = opportunity["column"]

                if column not in self.df.columns:
                    continue

                new_column = f"{column}_log1p"

                if new_column in self.df.columns:
                    continue

                try:

                    self.generate_log_transform(column)

                    generated_features.append(new_column)

                    generation_log.append({
                        "column": column,
                        "feature": new_column,
                        "transformation": "log1p",
                        "reason": reason
                    })

                except ValueError:
                    continue

            # ----------------------------------
            # INTERACAO ENTRE VARIAVEIS
            # ----------------------------------

            elif opportunity_type == "interaction_product":

                columns = opportunity["columns"]

                column_a = columns[0]
                column_b = columns[1]

                if (
                    column_a not in self.df.columns
                    or column_b not in self.df.columns
                ):
                    continue

                new_column = f"{column_a}_x_{column_b}"

                if new_column in self.df.columns:
                    continue

                try:

                    self.generate_interaction(
                        column_a,
                        column_b
                    )

                    generated_features.append(new_column)

                    generation_log.append({
                        "column": columns,
                        "feature": new_column,
                        "transformation": "interaction_product",
                        "reason": reason
                    })

                except ValueError:
                    continue

        return self.df.copy(), generation_log
