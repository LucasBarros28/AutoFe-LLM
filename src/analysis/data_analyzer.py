import pandas as pd
import numpy as np


class DataAnalyzer:

    def __init__(self, df):

        if not isinstance(df, pd.DataFrame):
            raise TypeError("A entrada deve ser um DataFrame.")

        if df.empty:
            raise ValueError("O DataFrame está vazio.")

        self.df = df.copy()

    # Estrutura do dataset
    def get_structure(self):

        structure = {
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "column_types": {},
            "cardinality": {}
        }

        for column in self.df.columns:

            structure["column_types"][column] = (
                str(self.df[column].dtype)
            )

            structure["cardinality"][column] = (
                self.df[column].nunique()
            )

        return structure

    # Estatísticas
    def get_statistics(self):

        numeric_df = self.df.select_dtypes(
            include=np.number
        )

        statistics = {}

        for column in numeric_df.columns:

            statistics[column] = {
                "mean": numeric_df[column].mean(),
                "median": numeric_df[column].median(),
                "std": numeric_df[column].std(),
                "min": numeric_df[column].min(),
                "max": numeric_df[column].max(),
                "q25": numeric_df[column].quantile(0.25),
                "q75": numeric_df[column].quantile(0.75)
            }

        return statistics

    # Valores ausentes
    def get_missing_values(self):

        missing = {}

        for column in self.df.columns:

            quantity = self.df[column].isna().sum()

            percentage = (
                quantity / len(self.df)
            ) * 100

            missing[column] = {
                "quantity": int(quantity),
                "percentage": percentage
            }

        return missing

    # Relatório geral
    def analyze(self):

        return {
            "structure": self.get_structure(),
            "statistics": self.get_statistics(),
            "missing_values": self.get_missing_values()
        }