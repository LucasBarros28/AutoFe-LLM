import pandas as pd


class TitanicFeatureGenerator:

    def __init__(self, df):

        if not isinstance(df, pd.DataFrame):
            raise TypeError("A entrada deve ser um DataFrame.")

        if df.empty:
            raise ValueError("O DataFrame está vazio.")

        self.df = df.copy()

    # Criar tamanho da família
    def generate_family_size(self):

        required_columns = ["SibSp", "Parch"]

        for column in required_columns:
            if column not in self.df.columns:
                raise ValueError(
                    f"Coluna obrigatória não encontrada: {column}"
                )

        self.df["FamilySize"] = (
            self.df["SibSp"] + self.df["Parch"] + 1
        )

        return self.df.copy()

    # Identificar passageiros que viajaram sozinhos
    def generate_is_alone(self):

        if "FamilySize" not in self.df.columns:
            self.generate_family_size()

        self.df["IsAlone"] = (
            self.df["FamilySize"] == 1
        ).astype(int)

        return self.df.copy()

    # Identificar passageiros com cabine registrada
    def generate_has_cabin(self):

        if "Cabin" not in self.df.columns:
            raise ValueError("Coluna Cabin não encontrada.")

        self.df["HasCabin"] = (
            self.df["Cabin"].notna()
        ).astype(int)

        return self.df.copy()