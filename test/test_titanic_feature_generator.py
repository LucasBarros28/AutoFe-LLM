import pandas as pd

from src.analysis.titanic_feature_generator import (
    TitanicFeatureGenerator
)


df = pd.read_csv("data/raw/Titanic-Dataset.csv")

generator = TitanicFeatureGenerator(df)

df = generator.generate_family_size()
df = generator.generate_is_alone()
df = generator.generate_has_cabin()

print("\n===== FEATURES ESPECÍFICAS DO TITANIC =====")

print(
    df[
        [
            "FamilySize",
            "IsAlone",
            "HasCabin"
        ]
    ].head()
)