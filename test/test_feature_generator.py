import pandas as pd

from src.analysis.opportunity_detector import OpportunityDetector
from src.analysis.feature_generator import FeatureGenerator


df = pd.read_csv("data/raw/Titanic-Dataset.csv")

detector = OpportunityDetector(
    df,
    target="Survived",
    column_roles={
        "Pclass": "categorical_ordinal"
    }
)

opportunities = detector.detect()

generator = FeatureGenerator(df)

df_generated, generation_log = generator.generate_candidates(
    opportunities
)

print("\n===== FEATURES GERADAS =====")

for feature in generation_log:
    print(feature)

print("\n===== NOVAS COLUNAS =====")

new_columns = [
    column for column in df_generated.columns
    if column not in df.columns
]

print(new_columns)