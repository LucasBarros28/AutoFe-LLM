import pandas as pd

from src.analysis.opportunity_detector import OpportunityDetector


df = pd.read_csv("data/raw/Titanic-Dataset.csv")

detector = OpportunityDetector(
    df,
    target="Survived",
    column_roles={
        "Pclass": "categorical_ordinal"
    }
)

print("\n===== CLASSIFICAÇÃO DAS COLUNAS =====")

classification = detector.classify_columns()

for column, information in classification.items():
    print(f"{column}: {information}")


print("\n===== OPORTUNIDADES DETECTADAS =====")

opportunities = detector.detect()

for opportunity in opportunities:
    print(opportunity)