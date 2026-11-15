
import pandas as pd

from src.analysis.opportunity_detector import OpportunityDetector


# Dataset sintético para classificação
df = pd.DataFrame({
    "idade": [20, 25, 30, 35, 40, 45, 50, 55, 60, 65],
    "renda": [1000, 1200, 1500, 1800, 2200,
              2500, 3000, 3500, 4000, 4500],
    "horas_estudo": [1, 2, 2, 3, 4, 5, 6, 7, 8, 9],
    "compras": [1, 1, 2, 2, 3, 4, 5, 6, 7, 8]
})

y = pd.Series([
    0, 0, 0, 0, 0, 1, 1, 1, 1, 1
])

detector = OpportunityDetector(df)

opportunities = detector.detect_interactions(
    y=y,
    problem_type="classification",
    max_interactions=3
)

print("\n===== INTERAÇÕES RANQUEADAS =====")

for opportunity in opportunities:
    print(
        opportunity["feature"],
        "-> Score:",
        round(opportunity["score"], 5)
    )
