
from src.data_loader import load_books_dataset
from src.analysis.opportunity_detector import OpportunityDetector
from src.analysis.feature_generator import FeatureGenerator


DATA_PATH = "data/raw/books.csv"
TARGET_COLUMN = "average_rating"

# Carrega a base.
df = load_books_dataset(DATA_PATH)

# Separa os atributos da variável alvo.
X = df.drop(columns=[TARGET_COLUMN])

# Remove identificadores e colunas que não utilizaremos.
DROP_COLUMNS = [
    "bookID",
    "title",
    "authors",
    "isbn",
    "isbn13",
    "publication_date",
    "publisher"
]

X = X.drop(columns=DROP_COLUMNS, errors="ignore")


# ==========================================
# CLASSIFICACAO DAS COLUNAS
# ==========================================

detector = OpportunityDetector(X)

column_roles = detector.classify_columns()

print("\n===== CLASSIFICACAO DAS COLUNAS =====")

for column, information in column_roles.items():
    print(column, "->", information)


# ==========================================
# DETECCAO DE OPORTUNIDADES
# ==========================================

opportunities = detector.detect()

print("\n===== OPORTUNIDADES DETECTADAS =====")

for opportunity in opportunities:
    print(opportunity)

print("\nTotal de oportunidades:", len(opportunities))


# ==========================================
# GERACAO DE FEATURES
# ==========================================

generator = FeatureGenerator(X)

generated_df, generation_log = generator.generate_candidates(
    opportunities
)

print("\n===== FEATURES GERADAS =====")

for feature in generation_log:
    print(feature)

print("\nTotal de features geradas:", len(generation_log))
