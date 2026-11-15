
from src.data_loader import load_books_dataset

DATA_PATH = "data/raw/books.csv"

df = load_books_dataset(DATA_PATH)

columns = [
    "num_pages",
    "ratings_count",
    "text_reviews_count"
]

print("\n===== ANALISE DAS DISTRIBUICOES =====")

for column in columns:

    series = df[column]

    print(f"\nColuna: {column}")
    print("Minimo:", series.min())
    print("Maximo:", series.max())
    print("Media:", round(series.mean(), 4))
    print("Mediana:", round(series.median(), 4))
    print("Assimetria:", round(series.skew(), 4))
    print("Valores ausentes:", series.isna().sum())
