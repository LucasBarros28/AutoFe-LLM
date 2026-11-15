
from src.data_loader import (
    load_books_dataset,
    prepare_rating_experiments
)


DATA_PATH = "data/raw/books.csv"

# Carrega a base.
df = load_books_dataset(DATA_PATH)

# Prepara os dois experimentos.
df_all, df_without_zero = prepare_rating_experiments(df)

print("\n===== BASE ORIGINAL =====")
print("Registros:", len(df))
print("Colunas:", len(df.columns))
print("Avaliações iguais a zero:", (df["average_rating"] == 0).sum())
print("Avaliações ausentes:", df["average_rating"].isna().sum())

print("\n===== EXPERIMENTO 1: TODOS OS REGISTROS =====")
print("Registros:", len(df_all))

print("\n===== EXPERIMENTO 2: SEM AVALIAÇÕES ZERO =====")
print("Registros:", len(df_without_zero))

print("\n===== COLUNAS =====")
print(df.columns.tolist())

print("\n===== TIPOS DE DADOS =====")
print(df.dtypes)

print("\n===== ESTATÍSTICAS DA AVALIAÇÃO =====")
print(df["average_rating"].describe())
print("\nSem avaliações zero:")
print(df_without_zero["average_rating"].describe())

# Verificações básicas.
assert len(df) == 11127, "A quantidade de registros está incorreta."
assert len(df_all) == len(df)
assert len(df_without_zero) + (df["average_rating"] == 0).sum() == len(df)
