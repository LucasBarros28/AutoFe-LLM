
import csv
import pandas as pd


def load_books_dataset(file_path):
    """
    Carrega o Goodreads Books preservando os registros
    que possuem vírgulas não protegidas por aspas.
    """

    rows = []

    # Registros conhecidos com vírgulas extras no campo authors.
    repaired_book_ids = {
        "12224",
        "16914",
        "22128",
        "34889"
    }

    with open(file_path, "r", encoding="utf-8-sig", newline="") as file:

        reader = csv.reader(file)

        columns = next(reader)

        for line_number, fields in enumerate(reader, start=2):

            # Corrige os quatro registros identificados.
            if len(fields) == 13 and fields[0] in repaired_book_ids:

                fields = (
                    fields[:2]
                    + [", ".join(fields[2:4])]
                    + fields[4:]
                )

            # Não permite descartar registros silenciosamente.
            if len(fields) != 12:
                raise ValueError(
                    f"Erro na linha {line_number}: "
                    f"esperadas 12 colunas, encontradas {len(fields)}."
                )

            rows.append(fields)

    df = pd.DataFrame(rows, columns=columns)

    # Remove espaços extras dos nomes das colunas.
    df.columns = df.columns.str.strip()

    # Converte colunas numéricas.
    numeric_columns = [
        "bookID",
        "average_rating",
        "num_pages",
        "ratings_count",
        "text_reviews_count"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Converte a data de publicação.
    df["publication_date"] = pd.to_datetime(
        df["publication_date"],
        format="%m/%d/%Y",
        errors="coerce"
    )

    return df


def prepare_rating_experiments(df, target="average_rating"):

    if target not in df.columns:
        raise ValueError(f"Coluna alvo '{target}' não encontrada.")

    # Experimento 1: todos os registros.
    df_all = df.copy()

    # Experimento 2: remove avaliações iguais a zero.
    df_without_zero = df[df[target] != 0].copy()

    return df_all, df_without_zero
