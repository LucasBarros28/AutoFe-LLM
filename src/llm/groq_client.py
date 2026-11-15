
import os
import json
import time

from dotenv import load_dotenv
from groq import Groq


class GroqClient:

    ALLOWED_OPERATIONS = {
        "missing_indicator",
        "log1p",
        "interaction_product"
    }

    MAX_SUGGESTIONS = 5

    MAX_ATTEMPTS = 3

    RETRY_DELAYS = [5, 15]

    QUOTA_ERRORS = {
        "429",
        "RESOURCE_EXHAUSTED",
        "QUOTA_EXCEEDED",
        "TOO_MANY_REQUESTS",
        "RATE_LIMIT"
    }

    RETRYABLE_ERRORS = {
        "500",
        "502",
        "503",
        "504",
        "INTERNAL_SERVER_ERROR",
        "SERVICE_UNAVAILABLE",
        "APICONNECTIONERROR",
        "APITIMEOUTERROR",
        "INTERNALSERVERERROR"
    }

    # =================================================
    # INICIALIZACAO
    # =================================================

    def __init__(self, model=None):

        load_dotenv()

        self.api_key = os.getenv("GROQ_API_KEY")

        self.model = (
            model
            or os.getenv(
                "GROQ_MODEL",
                "openai/gpt-oss-120b"
            )
        )

        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY não encontrada. "
                "Verifique o arquivo .env."
            )

        self.client = Groq(
            api_key=self.api_key,
            timeout=120.0,
            max_retries=0
        )

        self.last_rejected_suggestions_ = []

    # =================================================
    # IDENTIFICACAO DE ERROS
    # =================================================

    def _is_quota_error(self, error):

        error_text = str(error).upper()

        status_code = getattr(
            error, "status_code", None
        )

        if status_code == 429:
            return True

        return any(
            code in error_text
            for code in self.QUOTA_ERRORS
        )

    def _is_retryable_error(self, error):

        if self._is_quota_error(error):
            return False

        error_text = str(error).upper()

        error_type = type(error).__name__.upper()

        if error_type in self.RETRYABLE_ERRORS:
            return True

        status_code = getattr(
            error, "status_code", None
        )

        if status_code in (500, 502, 503, 504):
            return True

        return any(
            code in error_text
            for code in self.RETRYABLE_ERRORS
        )

    # =================================================
    # VALIDACAO DO CONTEXTO
    # =================================================

    def _validate_context(self, context):

        if not isinstance(context, dict):
            raise TypeError(
                "O contexto do dataset deve ser um dicionário."
            )

        columns = context.get("columns")

        if (
            not isinstance(columns, list)
            or not columns
            or not all(
                isinstance(col, str) for col in columns
            )
        ):
            raise ValueError(
                "O contexto deve possuir uma lista não vazia "
                "de nomes de colunas em 'columns'."
            )

        if len(columns) != len(set(columns)):
            raise ValueError(
                "Existem nomes de colunas duplicados."
            )

        numerical_columns = context.get(
            "numerical_columns", []
        )

        if not isinstance(numerical_columns, list):
            raise ValueError(
                "numerical_columns deve ser uma lista."
            )

        if not all(
            isinstance(col, str)
            for col in numerical_columns
        ):
            raise ValueError(
                "Os nomes das colunas numéricas devem ser strings."
            )

        if not set(numerical_columns).issubset(set(columns)):
            raise ValueError(
                "Existem colunas numéricas que não "
                "estão presentes em columns."
            )

        return columns, numerical_columns

    # =================================================
    # REDUCAO DO CONTEXTO
    # =================================================

    def _build_compact_context(self, context):

        columns, numerical_columns = (
            self._validate_context(context)
        )

        original_information = context.get(
            "column_information", {}
        )

        compact_information = {}

        for column in columns:

            information = original_information.get(
                column, {}
            )

            column_data = {}

            if "missing_values" in information:
                column_data["missing_values"] = (
                    information["missing_values"]
                )

            if column in numerical_columns:

                if "min" in information:
                    column_data["min"] = information["min"]

                if "skewness" in information:
                    column_data["skewness"] = (
                        information["skewness"]
                    )

            if column_data:
                compact_information[column] = column_data

        return {
            "problem_type": context.get(
                "problem_type", "unknown"
            ),
            "number_of_rows": context.get(
                "number_of_rows"
            ),
            "number_of_columns": len(columns),
            "columns": columns,
            "numerical_columns": numerical_columns,
            "column_information": compact_information
        }

    # =================================================
    # CONSTRUCAO DO PROMPT
    # =================================================

    def _build_prompt(self, context):

        compact_context = self._build_compact_context(
            context
        )

        context_json = json.dumps(
            compact_context,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str
        )

        prompt = f"""
Você é um especialista em Feature Engineering
para dados tabulares.

Analise o contexto do dataset e recomende transformações
que possam melhorar a representação dos atributos.

CONTEXTO:

{context_json}

OPERAÇÕES PERMITIDAS:

1. missing_indicator
Cria um indicador para uma coluna que possui valores ausentes.

2. log1p
Aplica log(1 + x) em uma coluna numérica.
Utilize somente quando os valores forem não negativos
e houver assimetria que justifique a transformação.

3. interaction_product
Multiplica duas colunas numéricas diferentes.

REGRAS:

- Utilize somente as operações permitidas.
- Utilize somente nomes de colunas existentes.
- Para log1p, escolha somente colunas numéricas.
- Para log1p, o valor mínimo deve ser não negativo.
- Para missing_indicator, a coluna deve possuir valores ausentes.
- Para interaction_product, escolha duas colunas numéricas diferentes.
- Não invente colunas.
- Não gere código Python.
- Cada sugestão deve possuir uma justificativa.
- Recomende no máximo {self.MAX_SUGGESTIONS} transformações.
- Se não houver boas oportunidades, retorne uma lista vazia.
- Responda somente com JSON válido.
- Não inclua Markdown ou blocos de código na resposta.

FORMATO:

{{
    "suggestions": [
        {{
            "operation": "log1p",
            "column": "nome_da_coluna",
            "reason": "Justificativa."
        }}
    ]
}}

Para interaction_product, utilize o campo "columns"
com exatamente duas colunas.
"""

        return prompt

    # =================================================
    # VALIDACAO DAS SUGESTOES
    # =================================================

    def validate_response(self, response_text, context):

        columns, numerical_columns = (
            self._validate_context(context)
        )

        column_information = context.get(
            "column_information", {}
        )

        try:

            response_data = json.loads(response_text)

        except (json.JSONDecodeError, TypeError):

            raise ValueError(
                "A Groq retornou uma resposta "
                "que não é um JSON válido."
            )

        if not isinstance(response_data, dict):
            raise ValueError(
                "A resposta da Groq deve ser um objeto JSON."
            )

        suggestions = response_data.get("suggestions")

        if not isinstance(suggestions, list):
            raise ValueError(
                "A resposta deve conter uma lista "
                "chamada 'suggestions'."
            )

        valid_suggestions = []
        rejected_suggestions = []

        for index, suggestion in enumerate(suggestions):

            if index >= self.MAX_SUGGESTIONS:

                rejected_suggestions.append({
                    "suggestion": suggestion,
                    "reason": "Limite de sugestões excedido."
                })

                continue

            if not isinstance(suggestion, dict):

                rejected_suggestions.append({
                    "suggestion": suggestion,
                    "reason": "Sugestão não é um objeto."
                })

                continue

            operation = suggestion.get("operation")

            if operation not in self.ALLOWED_OPERATIONS:

                rejected_suggestions.append({
                    "suggestion": suggestion,
                    "reason": "Operação não permitida."
                })

                continue

            reason = suggestion.get("reason")

            if (
                not isinstance(reason, str)
                or not reason.strip()
            ):

                rejected_suggestions.append({
                    "suggestion": suggestion,
                    "reason": "Justificativa inválida."
                })

                continue

            # MISSING INDICATOR

            if operation == "missing_indicator":

                column = suggestion.get("column")

                if column not in columns:

                    rejected_suggestions.append({
                        "suggestion": suggestion,
                        "reason": "Coluna inexistente."
                    })

                    continue

                information = column_information.get(
                    column, {}
                )

                missing_values = information.get(
                    "missing_values"
                )

                if (
                    missing_values is not None
                    and missing_values <= 0
                ):

                    rejected_suggestions.append({
                        "suggestion": suggestion,
                        "reason": (
                            "A coluna não possui valores ausentes."
                        )
                    })

                    continue

                valid_suggestions.append({
                    "operation": operation,
                    "column": column,
                    "reason": reason
                })

            # LOG1P

            elif operation == "log1p":

                column = suggestion.get("column")

                if column not in numerical_columns:

                    rejected_suggestions.append({
                        "suggestion": suggestion,
                        "reason": (
                            "A coluna não existe ou "
                            "não é numérica."
                        )
                    })

                    continue

                information = column_information.get(
                    column, {}
                )

                minimum = information.get("min")

                if minimum is not None and minimum < 0:

                    rejected_suggestions.append({
                        "suggestion": suggestion,
                        "reason": (
                            "A coluna possui valores negativos "
                            "e não pode receber log1p."
                        )
                    })

                    continue

                valid_suggestions.append({
                    "operation": operation,
                    "column": column,
                    "reason": reason
                })

            # INTERACTION PRODUCT

            elif operation == "interaction_product":

                interaction_columns = suggestion.get(
                    "columns"
                )

                if (
                    not isinstance(interaction_columns, list)
                    or len(interaction_columns) != 2
                ):

                    rejected_suggestions.append({
                        "suggestion": suggestion,
                        "reason": (
                            "A interação deve possuir "
                            "exatamente duas colunas."
                        )
                    })

                    continue

                column_a, column_b = interaction_columns

                if (
                    column_a not in numerical_columns
                    or column_b not in numerical_columns
                ):

                    rejected_suggestions.append({
                        "suggestion": suggestion,
                        "reason": (
                            "As duas colunas devem existir "
                            "e ser numéricas."
                        )
                    })

                    continue

                if column_a == column_b:

                    rejected_suggestions.append({
                        "suggestion": suggestion,
                        "reason": (
                            "A interação exige duas "
                            "colunas diferentes."
                        )
                    })

                    continue

                valid_suggestions.append({
                    "operation": operation,
                    "columns": [
                        column_a,
                        column_b
                    ],
                    "reason": reason
                })

        self.last_rejected_suggestions_ = (
            rejected_suggestions
        )

        return valid_suggestions

    # =================================================
    # SOLICITACAO COM NOVAS TENTATIVAS
    # =================================================

    def _request_with_retry(self, prompt):

        for attempt in range(self.MAX_ATTEMPTS):

            try:

                print(
                    f"[Groq] Tentativa "
                    f"{attempt + 1}/{self.MAX_ATTEMPTS}...",
                    flush=True
                )

                response = self.client.chat.completions.create(
                    model=self.model,

                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Você é um especialista em Feature "
                                "Engineering. Responda somente com "
                                "JSON válido no formato solicitado."
                            )
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],

                    temperature=0.2,
                    max_completion_tokens=2000,

                    response_format={
                        "type": "json_object"
                    }
                )

                return response

            except Exception as error:

                if self._is_quota_error(error):

                    print(
                        "[Groq] Limite de utilização identificado. "
                        "Não serão realizadas novas tentativas.",
                        flush=True
                    )

                    raise

                retryable = self._is_retryable_error(error)

                is_last_attempt = (
                    attempt == self.MAX_ATTEMPTS - 1
                )

                if not retryable or is_last_attempt:
                    raise

                delay = self.RETRY_DELAYS[
                    min(
                        attempt,
                        len(self.RETRY_DELAYS) - 1
                    )
                ]

                print(
                    f"[Groq] Erro temporário: {error}",
                    flush=True
                )

                print(
                    f"[Groq] Aguardando {delay}s "
                    f"antes da próxima tentativa...",
                    flush=True
                )

                time.sleep(delay)

        raise RuntimeError(
            "Não foi possível obter resposta da Groq."
        )

    # =================================================
    # SOLICITACAO A GROQ
    # =================================================

    def suggest_features(self, context):

        self._validate_context(context)

        print(
            "[Groq] Construindo prompt...",
            flush=True
        )

        prompt = self._build_prompt(context)

        print(
            f"[Groq] Prompt construído "
            f"({len(prompt)} caracteres).",
            flush=True
        )

        start = time.perf_counter()

        try:

            print(
                "[Groq] Iniciando consulta...",
                flush=True
            )

            response = self._request_with_retry(prompt)

            elapsed = time.perf_counter() - start

            print(
                f"[Groq] Resposta recebida em "
                f"{elapsed:.2f} segundos.",
                flush=True
            )

        except Exception as error:

            print(
                f"[Groq] Erro durante a requisição: "
                f"{type(error).__name__}: {error}",
                flush=True
            )

            raise RuntimeError(
                "Erro ao consultar a API da Groq: "
                f"{error}"
            ) from error

        print(
            "[Groq] Extraindo resposta...",
            flush=True
        )

        if not response.choices:
            raise ValueError(
                "A Groq não retornou nenhuma escolha."
            )

        response_text = (
            response.choices[0].message.content
        )

        if not response_text:

            raise ValueError(
                "A Groq retornou uma resposta vazia."
            )

        print(
            "[Groq] Validando sugestões...",
            flush=True
        )

        try:

            valid_suggestions = self.validate_response(
                response_text,
                context
            )

        except ValueError:

            print(
                "[Groq] Resposta recebida "
                "(primeiros 500 caracteres):",
                repr(response_text[:500]),
                flush=True
            )

            raise

        print(
            f"[Groq] {len(valid_suggestions)} sugestões válidas.",
            flush=True
        )

        print(
            f"[Groq] {len(self.last_rejected_suggestions_)} "
            "sugestões rejeitadas.",
            flush=True
        )

        return valid_suggestions
