
import os
import json
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types


class GeminiClient:

    ALLOWED_OPERATIONS = {
        "missing_indicator",
        "log1p",
        "interaction_product"
    }

    MAX_SUGGESTIONS = 5

    MAX_ATTEMPTS = 3

    RETRY_DELAYS = [5, 15]

    # Erros que podem justificar uma nova tentativa.
    # Erros de cota, como 429 e RESOURCE_EXHAUSTED,
    # foram removidos intencionalmente.
    RETRYABLE_ERRORS = {
        "500",
        "502",
        "503",
        "504",
        "DEADLINE_EXCEEDED",
        "UNAVAILABLE",
        "INTERNAL",
        "GATEWAY"
    }

    QUOTA_ERRORS = {
        "429",
        "RESOURCE_EXHAUSTED",
        "QUOTA_EXCEEDED",
        "TOO_MANY_REQUESTS"
    }

    def __init__(self, model=None):

        load_dotenv()

        self.api_key = os.getenv("GEMINI_API_KEY")

        self.model = (
            model
            or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        )

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY não encontrada. "
                "Verifique o arquivo .env."
            )

        self.client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(
                timeout=600000
            )
        )

        self.last_rejected_suggestions_ = []

    # =================================================
    # IDENTIFICACAO DE ERROS
    # =================================================

    def _is_quota_error(self, error):

        error_text = str(error).upper()

        return any(
            code in error_text
            for code in self.QUOTA_ERRORS
        )

    def _is_retryable_error(self, error):

        # Erros de cota nunca devem gerar novas tentativas.
        if self._is_quota_error(error):
            return False

        error_text = str(error).upper()

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
            or not all(isinstance(col, str) for col in columns)
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
            isinstance(col, str) for col in numerical_columns
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

        compact_context = {
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

        return compact_context

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
                "O Gemini retornou uma resposta "
                "que não é um JSON válido."
            )

        if not isinstance(response_data, dict):
            raise ValueError(
                "A resposta do Gemini deve ser um objeto JSON."
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
    # CONSULTA DE UMA INTERACAO
    # =================================================

    def _wait_for_interaction(self, interaction):

        max_wait = 600
        start_wait = time.perf_counter()
        last_status_message = 0

        get_attempts = 0
        max_get_attempts = 3

        while True:

            elapsed_wait = (
                time.perf_counter() - start_wait
            )

            if elapsed_wait >= max_wait:

                raise TimeoutError(
                    "Tempo máximo de espera pelo Gemini excedido."
                )

            time.sleep(5)

            try:

                response = self.client.interactions.get(
                    id=interaction.id
                )

                get_attempts = 0

            except Exception as error:

                # RESOURCE_EXHAUSTED e 429 não são repetidos.
                if self._is_quota_error(error):
                    raise

                if (
                    self._is_retryable_error(error)
                    and get_attempts < max_get_attempts - 1
                ):

                    get_attempts += 1

                    delay = self.RETRY_DELAYS[
                        min(
                            get_attempts - 1,
                            len(self.RETRY_DELAYS) - 1
                        )
                    ]

                    print(
                        f"[Gemini] Falha temporária ao consultar "
                        f"a interação. Nova consulta em {delay}s "
                        f"({get_attempts + 1}/{max_get_attempts}).",
                        flush=True
                    )

                    time.sleep(delay)

                    continue

                raise

            status = response.status

            if status == "completed":
                return response

            if status in ("failed", "cancelled"):

                errors = getattr(
                    response, "errors", None
                )

                error = getattr(
                    response, "error", None
                )

                details = errors or error or "Sem detalhes."

                raise RuntimeError(
                    f"Interação encerrada com status: "
                    f"{status}. Detalhes: {details}"
                )

            elapsed_int = int(
                time.perf_counter() - start_wait
            )

            if elapsed_int - last_status_message >= 15:

                print(
                    f"[Gemini] Status: {status} "
                    f"| Tempo: {elapsed_int}s",
                    flush=True
                )

                last_status_message = elapsed_int

    # =================================================
    # SOLICITACAO COM NOVAS TENTATIVAS
    # =================================================

    def _request_with_retry(self, prompt):

        for attempt in range(self.MAX_ATTEMPTS):

            try:

                print(
                    f"[Gemini] Tentativa "
                    f"{attempt + 1}/{self.MAX_ATTEMPTS}...",
                    flush=True
                )

                interaction = self.client.interactions.create(
                    model=self.model,
                    input=prompt,

                    generation_config={
                        "temperature": 0.2
                    },

                    response_format={
                        "type": "text",
                        "mime_type": "application/json",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "suggestions": {
                                    "type": "array",
                                    "maxItems": self.MAX_SUGGESTIONS,
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "operation": {
                                                "type": "string",
                                                "enum": [
                                                    "missing_indicator",
                                                    "log1p",
                                                    "interaction_product"
                                                ]
                                            },
                                            "column": {
                                                "type": "string"
                                            },
                                            "columns": {
                                                "type": "array",
                                                "items": {
                                                    "type": "string"
                                                },
                                                "minItems": 2,
                                                "maxItems": 2
                                            },
                                            "reason": {
                                                "type": "string"
                                            }
                                        },
                                        "required": [
                                            "operation",
                                            "reason"
                                        ]
                                    }
                                }
                            },
                            "required": [
                                "suggestions"
                            ]
                        }
                    },

                    background=True
                )

                print(
                    f"[Gemini] Interação criada: {interaction.id}",
                    flush=True
                )

                return self._wait_for_interaction(
                    interaction
                )

            except Exception as error:

                # Erro de cota: interromper imediatamente.
                if self._is_quota_error(error):

                    print(
                        "[Gemini] Erro de cota identificado. "
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
                    min(attempt, len(self.RETRY_DELAYS) - 1)
                ]

                print(
                    f"[Gemini] Erro temporário: {error}",
                    flush=True
                )

                print(
                    f"[Gemini] Aguardando {delay}s "
                    f"antes da próxima tentativa...",
                    flush=True
                )

                time.sleep(delay)

        raise RuntimeError(
            "Não foi possível obter resposta do Gemini."
        )

    # =================================================
    # SOLICITACAO AO GEMINI
    # =================================================

    def suggest_features(self, context):

        self._validate_context(context)

        print(
            "[Gemini] Construindo prompt...",
            flush=True
        )

        prompt = self._build_prompt(context)

        print(
            f"[Gemini] Prompt construído "
            f"({len(prompt)} caracteres).",
            flush=True
        )

        start = time.perf_counter()

        try:

            print(
                "[Gemini] Iniciando consulta...",
                flush=True
            )

            response = self._request_with_retry(prompt)

            elapsed = time.perf_counter() - start

            print(
                f"[Gemini] Resposta recebida em "
                f"{elapsed:.2f} segundos.",
                flush=True
            )

        except Exception as error:

            print(
                f"[Gemini] Erro durante a requisição: "
                f"{type(error).__name__}: {error}",
                flush=True
            )

            raise RuntimeError(
                "Erro ao consultar a API do Gemini: "
                f"{error}"
            ) from error

        print(
            "[Gemini] Extraindo resposta...",
            flush=True
        )

        response_text = response.output_text

        if not response_text:

            raise ValueError(
                "O Gemini retornou uma resposta vazia."
            )

        print(
            "[Gemini] Validando sugestões...",
            flush=True
        )

        try:

            valid_suggestions = self.validate_response(
                response_text,
                context
            )

        except ValueError:

            print(
                "[Gemini] Resposta recebida (primeiros 500 caracteres):",
                repr(response_text[:500]),
                flush=True
            )

            raise

        print(
            f"[Gemini] {len(valid_suggestions)} sugestões válidas.",
            flush=True
        )

        print(
            f"[Gemini] {len(self.last_rejected_suggestions_)} "
            "sugestões rejeitadas.",
            flush=True
        )

        return valid_suggestions
