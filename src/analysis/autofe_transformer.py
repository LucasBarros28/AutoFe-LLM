
import os
import json
import hashlib
import time

import pandas as pd

from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin

from src.analysis.opportunity_detector import OpportunityDetector
from src.analysis.feature_generator import FeatureGenerator


# =====================================================
# CIRCUIT BREAKER POR PROVEDOR
# =====================================================

_LLM_CIRCUITS = {
    "gemini": {
        "open": False,
        "reason": None
    },
    "groq": {
        "open": False,
        "reason": None
    }
}


def reset_llm_circuit(provider=None):
    """
    Reinicia o circuit breaker.

    Se provider=None, reinicia todos os provedores.
    Caso contrário, reinicia apenas Gemini ou Groq.
    """

    if provider is None:

        for name in _LLM_CIRCUITS:
            _LLM_CIRCUITS[name]["open"] = False
            _LLM_CIRCUITS[name]["reason"] = None

        print(
            "[AutoFE] Circuit breakers reiniciados.",
            flush=True
        )

    else:

        provider = provider.lower()

        if provider not in _LLM_CIRCUITS:
            raise ValueError(
                "Provedor inválido. Utilize 'gemini' ou 'groq'."
            )

        _LLM_CIRCUITS[provider]["open"] = False
        _LLM_CIRCUITS[provider]["reason"] = None

        print(
            f"[AutoFE] Circuit breaker de {provider} reiniciado.",
            flush=True
        )


def _is_quota_error(error):

    error_text = str(error).upper()

    quota_codes = (
        "429",
        "RESOURCE_EXHAUSTED",
        "QUOTA_EXCEEDED",
        "TOO_MANY_REQUESTS"
    )

    return any(code in error_text for code in quota_codes)


def _open_llm_circuit(provider, reason):

    circuit = _LLM_CIRCUITS[provider]

    circuit["open"] = True
    circuit["reason"] = str(reason)

    print(
        f"[AutoFE] CIRCUIT BREAKER DE {provider.upper()} ATIVADO.",
        flush=True
    )

    print(
        f"[AutoFE] Motivo: {reason}",
        flush=True
    )


# =====================================================
# AUTOFETRANSFORMER
# =====================================================

class AutoFETransformer(BaseEstimator, TransformerMixin):

    CACHE_VERSION = 2

    def __init__(
        self,
        max_interactions=10,
        include_interactions=True,
        problem_type="classification",
        random_state=42,
        use_llm=False,
        llm_model=None,
        llm_provider="groq"
    ):

        self.max_interactions = max_interactions
        self.include_interactions = include_interactions
        self.problem_type = problem_type
        self.random_state = random_state
        self.use_llm = use_llm
        self.llm_model = llm_model
        self.llm_provider = llm_provider

    # =================================================
    # CONFIGURACAO DO CACHE
    # =================================================

    def _get_cache_path(self):

        project_root = Path(__file__).resolve().parents[2]

        return (
            project_root
            / ".cache"
            / "autofe_llm_cache.json"
        )

    def _build_cache_key(self, context, model, provider):

        cache_data = {
            "version": self.CACHE_VERSION,
            "provider": provider,
            "model": model,
            "context": context
        }

        serialized = json.dumps(
            cache_data,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str
        )

        return hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()

    def _load_cache(self):

        cache_path = self._get_cache_path()

        if not cache_path.exists():
            return {}

        try:

            with open(cache_path, "r", encoding="utf-8") as file:
                cache = json.load(file)

            if not isinstance(cache, dict):
                return {}

            return cache

        except (OSError, json.JSONDecodeError) as error:

            print(
                f"[AutoFE] Não foi possível ler o cache: {error}",
                flush=True
            )

            return {}

    def _save_cache(self, cache):

        cache_path = self._get_cache_path()

        try:

            cache_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            temporary_path = cache_path.with_suffix(".tmp")

            with open(
                temporary_path,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    cache,
                    file,
                    ensure_ascii=False,
                    indent=2
                )

            os.replace(temporary_path, cache_path)

            print("[AutoFE] Cache salvo.", flush=True)

        except OSError as error:

            print(
                f"[AutoFE] Não foi possível salvar o cache: {error}",
                flush=True
            )

    # =================================================
    # CONSTRUCAO DO CONTEXTO PARA A LLM
    # =================================================

    def _build_llm_context(self, X):

        numerical_columns = X.select_dtypes(
            include="number"
        ).columns.tolist()

        numerical_columns = [
            column
            for column in numerical_columns
            if not pd.api.types.is_bool_dtype(X[column])
        ]

        column_information = {}

        for column in X.columns:

            missing_values = int(X[column].isna().sum())

            information = {
                "missing_values": missing_values,
                "missing_percentage": round(
                    X[column].isna().mean() * 100,
                    2
                ),
                "dtype": str(X[column].dtype)
            }

            if column in numerical_columns:

                values = X[column].dropna()

                if not values.empty:

                    information.update({
                        "mean": float(values.mean()),
                        "std": float(values.std()),
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "skewness": float(values.skew())
                    })

            column_information[column] = information

        return {
            "dataset_description": (
                "Dataset tabular para Feature Engineering."
            ),
            "problem_type": self.problem_type,
            "number_of_rows": len(X),
            "number_of_columns": len(X.columns),
            "columns": X.columns.tolist(),
            "numerical_columns": numerical_columns,
            "column_information": column_information
        }

    # =================================================
    # CONVERSAO DAS SUGESTOES DA LLM
    # =================================================

    def _convert_llm_suggestions(self, suggestions):

        opportunities = []

        for suggestion in suggestions:

            operation = suggestion["operation"]
            reason = suggestion["reason"]

            if operation == "missing_indicator":

                opportunities.append({
                    "type": "missing_values",
                    "column": suggestion["column"],
                    "severity": "medium",
                    "recommendation": "Create missing indicator",
                    "reason": reason
                })

            elif operation == "log1p":

                opportunities.append({
                    "type": "high_skewness",
                    "column": suggestion["column"],
                    "severity": "experimental",
                    "recommendation": "Apply log1p transformation",
                    "reason": reason
                })

            elif operation == "interaction_product":

                columns = suggestion["columns"]

                opportunities.append({
                    "type": "interaction_product",
                    "columns": columns,
                    "feature": f"{columns[0]}_x_{columns[1]}",
                    "severity": "experimental",
                    "score": 0.0,
                    "recommendation": "Evaluate multiplication interaction",
                    "reason": reason
                })

        return opportunities

    # =================================================
    # NORMALIZACAO DE CANDIDATAS A INTERACAO
    # =================================================

    def _get_interaction_feature_name(self, item):
        """
        Converte uma candidata a interação em seu nome de feature.

        Aceita:
        - String: "age_x_bmi"
        - Dicionário com feature
        - Dicionário com columns
        - Dicionário com column
        """

        if isinstance(item, str):
            return item

        if not isinstance(item, dict):
            return None

        feature = item.get("feature")

        if feature:
            return str(feature)

        columns = item.get("columns")

        if columns is None:
            columns = item.get("column")

        if isinstance(columns, (list, tuple)) and len(columns) == 2:

            return f"{columns[0]}_x_{columns[1]}"

        return None

    def _collect_interaction_candidates(
        self,
        traditional_candidates,
        llm_opportunities
    ):
        """
        Reúne as candidatas tradicionais e as sugeridas pela LLM.

        Esta lista representa oportunidades identificadas.
        Não representa necessariamente features executadas.
        """

        candidates = []

        for item in traditional_candidates:

            feature = self._get_interaction_feature_name(item)

            if feature:
                candidates.append(feature)

        for item in llm_opportunities:

            if not isinstance(item, dict):
                continue

            if item.get("type") != "interaction_product":
                continue

            feature = self._get_interaction_feature_name(item)

            if feature:
                candidates.append(feature)

        return list(dict.fromkeys(candidates))

    # =================================================
    # REMOVER OPORTUNIDADES DUPLICADAS
    # =================================================

    def _opportunity_key(self, opportunity):

        opportunity_type = opportunity.get("type")

        if opportunity_type == "interaction_product":

            columns = opportunity.get("columns", [])

            return opportunity_type, tuple(sorted(columns))

        return opportunity_type, opportunity.get("column")

    def _merge_opportunities(
        self,
        traditional_opportunities,
        llm_opportunities
    ):

        merged = []
        existing_keys = set()

        for opportunity in (
            traditional_opportunities + llm_opportunities
        ):

            key = self._opportunity_key(opportunity)

            if key not in existing_keys:

                merged.append(opportunity)
                existing_keys.add(key)

        return merged

    # =================================================
    # INICIALIZACAO DO CLIENTE LLM
    # =================================================

    def _initialize_llm_client(self):

        provider = self.llm_provider.lower()

        if provider == "gemini":

            from src.llm.gemini_client import GeminiClient

            print(
                "[AutoFE] Inicializando GeminiClient...",
                flush=True
            )

            return GeminiClient(model=self.llm_model)

        elif provider == "groq":

            from src.llm.groq_client import GroqClient

            print(
                "[AutoFE] Inicializando GroqClient...",
                flush=True
            )

            return GroqClient(model=self.llm_model)

        raise ValueError(
            "Provedor inválido. Utilize 'gemini' ou 'groq'."
        )

    # =================================================
    # INTEGRACAO COM A LLM
    # =================================================

    def _run_llm(self, X):

        self.llm_suggestions_ = []
        self.llm_opportunities_ = []
        self.llm_rejected_suggestions_ = []

        self.llm_cache_hit_ = False
        self.llm_status_ = "disabled"
        self.llm_error_ = None
        self.llm_attempted_ = False

        self.llm_provider_ = self.llm_provider.lower()
        self.llm_client_ = None

        if not self.use_llm:
            return

        provider = self.llm_provider_

        try:

            # 1. Inicializa o provedor.

            start = time.perf_counter()

            self.llm_client_ = self._initialize_llm_client()

            print(
                f"[AutoFE] Cliente {provider.upper()} inicializado "
                f"em {time.perf_counter() - start:.2f}s.",
                flush=True
            )

            # 2. Constrói o contexto.

            print("[AutoFE] Construindo contexto...", flush=True)

            start = time.perf_counter()

            context = self._build_llm_context(X)

            print(
                f"[AutoFE] Contexto construído em "
                f"{time.perf_counter() - start:.2f}s.",
                flush=True
            )

            # 3. Verifica o cache.

            cache = self._load_cache()

            cache_key = self._build_cache_key(
                context,
                self.llm_client_.model,
                provider
            )

            cached_suggestions = cache.get(cache_key)

            if isinstance(cached_suggestions, list):

                try:

                    print(
                        f"[AutoFE] Cache encontrado para {provider.upper()}.",
                        flush=True
                    )

                    cached_json = json.dumps(
                        {"suggestions": cached_suggestions},
                        ensure_ascii=False
                    )

                    self.llm_suggestions_ = (
                        self.llm_client_.validate_response(
                            cached_json,
                            context
                        )
                    )

                    self.llm_rejected_suggestions_ = getattr(
                        self.llm_client_,
                        "last_rejected_suggestions_",
                        []
                    )

                    self.llm_cache_hit_ = True
                    self.llm_status_ = "cache"

                    print(
                        "[AutoFE] Cache utilizado com sucesso.",
                        flush=True
                    )

                except (ValueError, TypeError, KeyError) as error:

                    print(
                        f"[AutoFE] Cache inválido: {error}",
                        flush=True
                    )

                    self.llm_suggestions_ = []
                    self.llm_rejected_suggestions_ = []
                    self.llm_cache_hit_ = False

            # 4. Consulta a LLM se não houver cache válido.

            if not self.llm_cache_hit_:

                circuit = _LLM_CIRCUITS[provider]

                if circuit["open"]:

                    self.llm_status_ = "circuit_open"

                    self.llm_error_ = (
                        "Consulta bloqueada pelo circuit breaker. "
                        f"Motivo original: {circuit['reason']}"
                    )

                    print(
                        f"[AutoFE] Consulta ao {provider.upper()} "
                        "ignorada: circuit breaker aberto.",
                        flush=True
                    )

                    print(
                        f"[AutoFE] Motivo: {circuit['reason']}",
                        flush=True
                    )

                    return

                self.llm_attempted_ = True

                print(
                    f"[AutoFE] Consultando {provider.upper()}...",
                    flush=True
                )

                start = time.perf_counter()

                self.llm_suggestions_ = (
                    self.llm_client_.suggest_features(context)
                )

                print(
                    f"[AutoFE] Consulta concluída em "
                    f"{time.perf_counter() - start:.2f}s.",
                    flush=True
                )

                self.llm_rejected_suggestions_ = getattr(
                    self.llm_client_,
                    "last_rejected_suggestions_",
                    []
                )

                cache[cache_key] = self.llm_suggestions_

                self._save_cache(cache)

                self.llm_status_ = "success"

            # 5. Converte sugestões em oportunidades.

            print("[AutoFE] Convertendo sugestões...", flush=True)

            self.llm_opportunities_ = (
                self._convert_llm_suggestions(
                    self.llm_suggestions_
                )
            )

            print(
                f"[AutoFE] Provedor: {provider.upper()}",
                flush=True
            )

            print(
                f"[AutoFE] Sugestões válidas: "
                f"{len(self.llm_suggestions_)}",
                flush=True
            )

            print(
                f"[AutoFE] Sugestões rejeitadas: "
                f"{len(self.llm_rejected_suggestions_)}",
                flush=True
            )

            print(
                f"[AutoFE] Oportunidades da LLM: "
                f"{len(self.llm_opportunities_)}",
                flush=True
            )

            print(
                f"[AutoFE] Status da LLM: {self.llm_status_}",
                flush=True
            )

            print("[AutoFE] Integração concluída.", flush=True)

        except Exception as error:

            self.llm_error_ = (
                f"{type(error).__name__}: {str(error)}"
            )

            self.llm_suggestions_ = []
            self.llm_opportunities_ = []
            self.llm_rejected_suggestions_ = []
            self.llm_cache_hit_ = False

            if _is_quota_error(error):

                self.llm_status_ = "quota_exhausted"

                _open_llm_circuit(
                    provider,
                    self.llm_error_
                )

            else:

                self.llm_status_ = "failed"

            print(
                f"[AutoFE] AVISO: Não foi possível utilizar "
                f"{provider.upper()}.",
                flush=True
            )

            print(
                f"[AutoFE] Motivo: {self.llm_error_}",
                flush=True
            )

            print(
                "[AutoFE] Continuando com o AutoFE tradicional.",
                flush=True
            )

    # =================================================
    # FIT
    # =================================================

    def fit(self, X, y=None):

        if not isinstance(X, pd.DataFrame):
            raise TypeError("A entrada deve ser um DataFrame.")

        if X.empty:
            raise ValueError("O DataFrame está vazio.")

        if not isinstance(self.include_interactions, bool):
            raise ValueError(
                "include_interactions deve ser True ou False."
            )

        if not isinstance(self.use_llm, bool):
            raise ValueError(
                "use_llm deve ser True ou False."
            )

        if (
            isinstance(self.max_interactions, bool)
            or not isinstance(self.max_interactions, int)
            or self.max_interactions < 1
        ):
            raise ValueError(
                "max_interactions deve ser um inteiro maior que zero."
            )

        if self.problem_type not in (
            "classification",
            "regression"
        ):
            raise ValueError(
                "problem_type deve ser 'classification' ou 'regression'."
            )

        if (
            not isinstance(self.llm_provider, str)
            or self.llm_provider.lower() not in ("gemini", "groq")
        ):
            raise ValueError(
                "llm_provider deve ser 'gemini' ou 'groq'."
            )

        if self.include_interactions and y is None:
            raise ValueError(
                "O target y é necessário para selecionar interações."
            )

        if y is not None and len(y) != len(X):
            raise ValueError(
                "O tamanho de y deve ser igual ao número de linhas de X."
            )

        self.columns_in_ = X.columns.tolist()

        # 1. Oportunidades tradicionais.

        detector = OpportunityDetector(X)

        opportunities = detector.detect()

        # 2. Candidatas tradicionais a interação.

        traditional_interaction_candidates = []

        if self.include_interactions:

            traditional_interaction_candidates = (
                detector.detect_interactions(
                    max_interactions=self.max_interactions,
                    y=y,
                    problem_type=self.problem_type,
                    random_state=self.random_state
                )
            )

        # Mantém o nome antigo para compatibilidade.
        # Este atributo representa candidatas tradicionais,
        # não necessariamente features efetivamente geradas.

        self.interaction_opportunities_ = (
            traditional_interaction_candidates
        )

        self.traditional_interaction_candidates_ = (
            traditional_interaction_candidates
        )

        traditional_opportunities = (
            opportunities + traditional_interaction_candidates
        )

        # 3. Integração com a LLM.

        self._run_llm(X)

        # 4. Junta as candidatas tradicionais e as da LLM.
        # Este campo não afirma que as features foram executadas.

        self.interaction_candidates_ = (
            self._collect_interaction_candidates(
                traditional_candidates=(
                    traditional_interaction_candidates
                ),
                llm_opportunities=self.llm_opportunities_
            )
        )

        # 5. União das oportunidades de geração.

        self.opportunities_ = self._merge_opportunities(
            traditional_opportunities,
            self.llm_opportunities_
        )

        # 6. Geração das features.

        print("[AutoFE] Gerando features...", flush=True)

        generator = FeatureGenerator(X)

        generated_df, self.generation_log_ = (
            generator.generate_candidates(
                self.opportunities_
            )
        )

        # Somente features efetivamente registradas no log.
        self.generated_features_ = [
            item["feature"]
            for item in self.generation_log_
            if isinstance(item, dict) and item.get("feature")
        ]

        print(
            f"[AutoFE] Features geradas: "
            f"{len(self.generated_features_)}",
            flush=True
        )

        return self

    # =================================================
    # TRANSFORM
    # =================================================

    def transform(self, X):

        if not isinstance(X, pd.DataFrame):
            raise TypeError("A entrada deve ser um DataFrame.")

        if not hasattr(self, "generation_log_"):
            raise RuntimeError(
                "O AutoFETransformer precisa ser ajustado antes do transform."
            )

        if X.columns.tolist() != self.columns_in_:
            raise ValueError(
                "As colunas recebidas são diferentes das utilizadas no fit."
            )

        generator = FeatureGenerator(X)

        for item in self.generation_log_:

            transformation = item["transformation"]

            if transformation == "missing_indicator":

                generator.generate_missing_indicator(
                    item["column"]
                )

            elif transformation == "log1p":

                generator.generate_log_transform(
                    item["column"]
                )

            elif transformation == "interaction_product":

                columns = item["column"]

                generator.generate_interaction(
                    columns[0],
                    columns[1]
                )

        return generator.df.copy()
