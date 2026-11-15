
import numpy as np
import pandas as pd

from scipy.sparse import issparse

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    make_scorer
)

from src.analysis.autofe_transformer import AutoFETransformer
from src.analysis.feature_selector import FeatureSelector


class DenseTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if issparse(X):
            return X.toarray()
        return X


class ModelEvaluator:

    AVAILABLE_MODELS = (
        "decision_tree",
        "random_forest",
        "gradient_boosting",
        "dummy"
    )

    AVAILABLE_LLM_PROVIDERS = ("gemini", "groq")

    def __init__(
        self,
        problem_type="classification",
        random_state=42,
        model_name="decision_tree",
        model_params=None,
        positive_label=None,
        llm_provider="groq",
        llm_model=None
    ):

        self.problem_type = problem_type
        self.random_state = random_state
        self.model_name = model_name
        self.model_params = model_params
        self.positive_label = positive_label
        self.llm_provider = llm_provider
        self.llm_model = llm_model

        self._validate_configuration()
        self._create_model()

        self.last_feature_audit_ = []
        self.last_fold_results_ = {}

    # =================================================
    # VALIDACAO DA CONFIGURACAO
    # =================================================

    def _validate_configuration(self):

        if self.problem_type not in ("classification", "regression"):
            raise ValueError(
                "problem_type deve ser 'classification' ou 'regression'."
            )

        if (
            self.random_state is not None
            and (
                isinstance(self.random_state, bool)
                or not isinstance(self.random_state, (int, np.integer))
                or self.random_state < 0
            )
        ):
            raise ValueError(
                "random_state deve ser um inteiro nao negativo ou None."
            )

        if self.model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Modelo invalido: {self.model_name}. "
                f"Modelos disponiveis: {self.AVAILABLE_MODELS}"
            )

        if self.model_params is not None and not isinstance(
            self.model_params, dict
        ):
            raise TypeError(
                "model_params deve ser um dicionario ou None."
            )

        if (
            self.problem_type == "regression"
            and self.positive_label is not None
        ):
            raise ValueError(
                "positive_label so pode ser utilizado em classificacao."
            )

        if (
            not isinstance(self.llm_provider, str)
            or self.llm_provider.lower() not in self.AVAILABLE_LLM_PROVIDERS
        ):
            raise ValueError(
                f"Provedor LLM invalido: {self.llm_provider}. "
                f"Provedores disponiveis: {self.AVAILABLE_LLM_PROVIDERS}"
            )

        if self.llm_model is not None and not isinstance(self.llm_model, str):
            raise TypeError("llm_model deve ser uma string ou None.")

    # =================================================
    # VALIDACAO DOS DADOS
    # =================================================

    def _validate_data(self, X, y):

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X deve ser um DataFrame.")

        if X.empty:
            raise ValueError("X esta vazio.")

        if X.columns.duplicated().any():
            raise ValueError("X possui nomes de colunas duplicados.")

        if y is None:
            raise ValueError("A variavel alvo y e obrigatoria.")

        if isinstance(y, pd.DataFrame):
            raise ValueError("y deve possuir apenas uma dimensao.")

        try:
            y_array = np.asarray(y)
        except (TypeError, ValueError):
            raise ValueError(
                "Nao foi possivel converter y para um vetor."
            )

        if y_array.ndim != 1:
            raise ValueError("y deve ser unidimensional.")

        if len(X) != len(y_array):
            raise ValueError(
                "X e y possuem quantidades diferentes de linhas."
            )

        if pd.isna(y_array).any():
            raise ValueError(
                "A variavel alvo possui valores ausentes."
            )

        if self.problem_type == "classification":

            if pd.Series(y_array).nunique() < 2:
                raise ValueError(
                    "Classificacao exige pelo menos duas classes."
                )

            if np.issubdtype(y_array.dtype, np.number):
                if not np.isfinite(y_array.astype(float)).all():
                    raise ValueError(
                        "O target possui valores infinitos ou nao finitos."
                    )

        else:

            try:
                y_array = pd.to_numeric(
                    pd.Series(y_array),
                    errors="raise"
                ).to_numpy(dtype=float)
            except (ValueError, TypeError):
                raise ValueError(
                    "Para regressao, y deve possuir valores numericos."
                )

            if not np.isfinite(y_array).all():
                raise ValueError(
                    "O target de regressao possui valores infinitos "
                    "ou nao finitos."
                )

            if np.unique(y_array).size < 2:
                raise ValueError(
                    "Regressao exige variacao na variavel alvo."
                )

        return y_array

    # =================================================
    # CLASSE POSITIVA
    # =================================================

    def _get_positive_label(self, y):

        classes = np.unique(y)

        if self.positive_label is not None:
            if self.positive_label not in classes:
                raise ValueError(
                    f"A classe positiva '{self.positive_label}' "
                    "nao existe na variavel alvo."
                )
            return self.positive_label

        if 1 in classes:
            return 1

        return classes[-1]

    # =================================================
    # CRIACAO DO MODELO
    # =================================================

    def _create_model(self):

        if self.problem_type == "classification":
            models = {
                "decision_tree": DecisionTreeClassifier,
                "random_forest": RandomForestClassifier,
                "gradient_boosting": GradientBoostingClassifier,
                "dummy": DummyClassifier
            }
        else:
            models = {
                "decision_tree": DecisionTreeRegressor,
                "random_forest": RandomForestRegressor,
                "gradient_boosting": GradientBoostingRegressor,
                "dummy": DummyRegressor
            }

        model_class = models[self.model_name]

        if self.model_name in (
            "decision_tree",
            "random_forest",
            "gradient_boosting"
        ):
            default_params = {"random_state": self.random_state}
        else:
            default_params = {}

        if self.model_params is not None:
            default_params.update(self.model_params)

        self.model = model_class(**default_params)

    # =================================================
    # TIPOS DE COLUNAS
    # =================================================

    def identify_column_types(self, X):

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X deve ser um DataFrame.")

        numerical_columns = X.select_dtypes(
            include=np.number
        ).columns.tolist()

        categorical_columns = X.select_dtypes(
            exclude=np.number
        ).columns.tolist()

        return numerical_columns, categorical_columns

    # =================================================
    # PRE-PROCESSAMENTO
    # =================================================

    def build_preprocessor(self):

        numerical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        keep_empty_features=True
                    )
                )
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent",
                        keep_empty_features=True
                    )
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore")
                )
            ]
        )

        return ColumnTransformer(
            transformers=[
                (
                    "numerical",
                    numerical_pipeline,
                    make_column_selector(dtype_include=np.number)
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    make_column_selector(dtype_exclude=np.number)
                )
            ],
            remainder="drop"
        )

    # =================================================
    # PIPELINE
    # =================================================

    def build_model_pipeline(
        self,
        original_columns,
        use_autofe=False,
        use_feature_selection=False,
        selector_k=3,
        include_interactions=True,
        use_llm=False,
        llm_model=None,
        llm_provider=None
    ):

        self._validate_configuration()

        if llm_provider is None:
            llm_provider = self.llm_provider

        llm_provider = llm_provider.lower()

        if llm_provider not in self.AVAILABLE_LLM_PROVIDERS:
            raise ValueError(
                f"Provedor LLM invalido: {llm_provider}."
            )

        if llm_model is None:
            llm_model = self.llm_model

        if use_llm and not use_autofe:
            raise ValueError(
                "A integracao com LLM exige use_autofe=True."
            )

        preprocessor = self.build_preprocessor()
        steps = []

        if use_autofe:
            steps.append(
                (
                    "autofe",
                    AutoFETransformer(
                        include_interactions=include_interactions,
                        problem_type=self.problem_type,
                        random_state=self.random_state,
                        use_llm=use_llm,
                        llm_model=llm_model,
                        llm_provider=llm_provider
                    )
                )
            )

        if use_feature_selection:

            if not use_autofe:
                raise ValueError(
                    "A selecao de features exige use_autofe=True."
                )

            steps.append(
                (
                    "feature_selector",
                    FeatureSelector(
                        original_columns=original_columns,
                        k=selector_k,
                        problem_type=self.problem_type,
                        random_state=self.random_state
                    )
                )
            )

        steps.append(("preprocessor", preprocessor))

        if self.model_name == "gradient_boosting":
            steps.append(("to_dense", DenseTransformer()))

        steps.append(("model", self.model))

        return Pipeline(steps=steps)

    # =================================================
    # VALIDACAO CRUZADA E METRICAS
    # =================================================

    def _get_cv_and_scoring(self, cv, y=None):

        if self.problem_type == "classification":

            if cv is None:
                cv = StratifiedKFold(
                    n_splits=5,
                    shuffle=True,
                    random_state=self.random_state
                )

            scoring = {
                "accuracy": "accuracy",
                "precision": make_scorer(
                    precision_score,
                    average="macro",
                    zero_division=0
                ),
                "recall": make_scorer(
                    recall_score,
                    average="macro",
                    zero_division=0
                ),
                "f1": make_scorer(
                    f1_score,
                    average="macro",
                    zero_division=0
                )
            }

            if y is not None:

                classes = np.unique(y)
                positive_label = self._get_positive_label(y)

                self.positive_label_ = positive_label
                self.classes_ = classes.tolist()

                for class_label in classes:

                    class_name = str(class_label)

                    scoring[f"precision_class_{class_name}"] = make_scorer(
                        precision_score,
                        labels=[class_label],
                        average="macro",
                        zero_division=0
                    )

                    scoring[f"recall_class_{class_name}"] = make_scorer(
                        recall_score,
                        labels=[class_label],
                        average="macro",
                        zero_division=0
                    )

                    scoring[f"f1_class_{class_name}"] = make_scorer(
                        f1_score,
                        labels=[class_label],
                        average="macro",
                        zero_division=0
                    )

                scoring["precision_positive_class"] = make_scorer(
                    precision_score,
                    labels=[positive_label],
                    average="macro",
                    zero_division=0
                )

                scoring["recall_positive_class"] = make_scorer(
                    recall_score,
                    labels=[positive_label],
                    average="macro",
                    zero_division=0
                )

                scoring["f1_positive_class"] = make_scorer(
                    f1_score,
                    labels=[positive_label],
                    average="macro",
                    zero_division=0
                )

        else:

            if cv is None:
                cv = KFold(
                    n_splits=5,
                    shuffle=True,
                    random_state=self.random_state
                )

            scoring = {
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
                "r2": "r2"
            }

        return cv, scoring

    # =================================================
    # STATUS DA LLM
    # =================================================

    def _get_llm_status(self, autofe):

        status = getattr(autofe, "llm_status_", "disabled")
        error = getattr(autofe, "llm_error_", None)
        attempted = getattr(autofe, "llm_attempted_", False)
        cache_hit = getattr(autofe, "llm_cache_hit_", False)

        status = str(status).lower() if status is not None else "unknown"

        error_text = str(error).lower() if error is not None else ""

        circuit_breaker_terms = (
            "circuit breaker",
            "circuit_open",
            "circuit open",
            "consulta ignorada",
            "chamadas bloqueadas"
        )

        if any(term in error_text for term in circuit_breaker_terms):
            status = "circuit_open"

        if cache_hit:
            status = "cache"

        valid_statuses = (
            "success",
            "cache",
            "failed",
            "quota_exhausted",
            "circuit_open",
            "disabled"
        )

        if status not in valid_statuses:

            if not attempted and status in (
                "skipped",
                "not_started",
                "unknown"
            ):
                status = "disabled"

            elif error is not None:
                status = "failed"

            else:
                status = "unknown"

        return {
            "llm_status": status,
            "llm_error": error,
            "llm_attempted": bool(attempted),
            "llm_cache_hit": bool(cache_hit)
        }

    # =================================================
    # NORMALIZACAO DE CANDIDATAS
    # =================================================

    def _normalize_interaction_candidates(self, candidates):

        normalized = []

        for item in candidates or []:

            if isinstance(item, str):
                feature = item

            elif isinstance(item, dict):

                feature = item.get("feature")

                if not feature:

                    columns = item.get("columns")

                    if columns is None:
                        columns = item.get("column")

                    if (
                        isinstance(columns, (list, tuple))
                        and len(columns) == 2
                    ):
                        feature = f"{columns[0]}_x_{columns[1]}"

            else:
                feature = None

            if feature:
                normalized.append(str(feature))

        return list(dict.fromkeys(normalized))

    # =================================================
    # AUDITORIA DA GERACAO
    # =================================================

    def _get_generation_audit(self, autofe):
        """
        generation_log_ e a fonte principal das transformacoes
        efetivamente executadas.

        interaction_candidates_ representa as candidatas detectadas.
        """

        generation_log = getattr(
            autofe,
            "generation_log_",
            []
        ) or []

        generated_features = getattr(
            autofe,
            "generated_features_",
            []
        ) or []

        if not generated_features and generation_log:

            generated_features = [
                item.get("feature")
                for item in generation_log
                if isinstance(item, dict) and item.get("feature")
            ]

        generated_interactions = []

        for item in generation_log:

            if not isinstance(item, dict):
                continue

            transformation = str(
                item.get("transformation", "")
            ).lower()

            column = item.get("column")
            feature = item.get("feature")

            is_interaction = (
                transformation == "interaction_product"
                or (
                    isinstance(column, (list, tuple))
                    and len(column) == 2
                )
            )

            if is_interaction:

                if feature:
                    generated_interactions.append(str(feature))

                elif isinstance(column, (list, tuple)):
                    generated_interactions.append(
                        f"{column[0]}_x_{column[1]}"
                    )

        # Compatibilidade com logs antigos.
        if not generation_log:

            generated_interactions = [
                feature
                for feature in generated_features
                if isinstance(feature, str) and "_x_" in feature
            ]

        generated_features = [
            str(feature)
            for feature in generated_features
            if feature
        ]

        generated_features = list(
            dict.fromkeys(generated_features)
        )

        generated_interactions = list(
            dict.fromkeys(generated_interactions)
        )

        return {
            "generation_log": generation_log,
            "generated_features": generated_features,
            "generated_interactions": generated_interactions
        }

    # =================================================
    # AUDITORIA DAS FEATURES POR FOLD
    # =================================================

    def _audit_features(
        self,
        estimators,
        use_autofe,
        use_feature_selection,
        use_llm=False
    ):

        self.last_feature_audit_ = []

        if not use_autofe:
            return

        for fold_number, estimator in enumerate(estimators, start=1):

            autofe = estimator.named_steps["autofe"]

            generation_information = self._get_generation_audit(autofe)

            if use_llm:
                llm_information = self._get_llm_status(autofe)
            else:
                llm_information = {
                    "llm_status": "not_applicable",
                    "llm_error": None,
                    "llm_attempted": False,
                    "llm_cache_hit": False
                }

            # Candidatas tradicionais.
            traditional_candidates = self._normalize_interaction_candidates(
                getattr(
                    autofe,
                    "traditional_interaction_candidates_",
                    []
                )
            )

            # Candidatas sugeridas pela LLM.
            llm_interaction_candidates = []

            if use_llm:

                llm_opportunities = getattr(
                    autofe,
                    "llm_opportunities_",
                    []
                ) or []

                for opportunity in llm_opportunities:

                    if not isinstance(opportunity, dict):
                        continue

                    if opportunity.get("type") != "interaction_product":
                        continue

                    llm_interaction_candidates.append(opportunity)

            llm_interaction_candidates = (
                self._normalize_interaction_candidates(
                    llm_interaction_candidates
                )
            )

            # União das candidatas.
            interaction_candidates = list(
                dict.fromkeys(
                    traditional_candidates
                    + llm_interaction_candidates
                )
            )

            # Interações efetivamente geradas.
            generated_interactions = generation_information[
                "generated_interactions"
            ]

            # Candidatas que não foram executadas.
            not_generated = [
                feature
                for feature in interaction_candidates
                if feature not in generated_interactions
            ]

            fold_audit = {
                "fold": fold_number,

                "llm_provider": (
                    getattr(autofe, "llm_provider_", "unknown")
                    if use_llm else "not_applicable"
                ),

                "llm_model": (
                    getattr(
                        getattr(autofe, "llm_client_", None),
                        "model",
                        None
                    )
                    if use_llm else None
                ),

                "opportunities": getattr(
                    autofe,
                    "opportunities_",
                    []
                ),

                "generation_log": generation_information[
                    "generation_log"
                ],

                "generated_features": generation_information[
                    "generated_features"
                ],

                # Mantido para compatibilidade com main.py.
                # Representa interacoes efetivamente executadas.
                "interaction_opportunities": generated_interactions,

                # Todas as candidatas identificadas.
                "interaction_candidates": interaction_candidates,

                # Candidatas tradicionais.
                "traditional_interaction_candidates": (
                    traditional_candidates
                ),

                # Candidatas sugeridas pela LLM.
                "llm_interaction_candidates": (
                    llm_interaction_candidates
                ),

                # Candidatas que nao foram efetivamente geradas.
                "interaction_candidates_not_generated": not_generated,

                # Sugestoes originais da LLM.
                "llm_suggestions": (
                    getattr(autofe, "llm_suggestions_", [])
                    if use_llm else []
                ),

                # Oportunidades convertidas pela LLM.
                "llm_opportunities": (
                    getattr(autofe, "llm_opportunities_", [])
                    if use_llm else []
                ),

                "llm_rejected_suggestions": (
                    getattr(autofe, "llm_rejected_suggestions_", [])
                    if use_llm else []
                ),

                "llm_status": llm_information["llm_status"],
                "llm_error": llm_information["llm_error"],
                "llm_attempted": llm_information["llm_attempted"],
                "llm_cache_hit": llm_information["llm_cache_hit"],

                "feature_scores": {},
                "selected_features": []
            }

            if use_feature_selection:

                selector = estimator.named_steps["feature_selector"]

                fold_audit["feature_scores"] = (
                    selector.get_feature_scores()
                )

                fold_audit["selected_features"] = (
                    selector.selected_features_.copy()
                )

            self.last_feature_audit_.append(fold_audit)

    # =================================================
    # RESUMO DOS RESULTADOS
    # =================================================

    def _summarize_results(self, results, scoring):

        summary = {}
        self.last_fold_results_ = {}

        for metric in scoring:

            scores = np.asarray(
                results[f"test_{metric}"],
                dtype=float
            )

            if metric in ("mae", "rmse"):
                scores = -scores

            fold_values = scores.tolist()

            self.last_fold_results_[metric] = fold_values.copy()

            summary[metric] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "folds": fold_values
            }

        return summary

    # =================================================
    # AVALIACAO
    # =================================================

    def evaluate(
        self,
        X,
        y,
        cv=None,
        use_autofe=False,
        use_feature_selection=False,
        selector_k=3,
        include_interactions=True,
        use_llm=False,
        llm_model=None,
        llm_provider=None
    ):

        self._validate_configuration()

        if llm_provider is None:
            llm_provider = self.llm_provider

        llm_provider = llm_provider.lower()

        if llm_provider not in self.AVAILABLE_LLM_PROVIDERS:
            raise ValueError(
                f"Provedor LLM invalido: {llm_provider}."
            )

        if llm_model is None:
            llm_model = self.llm_model

        if use_llm and not use_autofe:
            raise ValueError(
                "Para utilizar LLM, use_autofe deve ser True."
            )

        y_array = self._validate_data(X, y)

        self.last_feature_audit_ = []
        self.last_fold_results_ = {}

        pipeline = self.build_model_pipeline(
            original_columns=list(X.columns),
            use_autofe=use_autofe,
            use_feature_selection=use_feature_selection,
            selector_k=selector_k,
            include_interactions=include_interactions,
            use_llm=use_llm,
            llm_model=llm_model,
            llm_provider=llm_provider
        )

        cv, scoring = self._get_cv_and_scoring(cv, y_array)

        results = cross_validate(
            pipeline,
            X,
            y_array,
            cv=cv,
            scoring=scoring,
            error_score="raise",
            return_estimator=True
        )

        self._audit_features(
            estimators=results["estimator"],
            use_autofe=use_autofe,
            use_feature_selection=use_feature_selection,
            use_llm=use_llm
        )

        return self._summarize_results(results, scoring)

    # =================================================
    # COMPATIBILIDADE
    # =================================================

    def evaluate_classifier(self, X, y):

        if self.problem_type != "classification":
            raise ValueError(
                "evaluate_classifier exige problem_type='classification'."
            )

        return self.evaluate(X, y)
