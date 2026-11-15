
import pandas as pd

from collections import Counter
from pathlib import Path
from datetime import datetime

from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import KFold, StratifiedKFold

from src.analysis.autofe_transformer import reset_llm_circuit
from src.evaluation.model_evaluator import ModelEvaluator


# =====================================================
# CONFIGURACOES GERAIS
# =====================================================

DATASET_NAMES = ["diabetes"]

MODEL_NAMES = ["random_forest"]

N_SPLITS = 5
RANDOM_STATE = 42
SELECTOR_K = 3

DATA_PATHS = {
    "titanic": "data/raw/Titanic-Dataset.csv"
}


# Para executar todos os modelos:
#
# MODEL_NAMES = [
#     "decision_tree",
#     "random_forest",
#     "gradient_boosting",
#     "dummy"
# ]


# =====================================================
# CONFIGURACAO DA LLM
# =====================================================

LLM_PROVIDER = "groq"  # "groq" ou "gemini"

# None utiliza o modelo padrao definido no cliente/.env.
LLM_MODEL = None


# =====================================================
# CONFIGURACOES DOS DATASETS
# =====================================================

DATASET_CONFIGS = {
    "titanic": {
        "target": "Survived",
        "problem_type": "classification",
        "drop_columns": ["PassengerId", "Name", "Ticket"],
        "cv_type": "stratified",
        "class_names": {
            0: "Nao sobreviveu",
            1: "Sobreviveu"
        }
    },

    "breast_cancer": {
        "target": "target",
        "problem_type": "classification",
        "drop_columns": [],
        "cv_type": "stratified",
        "class_names": {
            0: "Maligno",
            1: "Benigno"
        }
    },

    "diabetes": {
        "target": "target",
        "problem_type": "regression",
        "drop_columns": [],
        "cv_type": "kfold",
        "class_names": None
    }
}


# =====================================================
# CENARIOS EXPERIMENTAIS
# =====================================================

SCENARIOS = {
    "BASELINE": {
        "use_autofe": False,
        "use_feature_selection": False,
        "include_interactions": False,
        "use_llm": False
    },

    "AUTOFE TRADICIONAL": {
        "use_autofe": True,
        "use_feature_selection": False,
        "include_interactions": False,
        "use_llm": False
    },

    "AUTOFE + LLM": {
        "use_autofe": True,
        "use_feature_selection": False,
        "include_interactions": False,
        "use_llm": True
    },

    "AUTOFE COM INTERACOES": {
        "use_autofe": True,
        "use_feature_selection": False,
        "include_interactions": True,
        "use_llm": False
    },

    "AUTOFE + INTERACOES + SELECAO": {
        "use_autofe": True,
        "use_feature_selection": True,
        "include_interactions": True,
        "use_llm": False
    }
}


# =====================================================
# CARREGAMENTO DOS DATASETS
# =====================================================

def load_selected_dataset(dataset_name):

    print("===== CARREGANDO DATASET =====")

    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Dataset nao reconhecido: {dataset_name}. "
            f"Opcoes disponiveis: {list(DATASET_CONFIGS.keys())}"
        )

    if dataset_name == "titanic":

        df = pd.read_csv(DATA_PATHS["titanic"])

    elif dataset_name == "diabetes":

        dataset = load_diabetes(as_frame=True)
        df = dataset.frame.copy()

    elif dataset_name == "breast_cancer":

        dataset = load_breast_cancer(as_frame=True)
        df = dataset.frame.copy()

    else:

        raise ValueError(
            f"Nao existe carregador para: {dataset_name}"
        )

    return df, DATASET_CONFIGS[dataset_name]


# =====================================================
# METADADOS DA LLM
# =====================================================

def get_llm_metadata(folds, use_llm):

    if not use_llm:
        return {
            "llm_provider": "not_applicable",
            "llm_model": None
        }

    provider = LLM_PROVIDER
    model = LLM_MODEL

    for fold in folds:

        fold_provider = fold.get("llm_provider")
        fold_model = fold.get("llm_model")

        if fold_provider:
            provider = fold_provider

        if fold_model:
            model = fold_model

        if provider and model:
            break

    return {
        "llm_provider": provider,
        "llm_model": model
    }


# =====================================================
# RESUMO DO STATUS DA LLM
# =====================================================

def summarize_llm_status(folds, use_llm=True):

    total_folds = len(folds)

    if not use_llm:
        return {
            "scenario_status": "not_applicable",
            "total_folds": total_folds,
            "successful_folds": 0,
            "cache_folds": 0,
            "failed_folds": 0,
            "quota_exhausted_folds": 0,
            "circuit_open_folds": 0,
            "unknown_folds": 0
        }

    status_counter = Counter(
        fold.get("llm_status", "unknown")
        for fold in folds
    )

    successful_folds = (
        status_counter["success"] + status_counter["cache"]
    )

    failed_folds = status_counter["failed"]
    quota_exhausted_folds = status_counter["quota_exhausted"]
    circuit_open_folds = status_counter["circuit_open"]

    unknown_folds = (
        status_counter["unknown"] + status_counter["disabled"]
    )

    if total_folds > 0 and successful_folds == total_folds:
        scenario_status = "complete"
    else:
        scenario_status = "partial_or_incomplete"

    return {
        "scenario_status": scenario_status,
        "total_folds": total_folds,
        "successful_folds": successful_folds,
        "cache_folds": status_counter["cache"],
        "failed_folds": failed_folds,
        "quota_exhausted_folds": quota_exhausted_folds,
        "circuit_open_folds": circuit_open_folds,
        "unknown_folds": unknown_folds
    }


# =====================================================
# IMPRESSAO DO STATUS DA LLM
# =====================================================

def print_llm_status_summary(audit, scenario_name, use_llm=True):

    print("\n===== STATUS DA INTEGRACAO COM LLM =====")

    if not use_llm:
        print(
            "LLM: NAO SE APLICA "
            "(cenario sem integracao com LLM)."
        )
        return

    if not audit:
        print("Nenhuma auditoria disponivel.")
        return

    summary = summarize_llm_status(
        audit,
        use_llm=True
    )

    status_counter = Counter(
        fold.get("llm_status", "unknown")
        for fold in audit
    )

    for status in (
        "success",
        "cache",
        "failed",
        "quota_exhausted",
        "circuit_open",
        "disabled",
        "unknown"
    ):

        print(
            f"{status}: "
            f"{status_counter[status]}/{len(audit)} folds"
        )

    print("\n===== COMPLETUDE DO CENARIO =====")

    if summary["scenario_status"] == "complete":

        print("STATUS: COMPLETO")

    else:

        print("STATUS: PARCIAL/INCOMPLETO")

        print(
            "ATENCAO: as metricas deste cenario nao devem "
            "ser interpretadas como uma avaliacao completa "
            "da contribuicao da LLM."
        )

    print(
        f"Folds com resposta valida da LLM: "
        f"{summary['successful_folds']}/"
        f"{summary['total_folds']}"
    )

    for fold in audit:

        status = fold.get("llm_status", "unknown")
        error = fold.get("llm_error")

        if status in (
            "failed",
            "quota_exhausted",
            "circuit_open",
            "unknown"
        ):

            print(
                f"\nFold {fold.get('fold', '?')} - {status}"
            )

            if error:
                print(f"Erro: {error}")


# =====================================================
# AUDITORIA DAS FEATURES
# =====================================================

def print_feature_audit(
    audit,
    scenario_name,
    use_llm=False
):

    print("\n" + "=" * 55)
    print(f"AUDITORIA: {scenario_name}")
    print("=" * 55)

    if not audit:
        print("Nenhuma auditoria disponivel.")
        return

    generated_counter = Counter()
    selected_counter = Counter()
    interaction_counter = Counter()
    candidate_counter = Counter()
    llm_counter = Counter()

    cache_hits = 0

    for fold in audit:

        generated_counter.update(
            fold.get("generated_features", [])
        )

        selected_counter.update(
            fold.get("selected_features", [])
        )

        # Interacoes efetivamente geradas.
        interaction_features = []

        for item in fold.get("interaction_opportunities", []):

            if isinstance(item, str):

                interaction_features.append(item)

            elif isinstance(item, dict) and item.get("feature"):

                interaction_features.append(item["feature"])

        interaction_counter.update(interaction_features)

        # Candidatas a interacao detectadas.
        candidate_features = []

        for item in fold.get("interaction_candidates", []):

            if isinstance(item, str):

                candidate_features.append(item)

            elif isinstance(item, dict) and item.get("feature"):

                candidate_features.append(item["feature"])

        candidate_counter.update(candidate_features)

        if fold.get("llm_cache_hit", False):
            cache_hits += 1

        if use_llm:

            for suggestion in fold.get("llm_suggestions", []):

                if not isinstance(suggestion, dict):
                    continue

                operation = suggestion.get("operation")

                if operation == "log1p":

                    feature = (
                        f"{suggestion.get('column')}_log1p"
                    )

                elif operation == "missing_indicator":

                    feature = (
                        f"{suggestion.get('column')}_missing"
                    )

                elif operation == "interaction_product":

                    columns = suggestion.get("columns", [])

                    if len(columns) != 2:
                        continue

                    feature = (
                        f"{columns[0]}_x_{columns[1]}"
                    )

                else:

                    continue

                llm_counter[feature] += 1

    print_llm_status_summary(
        audit,
        scenario_name,
        use_llm=use_llm
    )

    print("\n===== FEATURES GERADAS =====")

    if generated_counter:

        for feature, frequency in generated_counter.most_common():

            print(
                f"{feature}: {frequency}/{len(audit)} folds"
            )

    else:

        print("Nenhuma feature gerada.")

    print("\n===== INTERACOES GERADAS =====")

    if interaction_counter:

        for feature, frequency in interaction_counter.most_common():

            print(
                f"{feature}: {frequency}/{len(audit)} folds"
            )

    else:

        print("Nenhuma interacao gerada.")

    print("\n===== CANDIDATAS A INTERACAO =====")

    if candidate_counter:

        for feature, frequency in candidate_counter.most_common():

            print(
                f"{feature}: {frequency}/{len(audit)} folds"
            )

    else:

        print("Nenhuma candidata a interacao registrada.")

    print("\n===== SUGESTOES DA LLM =====")

    if not use_llm:

        print(
            "NAO SE APLICA: este cenario nao utiliza LLM."
        )

    elif llm_counter:

        for feature, frequency in llm_counter.most_common():

            print(
                f"{feature}: {frequency}/{len(audit)} folds"
            )

    else:

        print("Nenhuma sugestao da LLM registrada.")

    print("\n===== CACHE DA LLM =====")

    if use_llm:

        print(
            f"Cache utilizado: {cache_hits}/{len(audit)} folds"
        )

    else:

        print("NAO SE APLICA.")

    print("\n===== FEATURES SELECIONADAS =====")

    if selected_counter:

        for feature, frequency in selected_counter.most_common():

            print(
                f"{feature}: {frequency}/{len(audit)} folds"
            )

    else:

        print("Selecao nao realizada neste cenario.")


# =====================================================
# ATUALIZACAO DOS ARQUIVOS CONSOLIDADOS
# =====================================================

def update_consolidated_csv(
    file_path,
    new_data,
    dataset_name,
    llm_provider
):

    file_path = Path(file_path)

    if file_path.exists():

        old_data = pd.read_csv(file_path)

        if "dataset" in old_data.columns:

            same_dataset = (
                old_data["dataset"] == dataset_name
            )

            if "llm_provider" in old_data.columns:

                old_provider = (
                    old_data["llm_provider"]
                    .fillna("not_applicable")
                )

                # Remove:
                # 1. Resultados anteriores do provedor atual.
                # 2. Resultados comuns, que nao utilizam LLM.
                same_provider = old_provider.isin([
                    llm_provider or "not_applicable",
                    "not_applicable"
                ])

            else:

                same_provider = pd.Series(
                    True,
                    index=old_data.index
                )

            old_data = old_data[
                ~(same_dataset & same_provider)
            ]

        updated_data = pd.concat(
            [old_data, new_data],
            ignore_index=True
        )

    else:

        updated_data = new_data.copy()

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    updated_data.to_csv(
        file_path,
        index=False,
        encoding="utf-8-sig"
    )


# =====================================================
# EXPORTACAO DOS RESULTADOS
# =====================================================

def save_results_to_csv(
    dataset_name,
    all_results,
    all_audits
):

    print("\n" + "=" * 75)
    print("EXPORTANDO RESULTADOS PARA CSV")
    print("=" * 75)

    now = datetime.now()

    run_timestamp = now.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    run_id = now.strftime(
        "%Y%m%d_%H%M%S"
    )

    metrics_dir = Path("results/metrics")
    audits_dir = Path("results/audits")

    metrics_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    audits_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # -------------------------------------------------
    # 1. METRICAS MEDIAS
    # -------------------------------------------------

    metric_rows = []

    for model_name, scenarios in all_results.items():

        for scenario_name, metrics in scenarios.items():

            scenario_audit = all_audits.get(
                model_name, {}
            ).get(scenario_name, [])

            use_llm = SCENARIOS[scenario_name]["use_llm"]

            llm_summary = summarize_llm_status(
                scenario_audit,
                use_llm=use_llm
            )

            llm_metadata = get_llm_metadata(
                scenario_audit,
                use_llm
            )

            for metric_name, values in metrics.items():

                metric_rows.append({

                    "run_id": run_id,
                    "timestamp": run_timestamp,
                    "dataset": dataset_name,
                    "model": model_name,
                    "scenario": scenario_name,
                    "metric": metric_name,

                    "mean": values["mean"],
                    "std": values["std"],

                    "scenario_status": (
                        llm_summary["scenario_status"]
                    ),

                    "llm_total_folds": (
                        llm_summary["total_folds"]
                    ),

                    "llm_successful_folds": (
                        llm_summary["successful_folds"]
                    ),

                    "llm_cache_folds": (
                        llm_summary["cache_folds"]
                    ),

                    "llm_failed_folds": (
                        llm_summary["failed_folds"]
                    ),

                    "llm_quota_exhausted_folds": (
                        llm_summary["quota_exhausted_folds"]
                    ),

                    "llm_circuit_open_folds": (
                        llm_summary["circuit_open_folds"]
                    ),

                    "llm_unknown_folds": (
                        llm_summary["unknown_folds"]
                    ),

                    "llm_provider": (
                        llm_metadata["llm_provider"]
                    ),

                    "llm_model": (
                        llm_metadata["llm_model"]
                    )
                })

    metric_columns = [
        "run_id",
        "timestamp",
        "dataset",
        "model",
        "scenario",
        "metric",
        "mean",
        "std",
        "scenario_status",
        "llm_total_folds",
        "llm_successful_folds",
        "llm_cache_folds",
        "llm_failed_folds",
        "llm_quota_exhausted_folds",
        "llm_circuit_open_folds",
        "llm_unknown_folds",
        "llm_provider",
        "llm_model"
    ]

    metrics_df = pd.DataFrame(
        metric_rows,
        columns=metric_columns
    )

    metrics_file = (
        metrics_dir / f"{dataset_name}_results.csv"
    )

    metrics_df.to_csv(
        metrics_file,
        index=False,
        encoding="utf-8-sig"
    )

    consolidated_metrics_file = (
        metrics_dir / "all_datasets_metrics.csv"
    )

    update_consolidated_csv(
        consolidated_metrics_file,
        metrics_df,
        dataset_name,
        LLM_PROVIDER
    )

    print("\nMetricas medias exportadas:")
    print(metrics_file)
    print(consolidated_metrics_file)

    # -------------------------------------------------
    # 2. METRICAS INDIVIDUAIS DOS FOLDS
    # -------------------------------------------------

    fold_rows = []

    for model_name, scenarios in all_results.items():

        for scenario_name, metrics in scenarios.items():

            scenario_audit = all_audits.get(
                model_name, {}
            ).get(scenario_name, [])

            use_llm = SCENARIOS[scenario_name]["use_llm"]

            audit_by_fold = {
                item.get("fold", index + 1): item
                for index, item in enumerate(scenario_audit)
            }

            for metric_name, values in metrics.items():

                for fold_index, fold_value in enumerate(
                    values.get("folds", []),
                    start=1
                ):

                    fold_audit = audit_by_fold.get(
                        fold_index,
                        {}
                    )

                    fold_rows.append({

                        "run_id": run_id,
                        "timestamp": run_timestamp,
                        "dataset": dataset_name,
                        "model": model_name,
                        "scenario": scenario_name,
                        "fold": fold_index,
                        "metric": metric_name,
                        "value": fold_value,

                        "llm_status": (
                            fold_audit.get(
                                "llm_status",
                                "unknown"
                            )
                            if use_llm
                            else "not_applicable"
                        ),

                        "llm_attempted": (
                            fold_audit.get(
                                "llm_attempted",
                                False
                            )
                            if use_llm
                            else False
                        ),

                        "llm_cache_hit": (
                            fold_audit.get(
                                "llm_cache_hit",
                                False
                            )
                            if use_llm
                            else False
                        ),

                        "llm_error": (
                            fold_audit.get("llm_error")
                            if use_llm
                            else None
                        ),

                        "llm_provider": (
                            fold_audit.get(
                                "llm_provider",
                                LLM_PROVIDER
                            )
                            if use_llm
                            else "not_applicable"
                        ),

                        "llm_model": (
                            fold_audit.get(
                                "llm_model",
                                LLM_MODEL
                            )
                            if use_llm
                            else None
                        )
                    })

    fold_columns = [
        "run_id",
        "timestamp",
        "dataset",
        "model",
        "scenario",
        "fold",
        "metric",
        "value",
        "llm_status",
        "llm_attempted",
        "llm_cache_hit",
        "llm_error",
        "llm_provider",
        "llm_model"
    ]

    folds_df = pd.DataFrame(
        fold_rows,
        columns=fold_columns
    )

    folds_file = (
        metrics_dir / f"{dataset_name}_folds.csv"
    )

    folds_df.to_csv(
        folds_file,
        index=False,
        encoding="utf-8-sig"
    )

    consolidated_folds_file = (
        metrics_dir / "all_datasets_folds.csv"
    )

    update_consolidated_csv(
        consolidated_folds_file,
        folds_df,
        dataset_name,
        LLM_PROVIDER
    )

    print("\nMetricas individuais dos folds exportadas:")
    print(folds_file)
    print(consolidated_folds_file)

    print(
        f"Total de registros de folds exportados: "
        f"{len(folds_df)}"
    )

    # -------------------------------------------------
    # 3. AUDITORIAS
    # -------------------------------------------------

    audit_rows = []

    for model_name, scenarios in all_audits.items():

        for scenario_name, folds in scenarios.items():

            use_llm = SCENARIOS[scenario_name]["use_llm"]

            for fold_index, fold in enumerate(
                folds,
                start=1
            ):

                fold_number = fold.get(
                    "fold",
                    fold_index
                )

                llm_status = (
                    fold.get("llm_status", "unknown")
                    if use_llm
                    else "not_applicable"
                )

                llm_provider = (
                    fold.get("llm_provider", LLM_PROVIDER)
                    if use_llm
                    else "not_applicable"
                )

                llm_model = (
                    fold.get("llm_model", LLM_MODEL)
                    if use_llm
                    else None
                )

                # Dados comuns de cada registro de auditoria.
                base_audit = {

                    "run_id": run_id,
                    "timestamp": run_timestamp,
                    "dataset": dataset_name,
                    "model": model_name,
                    "scenario": scenario_name,
                    "fold": fold_number,

                    "llm_status": llm_status,

                    "llm_attempted": (
                        fold.get("llm_attempted", False)
                        if use_llm
                        else False
                    ),

                    "llm_cache_hit": (
                        fold.get("llm_cache_hit", False)
                        if use_llm
                        else False
                    ),

                    "llm_error": (
                        fold.get("llm_error")
                        if use_llm
                        else None
                    ),

                    "llm_provider": llm_provider,
                    "llm_model": llm_model
                }

                # Uma linha de status por fold.
                audit_rows.append({

                    **base_audit,
                    "audit_type": "llm_status",
                    "feature": None,
                    "score": None

                })

                # -------------------------------------
                # FEATURES EFETIVAMENTE GERADAS
                # -------------------------------------

                for feature in fold.get(
                    "generated_features",
                    []
                ):

                    audit_rows.append({

                        **base_audit,
                        "audit_type": "generated",
                        "feature": feature,
                        "score": None

                    })

                # -------------------------------------
                # INTERACOES EFETIVAMENTE GERADAS
                # -------------------------------------

                for item in fold.get(
                    "interaction_opportunities",
                    []
                ):

                    if isinstance(item, str):

                        feature = item

                    elif isinstance(item, dict):

                        feature = item.get("feature")

                    else:

                        continue

                    if feature:

                        audit_rows.append({

                            **base_audit,
                            "audit_type": "interaction",
                            "feature": feature,
                            "score": None

                        })

                # -------------------------------------
                # CANDIDATAS A INTERACAO
                # -------------------------------------

                for item in fold.get(
                    "interaction_candidates",
                    []
                ):

                    if isinstance(item, str):

                        feature = item

                    elif isinstance(item, dict):

                        feature = item.get("feature")

                    else:

                        continue

                    if feature:

                        audit_rows.append({

                            **base_audit,
                            "audit_type": "interaction_candidate",
                            "feature": feature,
                            "score": None

                        })

                # -------------------------------------
                # SUGESTOES DA LLM
                # -------------------------------------

                if use_llm:

                    for suggestion in fold.get(
                        "llm_suggestions",
                        []
                    ):

                        if not isinstance(suggestion, dict):
                            continue

                        operation = suggestion.get("operation")

                        if operation == "log1p":

                            feature = (
                                f"{suggestion.get('column')}_log1p"
                            )

                        elif operation == "missing_indicator":

                            feature = (
                                f"{suggestion.get('column')}_missing"
                            )

                        elif operation == "interaction_product":

                            columns = suggestion.get(
                                "columns",
                                []
                            )

                            if len(columns) != 2:
                                continue

                            feature = (
                                f"{columns[0]}_x_{columns[1]}"
                            )

                        else:

                            continue

                        audit_rows.append({

                            **base_audit,
                            "audit_type": "llm_suggestion",
                            "feature": feature,
                            "score": None

                        })

                # -------------------------------------
                # FEATURES SELECIONADAS
                # -------------------------------------

                for feature in fold.get(
                    "selected_features",
                    []
                ):

                    audit_rows.append({

                        **base_audit,
                        "audit_type": "selected",
                        "feature": feature,
                        "score": None

                    })

                # -------------------------------------
                # PONTUACOES DAS FEATURES
                # -------------------------------------

                feature_scores = fold.get(
                    "feature_scores",
                    {}
                )

                if isinstance(feature_scores, dict):

                    for feature, score in feature_scores.items():

                        audit_rows.append({

                            **base_audit,
                            "audit_type": "feature_score",
                            "feature": feature,
                            "score": score

                        })

    audit_columns = [
        "run_id",
        "timestamp",
        "dataset",
        "model",
        "scenario",
        "fold",
        "audit_type",
        "feature",
        "score",
        "llm_status",
        "llm_attempted",
        "llm_cache_hit",
        "llm_error",
        "llm_provider",
        "llm_model"
    ]

    audits_df = pd.DataFrame(
        audit_rows,
        columns=audit_columns
    )

    audits_file = (
        audits_dir / f"{dataset_name}_audit.csv"
    )

    audits_df.to_csv(
        audits_file,
        index=False,
        encoding="utf-8-sig"
    )

    consolidated_audits_file = (
        audits_dir / "all_datasets_audits.csv"
    )

    update_consolidated_csv(
        consolidated_audits_file,
        audits_df,
        dataset_name,
        LLM_PROVIDER
    )

    print("\nAuditorias exportadas:")
    print(audits_file)
    print(consolidated_audits_file)

    print("\n" + "=" * 75)
    print("EXPORTACAO CONCLUIDA!")
    print("=" * 75)


# =====================================================
# CRIACAO DA VALIDACAO CRUZADA
# =====================================================

def create_cv_splits(X, y, config):

    if config["cv_type"] == "stratified":

        cv_generator = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE
        )

    elif config["cv_type"] == "kfold":

        cv_generator = KFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE
        )

    else:

        raise ValueError(
            f"Tipo de validacao desconhecido: "
            f"{config['cv_type']}"
        )

    return list(
        cv_generator.split(X, y)
    )


# =====================================================
# FUNCAO DE AVALIACAO
# =====================================================

def evaluate_experiment(
    df,
    experiment_name,
    config
):

    print("\n" + "=" * 65)
    print(f"EXPERIMENTO: {experiment_name}")
    print("=" * 65)

    target_column = config["target"]
    problem_type = config["problem_type"]
    drop_columns = config["drop_columns"]

    if target_column not in df.columns:

        raise ValueError(
            f"Coluna alvo '{target_column}' nao encontrada."
        )

    y = df[target_column].copy()

    X = df.drop(
        columns=[target_column]
    )

    X = X.drop(
        columns=drop_columns,
        errors="ignore"
    )

    if y.isna().any():

        raise ValueError(
            "A coluna alvo possui valores ausentes."
        )

    if X.empty:

        raise ValueError(
            "Nao existem atributos disponiveis para avaliacao."
        )

    print(f"Provedor LLM configurado: {LLM_PROVIDER}")

    print(
        f"Modelo LLM configurado: "
        f"{LLM_MODEL or 'padrao do cliente'}"
    )

    print(f"Registros: {len(X)}")
    print(f"Atributos utilizados: {len(X.columns)}")
    print(f"Target: {target_column}")
    print(f"Tipo de problema: {problem_type}")

    print("\n===== COLUNAS UTILIZADAS =====")
    print(X.columns.tolist())

    print("\n===== VALORES AUSENTES =====")
    print(X.isnull().sum())

    print("\n===== INFORMACOES DO TARGET =====")

    if problem_type == "classification":

        print(
            y.value_counts().sort_index()
        )

        class_names = config["class_names"]

        if class_names is not None:

            print("\n===== DISTRIBUICAO DAS CLASSES =====")

            print(
                y.map(class_names).value_counts()
            )

    else:

        print(y.describe())

    cv_splits = create_cv_splits(
        X,
        y,
        config
    )

    print("\n===== VALIDACAO CRUZADA =====")
    print(f"Quantidade de folds: {len(cv_splits)}")
    print(f"Metodo: {config['cv_type']}")
    print(
        "Divisoes compartilhadas entre todos os experimentos."
    )

    all_results = {}
    all_audits = {}

    for model_name in MODEL_NAMES:

        print("\n\n" + "#" * 65)
        print(f"MODELO: {model_name.upper()}")
        print("#" * 65)

        evaluator = ModelEvaluator(
            problem_type=problem_type,
            random_state=RANDOM_STATE,
            model_name=model_name,
            llm_provider=LLM_PROVIDER,
            llm_model=LLM_MODEL
        )

        all_results[model_name] = {}
        all_audits[model_name] = {}

        for scenario_name, configuration in SCENARIOS.items():

            print("\n" + "-" * 55)
            print(f"EXECUTANDO: {scenario_name}")
            print("-" * 55)

            print(
                f"AutoFE: {configuration['use_autofe']} | "
                f"LLM: {configuration['use_llm']} | "
                f"Interacoes: {configuration['include_interactions']} | "
                f"Selecao: {configuration['use_feature_selection']}"
            )

            results = evaluator.evaluate(
                X,
                y,
                cv=cv_splits,

                use_autofe=configuration["use_autofe"],

                use_feature_selection=configuration[
                    "use_feature_selection"
                ],

                selector_k=SELECTOR_K,

                include_interactions=configuration[
                    "include_interactions"
                ],

                use_llm=configuration["use_llm"],

                llm_provider=LLM_PROVIDER,
                llm_model=LLM_MODEL
            )

            all_results[model_name][scenario_name] = results

            all_audits[model_name][scenario_name] = list(
                evaluator.last_feature_audit_
            )

            for metric, values in results.items():

                print(
                    f"{metric.upper()}: "
                    f"{values['mean']:.4f} "
                    f"(+/- {values['std']:.4f})"
                )

            if configuration["use_llm"]:

                print_llm_status_summary(
                    all_audits[model_name][scenario_name],
                    scenario_name,
                    use_llm=True
                )

    # =================================================
    # COMPARACAO DOS RESULTADOS
    # =================================================

    print("\n\n" + "=" * 75)
    print("COMPARACAO GERAL DOS RESULTADOS")
    print("=" * 75)

    for model_name in MODEL_NAMES:

        model_results = all_results[model_name]

        print("\n" + "#" * 65)
        print(f"MODELO: {model_name.upper()}")
        print("#" * 65)

        baseline_results = model_results["BASELINE"]

        for metric in baseline_results:

            print("\n" + "-" * 55)
            print(f"METRICA: {metric.upper()}")
            print("-" * 55)

            baseline_mean = baseline_results[metric]["mean"]

            for scenario_name in SCENARIOS:

                values = model_results[scenario_name][metric]

                mean = values["mean"]
                std = values["std"]

                difference = mean - baseline_mean

                print(
                    f"{scenario_name}: "
                    f"{mean:.4f} (+/- {std:.4f}) "
                    f"| Diferenca: {difference:+.4f}"
                )

    # =================================================
    # AUDITORIA GERAL DAS FEATURES
    # =================================================

    print("\n\n" + "=" * 75)
    print("AUDITORIA GERAL DAS FEATURES")
    print("=" * 75)

    for model_name in MODEL_NAMES:

        print("\n\n" + "#" * 65)
        print(f"MODELO: {model_name.upper()}")
        print("#" * 65)

        for scenario_name, configuration in SCENARIOS.items():

            if not configuration["use_autofe"]:
                continue

            audit = all_audits[model_name][scenario_name]

            print_feature_audit(
                audit,
                f"{model_name.upper()} - {scenario_name}",
                use_llm=configuration["use_llm"]
            )

    # =================================================
    # RESUMO DOS MELHORES RESULTADOS
    # =================================================

    print("\n\n" + "=" * 75)
    print("MELHOR CENARIO POR MODELO")
    print("=" * 75)

    for model_name in MODEL_NAMES:

        print(f"\nModelo: {model_name.upper()}")

        model_results = all_results[model_name]

        for metric in model_results["BASELINE"]:

            lower_is_better = metric in ("mae", "rmse")

            eligible_scenarios = []

            for scenario_name in SCENARIOS:

                configuration = SCENARIOS[scenario_name]

                # Exclui do ranking cenarios com LLM incompletos.
                if configuration["use_llm"]:

                    audit = all_audits[model_name][scenario_name]

                    summary = summarize_llm_status(
                        audit,
                        use_llm=True
                    )

                    if summary["scenario_status"] != "complete":
                        continue

                eligible_scenarios.append(
                    scenario_name
                )

            if not eligible_scenarios:

                print(
                    f"{metric.upper()}: nenhum cenario elegivel."
                )

                continue

            if lower_is_better:

                best_scenario = min(
                    eligible_scenarios,
                    key=lambda name: (
                        model_results[name][metric]["mean"]
                    )
                )

            else:

                best_scenario = max(
                    eligible_scenarios,
                    key=lambda name: (
                        model_results[name][metric]["mean"]
                    )
                )

            best_value = (
                model_results[best_scenario][metric]["mean"]
            )

            print(
                f"{metric.upper()}: "
                f"{best_scenario} ({best_value:.4f})"
            )

            llm_scenario = "AUTOFE + LLM"

            if llm_scenario in SCENARIOS:

                llm_audit = all_audits[model_name][llm_scenario]

                llm_summary = summarize_llm_status(
                    llm_audit,
                    use_llm=True
                )

                if llm_summary["scenario_status"] != "complete":

                    print(
                        "  Observacao: AUTOFE + LLM foi excluido "
                        "do ranking porque a execucao ficou incompleta."
                    )

    # =================================================
    # VALIDACAO DA COMPLETUDE DOS CENARIOS
    # =================================================

    print("\n\n" + "=" * 75)
    print("VALIDACAO DA COMPLETUDE DOS EXPERIMENTOS")
    print("=" * 75)

    for model_name in MODEL_NAMES:

        for scenario_name, configuration in SCENARIOS.items():

            if not configuration["use_llm"]:

                print(
                    f"\n{model_name.upper()} - {scenario_name}: "
                    "NAO SE APLICA (cenario sem LLM)."
                )

                continue

            audit = all_audits[model_name][scenario_name]

            summary = summarize_llm_status(
                audit,
                use_llm=True
            )

            print(
                f"\n{model_name.upper()} - {scenario_name}"
            )

            print(
                f"Status: {summary['scenario_status'].upper()}"
            )

            print(
                f"Folds validos: "
                f"{summary['successful_folds']}/"
                f"{summary['total_folds']}"
            )

            if summary["scenario_status"] != "complete":

                print(
                    "Este resultado deve ser tratado como parcial "
                    "e nao deve ser utilizado para afirmar o "
                    "desempenho da integracao com LLM."
                )

    print("\n" + "=" * 75)
    print("FIM DO EXPERIMENTO")
    print("=" * 75)

    return all_results, all_audits


# =====================================================
# EXECUCAO PRINCIPAL
# =====================================================

def main():

    # Reinicia o circuit breaker no inicio
    # de uma nova execucao experimental.
    reset_llm_circuit()

    for dataset_name in DATASET_NAMES:

        df, config = load_selected_dataset(
            dataset_name
        )

        print(
            f"Dataset selecionado: {dataset_name}"
        )

        all_results, all_audits = evaluate_experiment(
            df,
            dataset_name.upper(),
            config
        )

        save_results_to_csv(
            dataset_name,
            all_results,
            all_audits
        )


if __name__ == "__main__":
    main()
