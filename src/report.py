"""
Gera o relatorio final dos experimentos a partir dos CSVs em results/.

Uso (na raiz do projeto):
    python -m src.report

Saidas em results/report/:
    RESULTADOS.md          relatorio com metricas, features e sugestoes da LLM
    fig_delta_baseline.png variacao de cada cenario em relacao ao baseline
    fig_features_selecionadas.png  features escolhidas pelo seletor
    fig_sugestoes_llm.png  sugestoes da LLM por frequencia
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

METRICS_FILE = Path("results/metrics/all_datasets_metrics.csv")
FOLDS_FILE = Path("results/metrics/all_datasets_folds.csv")
AUDITS_FILE = Path("results/audits/all_datasets_audits.csv")
OUT_DIR = Path("results/report")

SCENARIO_ORDER = [
    "BASELINE",
    "AUTOFE TRADICIONAL",
    "AUTOFE + LLM",
    "AUTOFE COM INTERACOES",
    "AUTOFE + INTERACOES + SELECAO",
]

# Metrica principal e metricas de apoio por dataset.
DATASETS = {
    "titanic": {
        "title": "Titanic (classificacao)",
        "primary": "accuracy",
        "extra": ["f1", "f1_positive_class"],
        "higher_is_better": True,
    },
    "breast_cancer": {
        "title": "Breast Cancer (classificacao)",
        "primary": "accuracy",
        "extra": ["f1", "recall_class_0"],
        "higher_is_better": True,
    },
    "diabetes": {
        "title": "Diabetes (regressao)",
        "primary": "r2",
        "extra": ["mae", "rmse"],
        "higher_is_better": True,
    },
}


# -----------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------

def normalize_feature(name):
    """Alinha o nome das sugestoes da LLM ao nome real da feature."""
    if isinstance(name, str) and name.endswith("_missing") and not name.endswith("_is_missing"):
        return name[: -len("_missing")] + "_is_missing"
    return name


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def load_data():
    metrics = pd.read_csv(METRICS_FILE)
    folds = pd.read_csv(FOLDS_FILE)
    audits = pd.read_csv(AUDITS_FILE)
    audits["feature"] = audits["feature"].map(normalize_feature)
    return metrics, folds, audits


def scenario_is_complete(metrics, dataset, scenario):
    rows = metrics[(metrics.dataset == dataset) & (metrics.scenario == scenario)]
    if rows.empty:
        return None
    status = rows["scenario_status"].iloc[0]
    return status != "partial_or_incomplete"


# -----------------------------------------------------------------
# 1. Metricas
# -----------------------------------------------------------------

def build_metric_table(metrics, dataset, cfg):
    names = [cfg["primary"]] + cfg["extra"]
    sub = metrics[(metrics.dataset == dataset) & (metrics.metric.isin(names))]
    rows = []
    base = sub[(sub.scenario == "BASELINE") & (sub.metric == cfg["primary"])]["mean"]
    base = float(base.iloc[0]) if not base.empty else None

    for scenario in SCENARIO_ORDER:
        s = sub[sub.scenario == scenario]
        if s.empty:
            continue
        row = {"Cenario": scenario}
        for m in names:
            v = s[s.metric == m]
            row[m] = f"{v['mean'].iloc[0]:.4f} ± {v['std'].iloc[0]:.4f}" if not v.empty else "-"
        p = s[s.metric == cfg["primary"]]["mean"]
        if base is not None and not p.empty:
            row[f"Δ {cfg['primary']}"] = f"{float(p.iloc[0]) - base:+.4f}"
        complete = scenario_is_complete(metrics, dataset, scenario)
        if complete is False:
            row["Cenario"] += " ⚠ INCOMPLETO"
        rows.append(row)
    return pd.DataFrame(rows)


def build_paired_table(folds, dataset, cfg):
    """Compara cada cenario com o baseline fold a fold (mesmos splits)."""
    d = folds[(folds.dataset == dataset) & (folds.metric == cfg["primary"])]
    piv = d.pivot_table(index="fold", columns="scenario", values="value")
    if "BASELINE" not in piv:
        return pd.DataFrame()
    rows = []
    for s in SCENARIO_ORDER:
        if s == "BASELINE" or s not in piv:
            continue
        diff = piv[s] - piv["BASELINE"]
        rows.append({
            "Cenario": s,
            "Folds melhores": f"{int((diff > 0).sum())}/{len(diff)}",
            "Empates": int((diff == 0).sum()),
            "Folds piores": f"{int((diff < 0).sum())}/{len(diff)}",
            "Δ medio": f"{diff.mean():+.4f}",
        })
    return pd.DataFrame(rows)


def plot_delta(metrics, folds):
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(15, 4.6))
    for ax, (ds, cfg) in zip(axes, DATASETS.items()):
        d = folds[(folds.dataset == ds) & (folds.metric == cfg["primary"])]
        piv = d.pivot_table(index="fold", columns="scenario", values="value")
        if "BASELINE" not in piv:
            continue
        labels, means, errs, hatches = [], [], [], []
        for s in SCENARIO_ORDER[1:]:
            if s not in piv:
                continue
            diff = piv[s] - piv["BASELINE"]
            labels.append(s.replace("AUTOFE ", "").replace(" + ", "+\n").replace(" COM ", "\n"))
            means.append(diff.mean())
            errs.append(diff.std())
            hatches.append(scenario_is_complete(metrics, ds, s) is False)
        colors = ["#2a9d8f" if m > 0 else "#e76f51" for m in means]
        bars = ax.bar(range(len(means)), means, yerr=errs, color=colors,
                      capsize=4, edgecolor="black", linewidth=0.6)
        for bar, partial in zip(bars, hatches):
            if partial:
                bar.set_hatch("//")
                bar.set_alpha(0.45)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_title(f"{cfg['title']}\nΔ {cfg['primary']} vs baseline", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Variacao media por fold em relacao ao baseline "
                 "(barra de erro = desvio entre folds; hachurado = cenario LLM incompleto)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_delta_baseline.png", dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------
# 2. Features
# -----------------------------------------------------------------

def feature_frequency(audits, dataset, scenario, audit_type):
    d = audits[(audits.dataset == dataset) & (audits.scenario == scenario)
               & (audits.audit_type == audit_type)]
    if d.empty:
        return pd.Series(dtype=int)
    return d.groupby("feature")["fold"].nunique().sort_values(ascending=False)


def build_generated_summary(audits, dataset):
    rows = []
    for scenario in SCENARIO_ORDER[1:]:
        gen = feature_frequency(audits, dataset, scenario, "generated")
        row = {
            "Cenario": scenario,
            "Features distintas geradas": len(gen),
            "Presentes nos 5 folds": int((gen == 5).sum()),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_selected_table(audits, dataset):
    sel = feature_frequency(audits, dataset, "AUTOFE + INTERACOES + SELECAO", "selected")
    if sel.empty:
        return pd.DataFrame()
    scores = audits[(audits.dataset == dataset)
                    & (audits.scenario == "AUTOFE + INTERACOES + SELECAO")
                    & (audits.audit_type == "feature_score")].groupby("feature")["score"].mean()
    out = pd.DataFrame({
        "Feature": sel.index,
        "Folds em que foi selecionada": [f"{v}/5" for v in sel.values],
        "Score medio": [f"{scores.get(f, float('nan')):.4f}" for f in sel.index],
    })
    return out


def plot_selected(audits):
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(15, 4.4))
    for ax, (ds, cfg) in zip(axes, DATASETS.items()):
        sel = feature_frequency(audits, ds, "AUTOFE + INTERACOES + SELECAO", "selected")
        ax.set_title(cfg["title"], fontsize=10)
        if sel.empty:
            ax.axis("off")
            continue
        sel = sel.sort_values()
        ax.barh(sel.index, sel.values, color="#264653")
        ax.set_xlim(0, 5)
        ax.set_xlabel("folds em que foi selecionada (de 5)")
        ax.tick_params(axis="y", labelsize=7.5)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Features escolhidas pelo seletor (cenario com interacoes + selecao)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_features_selecionadas.png", dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------
# 3. Sugestoes da LLM
# -----------------------------------------------------------------

def build_llm_table(audits, dataset):
    sug = feature_frequency(audits, dataset, "AUTOFE + LLM", "llm_suggestion")
    if sug.empty:
        return pd.DataFrame()
    trad = set(audits[(audits.dataset == dataset)
                      & (audits.scenario == "AUTOFE TRADICIONAL")
                      & (audits.audit_type == "generated")]["feature"])
    gen_llm = set(audits[(audits.dataset == dataset)
                         & (audits.scenario == "AUTOFE + LLM")
                         & (audits.audit_type == "generated")]["feature"])
    rows = []
    for feature, freq in sug.items():
        if "_x_" in feature:
            tipo = "interaction_product"
        elif feature.endswith("_is_missing"):
            tipo = "missing_indicator"
        else:
            tipo = "log1p"
        rows.append({
            "Feature sugerida": feature,
            "Operacao": tipo,
            "Folds": f"{freq}/5",
            "Ja gerada pelo AutoFE tradicional": "sim" if feature in trad else "nao",
            "Gerada no cenario LLM": "sim" if feature in gen_llm else "nao",
        })
    return pd.DataFrame(rows)


def llm_run_status(metrics, dataset):
    r = metrics[(metrics.dataset == dataset) & (metrics.scenario == "AUTOFE + LLM")]
    if r.empty:
        return None
    r = r.iloc[0]
    return {
        "total": int(r["llm_total_folds"]),
        "ok": int(r["llm_successful_folds"]),
        "quota": int(r["llm_quota_exhausted_folds"]),
        "circuit": int(r["llm_circuit_open_folds"]),
        "failed": int(r["llm_failed_folds"]),
        "provider": r["llm_provider"] if pd.notna(r["llm_provider"]) else "gemini (execucao anterior, sem metadado)",
        "model": r["llm_model"] if pd.notna(r["llm_model"]) else "-",
    }


def plot_llm(audits):
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(15, 4.4))
    for ax, (ds, cfg) in zip(axes, DATASETS.items()):
        sug = feature_frequency(audits, ds, "AUTOFE + LLM", "llm_suggestion")
        ax.set_title(cfg["title"], fontsize=10)
        if sug.empty:
            ax.axis("off")
            continue
        sug = sug.sort_values()
        ax.barh(sug.index, sug.values, color="#e9c46a", edgecolor="black", linewidth=0.5)
        ax.set_xlim(0, 5)
        ax.set_xlabel("folds em que foi sugerida (de 5)")
        ax.tick_params(axis="y", labelsize=7.5)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Sugestoes validas da LLM (apos filtro de operacoes permitidas)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_sugestoes_llm.png", dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------
# Montagem do relatorio
# -----------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics, folds, audits = load_data()

    plot_delta(metrics, folds)
    plot_selected(audits)
    plot_llm(audits)

    md = ["# Resultados dos experimentos AutoFE\n"]
    md.append(
        "Relatorio gerado automaticamente por `python -m src.report` a partir de "
        "`results/metrics/` e `results/audits/`. Validacao cruzada com 5 folds "
        "compartilhados entre os cenarios. Valores: media ± desvio entre folds.\n"
    )

    # Quadro de ressalvas
    md.append("## Leitura obrigatoria antes dos numeros\n")
    for ds in DATASETS:
        st = llm_run_status(metrics, ds)
        mod = metrics[metrics.dataset == ds]["model"].iloc[0]
        md.append(f"- **{ds}**: modelo `{mod}`.")
        if st and st["ok"] < st["total"]:
            md.append(
                f"  - ⚠ Cenario `AUTOFE + LLM` **incompleto**: {st['ok']}/{st['total']} folds "
                f"com resposta da LLM ({st['quota']} por cota esgotada, "
                f"{st['circuit']} bloqueados pelo circuit breaker). "
                "Esse resultado nao mede a contribuicao da LLM."
            )
        elif st:
            md.append(f"  - LLM: {st['provider']} / `{st['model']}` — {st['ok']}/{st['total']} folds validos.")
    md.append("")

    # 1. Metricas
    md.append("## 1. Metricas por dataset\n")
    for ds, cfg in DATASETS.items():
        md.append(f"### {cfg['title']}\n")
        md.append(f"Metrica principal: `{cfg['primary']}`.\n")
        md.append(md_table(build_metric_table(metrics, ds, cfg)))
        md.append("\n**Comparacao pareada com o baseline (mesmos folds):**\n")
        md.append(md_table(build_paired_table(folds, ds, cfg)))
        md.append("")
    md.append("![Delta vs baseline](fig_delta_baseline.png)\n")

    # 2. Features
    md.append("## 2. Features geradas e selecionadas\n")
    for ds, cfg in DATASETS.items():
        md.append(f"### {cfg['title']}\n")
        md.append(md_table(build_generated_summary(audits, ds)))
        sel = build_selected_table(audits, ds)
        if not sel.empty:
            md.append("\n**Features selecionadas** (cenario com interacoes + selecao):\n")
            md.append(md_table(sel))
        md.append("")
    md.append("![Features selecionadas](fig_features_selecionadas.png)\n")

    # 3. LLM
    md.append("## 3. Sugestoes da LLM\n")
    for ds, cfg in DATASETS.items():
        md.append(f"### {cfg['title']}\n")
        t = build_llm_table(audits, ds)
        if t.empty:
            md.append("Sem sugestoes registradas.\n")
            continue
        st = llm_run_status(metrics, ds)
        if st and st["ok"] < st["total"]:
            md.append(f"⚠ Apenas {st['ok']}/{st['total']} folds receberam resposta da LLM.\n")
        md.append(md_table(t))
        md.append("")
    md.append("![Sugestoes da LLM](fig_sugestoes_llm.png)\n")

    out = OUT_DIR / "RESULTADOS.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"Relatorio gerado em {out}")


if __name__ == "__main__":
    main()
