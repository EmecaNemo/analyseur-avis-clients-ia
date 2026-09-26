"""Compare plusieurs modèles Claude sur des avis annotés manuellement (précision, coût, latence).

Usage :
    python evaluer.py                                   # Haiku 4.5, Sonnet 5 et Opus 5
    python evaluer.py --models claude-haiku-4-5         # un seul modèle
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import pandas as pd

from analyser import classify_with_usage

# Prix en $ par million de tokens (entrée, sortie)
PRICES = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-5": (5.00, 25.00),
}
SENTIMENTS = ["positif", "neutre", "négatif"]


def run_model(df: pd.DataFrame, client: anthropic.Anthropic, model: str, workers: int) -> pd.DataFrame:
    """Classifie tous les avis avec `model` ; renvoie prédictions, tokens et latence par avis."""

    def one(text):
        start = time.perf_counter()
        try:
            result, usage = classify_with_usage(text, client, model)
            tokens = (usage.input_tokens, usage.output_tokens)
        except anthropic.APIError as e:
            print(f"  erreur API : {e}")
            result, tokens = None, (0, 0)
        return result, tokens, time.perf_counter() - start

    with ThreadPoolExecutor(max_workers=workers) as pool:
        outputs = list(pool.map(one, df["avis"]))

    return pd.DataFrame(
        {
            "id": df["id"],
            "model": model,
            "sentiment": [r.sentiment if r else None for r, _, _ in outputs],
            "themes": ["|".join(sorted(r.themes)) if r else None for r, _, _ in outputs],
            "input_tokens": [t[0] for _, t, _ in outputs],
            "output_tokens": [t[1] for _, t, _ in outputs],
            "latency_s": [lat for _, _, lat in outputs],
        }
    )


def theme_scores(pred: pd.Series, ref: pd.Series) -> tuple[float, float, float, float]:
    """Correspondance exacte, précision, rappel et F1 (micro) sur les couples (avis, thème)."""
    pred_sets = [set(p.split("|")) if isinstance(p, str) else set() for p in pred]
    ref_sets = [set(r.split("|")) for r in ref]
    tp = sum(len(p & r) for p, r in zip(pred_sets, ref_sets))
    n_pred = sum(len(p) for p in pred_sets)
    n_ref = sum(len(r) for r in ref_sets)
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_ref if n_ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact = sum(p == r for p, r in zip(pred_sets, ref_sets)) / len(ref_sets)
    return exact, precision, recall, f1


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, g in results.groupby("model", sort=False):
        exact, precision, recall, f1 = theme_scores(g["themes"], g["themes_ref"])
        price_in, price_out = PRICES.get(model, (float("nan"), float("nan")))
        cost = (g["input_tokens"].sum() * price_in + g["output_tokens"].sum() * price_out) / 1e6
        rows.append(
            {
                "modèle": model,
                "sentiment (exactitude)": f"{(g['sentiment'] == g['sentiment_ref']).mean():.0%}",
                "thèmes (correspondance exacte)": f"{exact:.0%}",
                "thèmes (F1)": f"{f1:.2f}",
                "coût pour 1 000 avis": f"{cost / len(g) * 1000:.2f} $",
                "latence moyenne": f"{g['latency_s'].mean():.1f} s",
                "échecs": int(g["sentiment"].isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def build_report(results: pd.DataFrame, reviews: pd.DataFrame) -> str:
    lines = [
        "# Évaluation des modèles",
        "",
        f"{results['id'].nunique()} avis comparés à des annotations manuelles "
        "([`data/annotations.csv`](../data/annotations.csv)).",
        "",
        summarize(results).to_markdown(index=False),
        "",
    ]
    for model, g in results.groupby("model", sort=False):
        confusion = pd.crosstab(g["sentiment_ref"], g["sentiment"].fillna("échec"))
        columns = [c for c in SENTIMENTS + ["échec"] if c in confusion.columns]
        confusion = confusion.reindex(index=SENTIMENTS, columns=columns, fill_value=0)
        confusion.index.name, confusion.columns.name = "référence ↓ / prédit →", None
        errors = g[g["sentiment"] != g["sentiment_ref"]].merge(reviews[["id", "avis"]], on="id")
        lines += [f"## {model}", "", "Matrice de confusion (sentiment) :", "", confusion.to_markdown(), ""]
        if not errors.empty:
            lines += ["Erreurs de sentiment :", ""]
            lines += [
                f"- « {row.avis} » → prédit **{row.sentiment}**, attendu **{row.sentiment_ref}**"
                for row in errors.itertuples()
            ]
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/avis.csv")
    parser.add_argument("--annotations", default="data/annotations.csv")
    parser.add_argument("--models", default=",".join(PRICES))
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    reviews = pd.read_csv(args.input)
    annotations = pd.read_csv(args.annotations)
    client = anthropic.Anthropic()

    runs = []
    for model in args.models.split(","):
        print(f"Évaluation de {model}...")
        runs.append(run_model(reviews, client, model, args.workers))
    results = pd.concat(runs).merge(annotations, on="id")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "evaluation_details.csv", index=False)
    (out_dir / "evaluation.md").write_text(build_report(results, reviews), encoding="utf-8")
    print(summarize(results).to_string(index=False))
    print(f"Rapport : {out_dir / 'evaluation.md'}")


if __name__ == "__main__":
    main()
