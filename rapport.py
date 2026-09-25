"""Agrège les avis classifiés et génère un rapport synthétique par commerce.

Usage :
    python rapport.py   # output/avis_classes.csv -> output/rapport.md + graphiques
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SENTIMENTS = ["positif", "neutre", "négatif"]
COLORS = {"positif": "#2e9d5b", "neutre": "#a0a0a0", "négatif": "#d1453b"}


def load(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Renvoie les avis classifiés et une table longue (une ligne par avis x thème)."""
    df = pd.read_csv(path).dropna(subset=["sentiment"])
    themes = df.assign(theme=df["themes"].fillna("").str.split("|")).explode("theme", ignore_index=True)
    themes = themes[themes["theme"] != ""]
    return df, themes


def sentiment_table(df: pd.DataFrame) -> pd.DataFrame:
    """Répartition des sentiments (%) par commerce."""
    table = pd.crosstab(df["commerce"], df["sentiment"], normalize="index") * 100
    return table.reindex(columns=SENTIMENTS, fill_value=0).round(1)


def complaints_by_theme(themes: pd.DataFrame) -> pd.DataFrame:
    """Nombre d'avis négatifs par thème et par commerce."""
    negative = themes[themes["sentiment"] == "négatif"]
    return pd.crosstab(negative["commerce"], negative["theme"])


def priorities(themes: pd.DataFrame) -> pd.DataFrame:
    """Pour chaque commerce, le thème avec le plus d'avis négatifs = action prioritaire."""
    stats = (
        themes.groupby(["commerce", "theme"])["sentiment"]
        .agg(mentions="size", negatifs=lambda s: (s == "négatif").sum())
        .reset_index()
    )
    stats["taux_negatif"] = (stats["negatifs"] / stats["mentions"] * 100).round(0)
    stats = stats.sort_values(["commerce", "negatifs", "taux_negatif"], ascending=[True, False, False])
    return stats.groupby("commerce").head(1).set_index("commerce")


def plot_sentiments(table: pd.DataFrame, out: Path) -> None:
    ax = table.plot.barh(stacked=True, color=[COLORS[s] for s in SENTIMENTS], figsize=(8, 3.5))
    ax.set_xlabel("% des avis")
    ax.set_ylabel("")
    ax.set_title("Répartition des sentiments par commerce")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_complaints(table: pd.DataFrame, out: Path) -> None:
    ax = table.plot.bar(figsize=(8, 3.5), color=["#4c72b0", "#dd8452", "#8172b3"])
    ax.set_ylabel("Nombre d'avis négatifs")
    ax.set_xlabel("")
    ax.set_title("Sources d'insatisfaction par thème")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def build_report(df: pd.DataFrame, themes: pd.DataFrame, out_dir: Path) -> str:
    sentiments = sentiment_table(df)
    complaints = complaints_by_theme(themes)
    prio = priorities(themes)

    plot_sentiments(sentiments, out_dir / "sentiments.png")
    plot_complaints(complaints, out_dir / "insatisfactions.png")

    lines = [
        "# Rapport d'analyse des avis clients",
        "",
        f"**{len(df)} avis analysés** sur {df['commerce'].nunique()} commerces.",
        "",
        "## Répartition des sentiments (%)",
        "",
        sentiments.to_markdown(),
        "",
        "![Sentiments](sentiments.png)",
        "",
        "## Avis négatifs par thématique",
        "",
        complaints.to_markdown(),
        "",
        "![Insatisfactions](insatisfactions.png)",
        "",
        "## Actions prioritaires",
        "",
    ]
    for commerce, row in prio.iterrows():
        examples = df[
            (df["commerce"] == commerce)
            & (df["sentiment"] == "négatif")
            & df["themes"].str.contains(row["theme"], regex=False)
        ]["resume"].head(3)
        lines.append(
            f"### {commerce} → améliorer **{row['theme']}**\n\n"
            f"{int(row['negatifs'])} avis négatifs sur {int(row['mentions'])} mentions "
            f"({int(row['taux_negatif'])} %). Exemples :\n"
        )
        lines += [f"- {e}" for e in examples]
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="output/avis_classes.csv")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df, themes = load(args.input)
    report = build_report(df, themes, out_dir)
    (out_dir / "rapport.md").write_text(report, encoding="utf-8")
    print(f"Rapport généré : {out_dir / 'rapport.md'}")


if __name__ == "__main__":
    main()
