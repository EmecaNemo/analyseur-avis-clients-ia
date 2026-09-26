"""Agrège les avis classifiés et génère un rapport synthétique par commerce.

Usage :
    python rapport.py           # output/avis_classes.csv -> output/rapport.md + graphiques
    python rapport.py --sans-ia # sans les recommandations générées par Claude
"""

import argparse
import json
from pathlib import Path

import anthropic
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analyser import MODEL

SENTIMENTS = ["positif", "neutre", "négatif"]
COLORS = {"positif": "#2e9d5b", "neutre": "#a0a0a0", "négatif": "#d1453b"}

RECO_PROMPT = """Tu conseilles le gérant du commerce « {commerce} ».
Voici la synthèse de ses avis clients :

Répartition des sentiments : {sentiments}
Avis négatifs par thématique : {complaints}

Avis négatifs :
{reviews}

Propose 3 actions concrètes et réalistes, classées par priorité, pour réduire l'insatisfaction.
Réponds uniquement par une liste Markdown de 3 points, une phrase courte par point, en français."""


def explode_themes(df: pd.DataFrame) -> pd.DataFrame:
    """Table longue : une ligne par couple (avis, thème)."""
    themes = df.assign(theme=df["themes"].fillna("").str.split("|")).explode("theme", ignore_index=True)
    return themes[themes["theme"] != ""]


def load(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Renvoie les avis classifiés et leur table longue par thème."""
    df = pd.read_csv(path).dropna(subset=["sentiment"])
    return df, explode_themes(df)


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


def recommend(df: pd.DataFrame, themes: pd.DataFrame, client: anthropic.Anthropic) -> dict[str, str]:
    """Demande à Claude 3 actions prioritaires par commerce, à partir des avis négatifs."""
    sentiments = sentiment_table(df)
    complaints = complaints_by_theme(themes)
    recommendations = {}
    for commerce in sorted(df["commerce"].unique()):
        negative = df[(df["commerce"] == commerce) & (df["sentiment"] == "négatif")]
        if negative.empty:
            recommendations[commerce] = "- Aucun avis négatif : continuer ainsi."
            continue
        prompt = RECO_PROMPT.format(
            commerce=commerce,
            sentiments=sentiments.loc[commerce].to_dict(),
            complaints=complaints.loc[commerce].to_dict() if commerce in complaints.index else {},
            reviews="\n".join(f"- {a}" for a in negative["avis"]),
        )
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = next((b.text for b in response.content if b.type == "text"), "")
        recommendations[commerce] = strip_headings(text)
    return recommendations


def strip_headings(text: str) -> str:
    """Retire les titres Markdown que le modèle ajoute parfois avant la liste."""
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#")).strip()


def plot_sentiments(table: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    table.plot.barh(ax=ax, stacked=True, color=[COLORS[s] for s in SENTIMENTS])
    ax.set_xlabel("% des avis")
    ax.set_ylabel("")
    ax.set_title("Répartition des sentiments par commerce")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)
    fig.tight_layout()
    return fig


def plot_complaints(table: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    table.plot.bar(ax=ax, color=["#4c72b0", "#dd8452", "#8172b3"])
    ax.set_ylabel("Nombre d'avis négatifs")
    ax.set_xlabel("")
    ax.set_title("Sources d'insatisfaction par thème")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    return fig


def save(fig: plt.Figure, out: Path) -> None:
    fig.savefig(out, dpi=120)
    plt.close(fig)


def build_report(
    df: pd.DataFrame, themes: pd.DataFrame, recommendations: dict[str, str] | None = None
) -> str:
    sentiments = sentiment_table(df)
    complaints = complaints_by_theme(themes)
    prio = priorities(themes)

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
        if recommendations and commerce in recommendations:
            lines += ["**Recommandations (générées par Claude) :**", "", recommendations[commerce], ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="output/avis_classes.csv")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--sans-ia", action="store_true", help="ne pas générer de recommandations")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df, themes = load(args.input)

    save(plot_sentiments(sentiment_table(df)), out_dir / "sentiments.png")
    save(plot_complaints(complaints_by_theme(themes)), out_dir / "insatisfactions.png")

    recommendations = None
    if not args.sans_ia:
        print(f"Génération des recommandations avec {MODEL}...")
        recommendations = recommend(df, themes, anthropic.Anthropic())
        (out_dir / "recommandations.json").write_text(
            json.dumps(recommendations, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    (out_dir / "rapport.md").write_text(build_report(df, themes, recommendations), encoding="utf-8")
    print(f"Rapport généré : {out_dir / 'rapport.md'}")


if __name__ == "__main__":
    main()
