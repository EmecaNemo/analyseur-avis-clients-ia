"""Classifie des avis clients par sentiment et par thématique avec l'API Claude.

Usage :
    python analyser.py                       # data/avis.csv -> output/avis_classes.csv
    python analyser.py --input mes_avis.csv  # autre fichier (colonnes requises : id, avis)
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, List, Literal

import anthropic
import pandas as pd
from pydantic import BaseModel, Field

MODEL = "claude-haiku-4-5"

SYSTEM_PROMPT = """Tu analyses des avis clients laissés sur des commerces de proximité.
Pour chaque avis, détermine :
- le sentiment global (positif, neutre ou négatif) ;
- les thématiques abordées parmi : service (accueil, attente, amabilité, gestion des erreurs),
  prix (tarifs, rapport qualité-prix, facturation), qualité (produits, cuisine, fraîcheur, hygiène).
  Un avis peut aborder plusieurs thématiques ; indique-les toutes.
- un résumé très court (10 mots maximum) du point principal."""


class Classification(BaseModel):
    sentiment: Literal["positif", "neutre", "négatif"]
    themes: List[Literal["service", "prix", "qualité"]] = Field(
        description="Thématiques abordées dans l'avis"
    )
    resume: str = Field(description="Point principal de l'avis, 10 mots maximum")


def classify_with_usage(
    review: str, client: anthropic.Anthropic, model: str = MODEL
) -> tuple[Classification | None, anthropic.types.Usage]:
    """Envoie un avis à Claude ; renvoie la classification validée et la consommation de tokens."""
    response = client.messages.parse(
        model=model,
        max_tokens=4096,  # marge pour les modèles qui réfléchissent avant de répondre
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": review}],
        output_format=Classification,
    )
    if response.stop_reason in ("refusal", "max_tokens"):
        return None, response.usage
    return response.parsed_output, response.usage


def classify(review: str, client: anthropic.Anthropic) -> Classification | None:
    """Envoie un avis à Claude et renvoie sa classification validée."""
    return classify_with_usage(review, client)[0]


def classify_dataframe(
    df: pd.DataFrame,
    client: anthropic.Anthropic,
    workers: int = 5,
    on_progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Classifie la colonne `avis` et ajoute les colonnes sentiment, themes et resume."""

    def safe_classify(review_id, text):
        try:
            return classify(text, client)
        except anthropic.APIError as e:
            print(f"  [avis {review_id}] erreur API : {e}")
            return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(safe_classify, i, t) for i, t in zip(df["id"], df["avis"])]
        # La progression est signalée depuis le thread principal (requis par Streamlit)
        for done, _ in enumerate(as_completed(futures), start=1):
            if on_progress:
                on_progress(done, len(df))
        results = [f.result() for f in futures]

    df = df.copy()
    df["sentiment"] = [r.sentiment if r else None for r in results]
    df["themes"] = ["|".join(r.themes) if r else None for r in results]
    df["resume"] = [r.resume if r else None for r in results]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/avis.csv")
    parser.add_argument("--output", default="output/avis_classes.csv")
    parser.add_argument("--workers", type=int, default=5, help="requêtes en parallèle")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"{len(df)} avis à classifier avec {MODEL}...")
    df = classify_dataframe(
        df,
        anthropic.Anthropic(),
        workers=args.workers,
        on_progress=lambda done, total: print(f"  {done}/{total}", end="\r"),
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    failed = df["sentiment"].isna().sum()
    print(f"Terminé : {args.output} ({failed} échec(s))")


if __name__ == "__main__":
    main()
