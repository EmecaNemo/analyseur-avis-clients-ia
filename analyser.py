"""Classifie des avis clients par sentiment et par thématique avec l'API Claude.

Usage :
    python analyser.py                       # data/avis.csv -> output/avis_classes.csv
    python analyser.py --input mes_avis.csv  # autre fichier (colonnes requises : id, avis)
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Literal

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


client = anthropic.Anthropic()


def classify(review: str) -> Classification | None:
    """Envoie un avis à Claude et renvoie sa classification validée."""
    response = client.messages.parse(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": review}],
        output_format=Classification,
    )
    if response.stop_reason == "refusal":
        return None
    return response.parsed_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/avis.csv")
    parser.add_argument("--output", default="output/avis_classes.csv")
    parser.add_argument("--workers", type=int, default=5, help="requêtes en parallèle")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"{len(df)} avis à classifier avec {MODEL}...")

    def safe_classify(item):
        review_id, text = item
        try:
            result = classify(text)
        except anthropic.APIError as e:
            print(f"  [avis {review_id}] erreur API : {e}")
            return None
        print(f"  [avis {review_id}] {result.sentiment if result else 'refusé'}")
        return result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(safe_classify, zip(df["id"], df["avis"])))

    df["sentiment"] = [r.sentiment if r else None for r in results]
    df["themes"] = ["|".join(r.themes) if r else None for r in results]
    df["resume"] = [r.resume if r else None for r in results]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    failed = sum(r is None for r in results)
    print(f"Terminé : {args.output} ({failed} échec(s))")


if __name__ == "__main__":
    main()
