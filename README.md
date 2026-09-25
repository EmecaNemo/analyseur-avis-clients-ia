# Analyseur d'avis clients par IA

Outil Python qui utilise l'**API Claude** (Anthropic) pour classifier automatiquement des avis clients
par **sentiment** (positif / neutre / négatif) et par **thématique** (service, prix, qualité), puis
agrège les résultats avec **pandas** pour produire un rapport qui aide les commerçants à
**prioriser leurs actions**.

## Fonctionnement

```
data/avis.csv ──► analyser.py ──► output/avis_classes.csv ──► rapport.py ──► output/rapport.md
                 (API Claude)                                 (pandas +       + graphiques PNG
                                                               API Claude)
                                      app.py : dashboard Streamlit interactif
```

1. **`analyser.py`** envoie chaque avis à Claude. La réponse est contrainte par un schéma
   Pydantic (*structured outputs*) : le modèle renvoie toujours un JSON valide
   `{sentiment, themes, resume}`, sans parsing fragile. Les requêtes sont parallélisées.
2. **`rapport.py`** calcule avec pandas :
   - la répartition des sentiments par commerce ;
   - le nombre d'avis négatifs par thématique ;
   - pour chaque commerce, la thématique à **améliorer en priorité** (celle qui concentre le plus
     d'avis négatifs), avec des exemples ;
   - **3 recommandations concrètes par commerce**, rédigées par Claude à partir des avis négatifs.
3. **`app.py`** : dashboard **Streamlit** interactif (indicateurs clés, graphiques, filtres par
   commerce et par sentiment, export CSV). On peut explorer la démo ou importer ses propres avis
   (avec sa propre clé API).

## Installation

```bash
git clone https://github.com/EmecaNemo/analyseur-avis-clients-ia.git
cd analyseur-avis-clients-ia
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."   # clé disponible sur console.anthropic.com
```

## Utilisation

```bash
python analyser.py      # classification des avis de data/avis.csv
python rapport.py       # génération de output/rapport.md (+ recommandations IA)
streamlit run app.py    # dashboard interactif
```

Pour analyser vos propres avis : `python analyser.py --input mon_fichier.csv`
(colonnes requises : `id`, `commerce`, `avis`). `python rapport.py --sans-ia` génère le rapport
sans appel à l'API.

## Résultats

Le rapport complet est disponible dans [`output/rapport.md`](output/rapport.md).

**Contrôle de cohérence :** le sentiment détecté par le modèle (`claude-haiku-4-5`) correspond à la
note laissée par le client (1-2 ★ = négatif, 3 ★ = neutre, 4-5 ★ = positif) pour **56 avis sur 60 (93 %)**.
Les écarts concernent surtout des avis 3 ★ au ton critique, classés négatifs.

![Sentiments](output/sentiments.png)
![Insatisfactions](output/insatisfactions.png)

## Données

`data/avis.csv` contient 60 avis **fictifs** rédigés pour la démonstration (trois commerces
parisiens imaginaires : une boulangerie, un restaurant et une épicerie bio).

## Stack technique

Python 3.10+ · API Claude (SDK `anthropic`, structured outputs) · pandas · Pydantic · matplotlib · Streamlit
