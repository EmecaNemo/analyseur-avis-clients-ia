"""Dashboard Streamlit : analyse d'avis clients par IA.

Usage :
    streamlit run app.py
"""

import json
from pathlib import Path

import anthropic
import pandas as pd
import streamlit as st

from analyser import MODEL, classify_dataframe
from rapport import (
    complaints_by_theme,
    explode_themes,
    plot_complaints,
    plot_sentiments,
    priorities,
    recommend,
    sentiment_table,
)

DEMO_RESULTS = Path("output/avis_classes.csv")
DEMO_RECOMMENDATIONS = Path("output/recommandations.json")
MAX_REVIEWS = 200  # limite le coût d'une analyse depuis l'interface

st.set_page_config(page_title="Analyseur d'avis clients", page_icon="💬", layout="wide")


def load_demo() -> tuple[pd.DataFrame, dict[str, str]]:
    df = pd.read_csv(DEMO_RESULTS)
    reco = json.loads(DEMO_RECOMMENDATIONS.read_text(encoding="utf-8")) if DEMO_RECOMMENDATIONS.exists() else {}
    return df, reco


def prepare_upload(file) -> pd.DataFrame | None:
    """Lit le CSV importé et complète les colonnes optionnelles."""
    df = pd.read_csv(file)
    if "avis" not in df.columns:
        st.error("Le fichier doit contenir une colonne `avis`.")
        return None
    df = df.dropna(subset=["avis"]).head(MAX_REVIEWS)
    if "id" not in df.columns:
        df.insert(0, "id", range(1, len(df) + 1))
    if "commerce" not in df.columns:
        df["commerce"] = "Mon commerce"
    return df


# --- Barre latérale : choix des données -------------------------------------
st.sidebar.header("Données")
source = st.sidebar.radio("Source", ["Démo (60 avis fictifs)", "Importer mes avis (CSV)"])

if source.startswith("Démo"):
    st.session_state["results"] = load_demo()
else:
    st.sidebar.caption(
        f"CSV avec une colonne `avis` (et optionnellement `commerce`). "
        f"{MAX_REVIEWS} avis maximum."
    )
    file = st.sidebar.file_uploader("Fichier CSV", type="csv")
    api_key = st.sidebar.text_input(
        "Clé API Anthropic", type="password", help="Utilisée uniquement pour cette analyse, jamais stockée."
    )
    with_reco = st.sidebar.checkbox("Générer des recommandations", value=True)

    if st.sidebar.button("Analyser", type="primary", disabled=not (file and api_key)):
        df = prepare_upload(file)
        if df is not None:
            client = anthropic.Anthropic(api_key=api_key)
            bar = st.progress(0.0, text="Classification des avis...")
            df = classify_dataframe(
                df, client, on_progress=lambda done, total: bar.progress(done / total, text=f"{done}/{total} avis")
            )
            df = df.dropna(subset=["sentiment"])
            reco = {}
            if with_reco and not df.empty:
                bar.progress(1.0, text="Génération des recommandations...")
                reco = recommend(df, explode_themes(df), client)
            bar.empty()
            if df.empty:
                st.error("Aucun avis n'a pu être classifié. Vérifiez la clé API et votre crédit.")
            else:
                st.session_state["results"] = (df, reco)

    if source != st.session_state.get("source"):
        st.session_state.pop("results", None)

st.session_state["source"] = source

# --- Page principale ---------------------------------------------------------
st.title("💬 Analyseur d'avis clients par IA")
st.caption(f"Classification par sentiment et par thématique avec l'API Claude (`{MODEL}`) · pandas · Streamlit")

if "results" not in st.session_state or st.session_state["results"][0].empty:
    st.info("Importez un fichier CSV et votre clé API dans la barre latérale, puis cliquez sur **Analyser**.")
    st.stop()

df, recommendations = st.session_state["results"]

commerces = sorted(df["commerce"].unique())
selected = st.selectbox("Commerce", ["Tous"] + commerces) if len(commerces) > 1 else "Tous"
view = df if selected == "Tous" else df[df["commerce"] == selected]
themes = explode_themes(view)

# Indicateurs clés
shares = view["sentiment"].value_counts(normalize=True) * 100
negative_themes = themes[themes["sentiment"] == "négatif"]["theme"].value_counts()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Avis analysés", len(view))
c2.metric("Positifs", f"{shares.get('positif', 0):.0f} %")
c3.metric("Négatifs", f"{shares.get('négatif', 0):.0f} %")
c4.metric("Thème le plus critiqué", negative_themes.index[0] if not negative_themes.empty else "—")

# Graphiques
left, right = st.columns(2)
left.pyplot(plot_sentiments(sentiment_table(view)))
complaints = complaints_by_theme(themes)
if complaints.empty:
    right.success("Aucun avis négatif 🎉")
else:
    right.pyplot(plot_complaints(complaints))

# Actions prioritaires
st.header("Actions prioritaires")
for commerce, row in priorities(themes).iterrows():
    with st.container(border=True):
        st.subheader(f"{commerce} → améliorer « {row['theme']} »")
        st.write(
            f"{int(row['negatifs'])} avis négatifs sur {int(row['mentions'])} mentions "
            f"({int(row['taux_negatif'])} %)."
        )
        if commerce in recommendations:
            st.markdown("**Recommandations générées par Claude :**")
            st.markdown(recommendations[commerce])

# Détail des avis
st.header("Détail des avis")
sentiment_filter = st.multiselect("Sentiment", ["positif", "neutre", "négatif"], default=["négatif"])
detail = view[view["sentiment"].isin(sentiment_filter)]
st.dataframe(
    detail[["commerce", "avis", "sentiment", "themes", "resume"]],
    hide_index=True,
    width="stretch",
)
st.download_button(
    "Télécharger les résultats (CSV)",
    view.to_csv(index=False).encode("utf-8"),
    file_name="avis_classes.csv",
    mime="text/csv",
)
