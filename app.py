import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Predicción Mundial ELO", page_icon="⚽", layout="centered")

@st.cache_resource
def load_models():
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    
    home_clf = joblib.load(os.path.join(models_dir, 'home_clf.joblib'))
    away_clf = joblib.load(os.path.join(models_dir, 'away_clf.joblib'))
    preprocessor = joblib.load(os.path.join(models_dir, 'preprocessor.joblib'))
    elo_dict = joblib.load(os.path.join(models_dir, 'elo_dict.joblib'))
    latest_form_dict = joblib.load(os.path.join(models_dir, 'latest_form_dict.joblib'))
    equipos = joblib.load(os.path.join(models_dir, 'equipos.joblib'))
    return home_clf, away_clf, preprocessor, elo_dict, latest_form_dict, equipos

st.title("⚽ Predicción del Mundial de la FIFA")
st.markdown("Este modelo utiliza un **Sistema de Clasificación ELO Histórico** (desde 1872) y algoritmos de **Gradient Boosting** para predecir marcadores exactos superando el problema de falsos favoritos.")

try:
    home_clf, away_clf, preprocessor, elo_dict, latest_form_dict, equipos = load_models()
except FileNotFoundError:
    st.error("No se encontraron los modelos. Por favor ejecuta el script de entrenamiento '05_entrenar_modelo_elo.py' primero.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    equipo_a = st.selectbox("Selecciona al Equipo Local:", equipos, index=equipos.index("Spain") if "Spain" in equipos else 0)
with col2:
    equipo_b = st.selectbox("Selecciona al Equipo Visitante:", equipos, index=equipos.index("Portugal") if "Portugal" in equipos else 1)

if equipo_a == equipo_b:
    st.warning("Selecciona dos equipos diferentes.")
else:
    # Mostrar estadísticas actuales
    elo_a = elo_dict.get(equipo_a, 1500)
    elo_b = elo_dict.get(equipo_b, 1500)
    
    st.markdown("---")
    st.markdown(f"**Puntuación ELO Actual (Fuerza Relativa):** {equipo_a} (🏆 {int(elo_a)}) vs {equipo_b} (🏆 {int(elo_b)})")
    
    if st.button("🚀 Predecir Resultado", use_container_width=True):
        with st.spinner('Analizando millones de posibilidades...'):
            a_form = latest_form_dict.get(equipo_a, {'form_goals_scored': 1.0, 'form_goals_conceded': 1.0})
            b_form = latest_form_dict.get(equipo_b, {'form_goals_scored': 1.0, 'form_goals_conceded': 1.0})
            
            input_df = pd.DataFrame([{
                'home_team': equipo_a,
                'away_team': equipo_b,
                'neutral': True,
                'home_elo': elo_a,
                'away_elo': elo_b,
                'home_form_scored': a_form['form_goals_scored'],
                'home_form_conceded': a_form['form_goals_conceded'],
                'away_form_scored': b_form['form_goals_scored'],
                'away_form_conceded': b_form['form_goals_conceded']
            }])
            
            X_input = preprocessor.transform(input_df)
            
            probs_home = home_clf.predict_proba(X_input)[0]
            probs_away = away_clf.predict_proba(X_input)[0]
            
            classes_home = home_clf.classes_
            classes_away = away_clf.classes_
            
            joint_probs = np.outer(probs_home, probs_away)
            max_idx = np.unravel_index(np.argmax(joint_probs), joint_probs.shape)
            
            pred_h = classes_home[max_idx[0]]
            pred_a = classes_away[max_idx[1]]
            prob_max = joint_probs[max_idx]
            
            st.success(f"🎯 **Resultado más probable:** {equipo_a} {int(pred_h)} - {int(pred_a)} {equipo_b} ({prob_max*100:.1f}%)")
            
            # Heatmap
            st.subheader("Mapa de Calor de Probabilidades (%)")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(joint_probs * 100, 
                        annot=True, 
                        fmt=".1f", 
                        cmap="Reds", 
                        xticklabels=[int(c) for c in classes_away], 
                        yticklabels=[int(c) for c in classes_home],
                        ax=ax)
    
            ax.set_xlabel(f"Goles de {equipo_b} (Visitante)")
            ax.set_ylabel(f"Goles de {equipo_a} (Local)")
            ax.invert_yaxis()
            
            st.pyplot(fig)
