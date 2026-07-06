# %% [markdown]
# # Predicción Avanzada de Partidos del Mundial
# Este script mejora el modelo básico incorporando:
# 1. El estado de forma reciente (promedio de goles en los últimos 5 partidos).
# 2. Decaimiento temporal (mayor peso a los partidos jugados recientemente).
# 3. Un algoritmo más potente: HistGradientBoostingClassifier (similar a XGBoost).

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## 1. Carga de Datos
# %%
# Cargar el dataset real (manejando rutas relativas desde distintas terminales)
import os
csv_path = '../data/raw/real_results.csv'
if not os.path.exists(csv_path):
    csv_path = 'data/raw/real_results.csv'
df = pd.read_csv(csv_path)
# Asegurarnos de que los nombres de los equipos estén bien (el dataset usa inglés)
df['date'] = pd.to_datetime(df['date'])
df = df[df['date'].dt.year >= 2000].copy()
df = df.sort_values('date').reset_index(drop=True)

# Eliminar registros con nulos
df = df.dropna(subset=['home_score', 'away_score'])

print(f"Total de partidos recientes: {len(df)}")
df.head()

# %% [markdown]
# ## 2. Feature Engineering: Estado de Forma Reciente
# Calculamos los promedios de goles marcados y recibidos en los últimos 5 partidos para cada equipo.
# %%
def compute_recent_form(df):
    # Separar en partidos de local y visitante para tener 1 fila por cada equipo que jugó
    home_df = df[['date', 'home_team', 'home_score', 'away_score']].copy()
    home_df.columns = ['date', 'team', 'goals_scored', 'goals_conceded']
    home_df['is_home'] = 1
    home_df['original_index'] = df.index
    
    away_df = df[['date', 'away_team', 'away_score', 'home_score']].copy()
    away_df.columns = ['date', 'team', 'goals_scored', 'goals_conceded']
    away_df['is_home'] = 0
    away_df['original_index'] = df.index
    
    # Combinar y ordenar cronológicamente por equipo
    long_df = pd.concat([home_df, away_df]).sort_values(['team', 'date']).reset_index(drop=True)
    
    # Calcular promedio móvil de los ÚLTIMOS 5 partidos (usamos shift(1) para no incluir el partido actual)
    long_df['form_goals_scored'] = long_df.groupby('team')['goals_scored'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    long_df['form_goals_conceded'] = long_df.groupby('team')['goals_conceded'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    
    # Rellenar con promedio 1.0 para el primer partido de cada equipo
    long_df['form_goals_scored'] = long_df['form_goals_scored'].fillna(1.0)
    long_df['form_goals_conceded'] = long_df['form_goals_conceded'].fillna(1.0)
    
    return long_df

long_df = compute_recent_form(df)

# Unir estos nuevos cálculos al dataset principal
df_home_form = long_df[long_df['is_home'] == 1].set_index('original_index')
df_away_form = long_df[long_df['is_home'] == 0].set_index('original_index')

df['home_form_scored'] = df_home_form['form_goals_scored']
df['home_form_conceded'] = df_home_form['form_goals_conceded']
df['away_form_scored'] = df_away_form['form_goals_scored']
df['away_form_conceded'] = df_away_form['form_goals_conceded']

print("Características de estado de forma calculadas con éxito.")

# %% [markdown]
# ## 3. Feature Engineering: Clustering Histórico (Niveles)
# %%
home_stats = df.groupby('home_team').agg(
    goles_marcados=('home_score', 'mean'),
    goles_recibidos=('away_score', 'mean'),
    partidos=('home_score', 'count')
).reset_index().rename(columns={'home_team': 'team'})

home_stats = home_stats[home_stats['partidos'] >= 20]

kmeans = KMeans(n_clusters=4, random_state=42)
features_k = home_stats[['goles_marcados', 'goles_recibidos']]
home_stats['cluster_nivel'] = kmeans.fit_predict(features_k)

team_clusters = dict(zip(home_stats['team'], home_stats['cluster_nivel']))
df['home_cluster'] = df['home_team'].map(team_clusters).fillna(2)
df['away_cluster'] = df['away_team'].map(team_clusters).fillna(2)

# %% [markdown]
# ## 4. Preparación para Modelado y Pesos Temporales
# Le daremos más peso a los partidos recientes usando decaimiento exponencial.
# %%
# Simplificamos los goles altos
df['home_score_capped'] = df['home_score'].clip(upper=5)
df['away_score_capped'] = df['away_score'].clip(upper=5)

features = ['home_team', 'away_team', 'home_cluster', 'away_cluster', 'neutral',
            'home_form_scored', 'home_form_conceded', 'away_form_scored', 'away_form_conceded']

X = df[features]
y_home = df['home_score_capped']
y_away = df['away_score_capped']

# Calcular peso: los partidos de hace 4 años (aprox 1500 días) valen la mitad que los de hoy.
max_date = df['date'].max()
df['days_since'] = (max_date - df['date']).dt.days
half_life_days = 1500 
df['weight'] = np.exp(-np.log(2) * df['days_since'] / half_life_days)
weights = df['weight']

print(f"Pesos calculados. Peso del partido más reciente: {weights.iloc[-1]:.2f}. Peso del partido más antiguo: {weights.iloc[0]:.2f}")

# %% [markdown]
# ## 5. Entrenamiento de HistGradientBoosting
# %%
# Transformar variables categóricas (los nombres de los equipos) a números
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['home_team', 'away_team'])
    ],
    remainder='passthrough'
)

print("Transformando datos...")
X_transformed = preprocessor.fit_transform(X)

# Definir los modelos
home_clf = HistGradientBoostingClassifier(random_state=42, max_depth=5, learning_rate=0.05, max_iter=200)
away_clf = HistGradientBoostingClassifier(random_state=42, max_depth=5, learning_rate=0.05, max_iter=200)

print("Entrenando modelo de Goles Local...")
home_clf.fit(X_transformed, y_home, sample_weight=weights)

print("Entrenando modelo de Goles Visitante...")
away_clf.fit(X_transformed, y_away, sample_weight=weights)

print("¡Entrenamiento completado exitosamente!")

# %% [markdown]
# ## 6. Interfaz de Predicción
# %%
def predecir_partido(equipo_a, equipo_b):
    # Validar nombres de equipos
    if equipo_a not in team_clusters:
        print(f"Advertencia: No hay suficientes datos históricos para {equipo_a}")
    if equipo_b not in team_clusters:
        print(f"Advertencia: No hay suficientes datos históricos para {equipo_b}")
        
    cluster_a = team_clusters.get(equipo_a, 2)
    cluster_b = team_clusters.get(equipo_b, 2)
    
    # Buscar forma más reciente de los equipos
    a_matches = long_df[long_df['team'] == equipo_a].sort_values('date')
    if len(a_matches) > 0:
        a_form_scored = a_matches.iloc[-1]['goals_scored']
        a_form_conceded = a_matches.iloc[-1]['goals_conceded']
    else:
        a_form_scored, a_form_conceded = 1.0, 1.0
        
    b_matches = long_df[long_df['team'] == equipo_b].sort_values('date')
    if len(b_matches) > 0:
        b_form_scored = b_matches.iloc[-1]['goals_scored']
        b_form_conceded = b_matches.iloc[-1]['goals_conceded']
    else:
        b_form_scored, b_form_conceded = 1.0, 1.0
        
    # Crear input
    input_df = pd.DataFrame([{
        'home_team': equipo_a,
        'away_team': equipo_b,
        'home_cluster': cluster_a,
        'away_cluster': cluster_b,
        'neutral': True,
        'home_form_scored': a_form_scored,
        'home_form_conceded': a_form_conceded,
        'away_form_scored': b_form_scored,
        'away_form_conceded': b_form_conceded
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
    
    print(f"==========================================")
    print(f"⚽ PREDICCIÓN: {equipo_a} vs {equipo_b}")
    print(f"🎯 Resultado más probable: {equipo_a} {pred_h} - {pred_a} {equipo_b} ({prob_max*100:.1f}%)")
    print(f"==========================================")
    
    print("\n--- MATRIZ DE PROBABILIDADES (%) ---")
    header = " " * 10 + "".join([f"{int(c):>7}g" for c in classes_away])
    print(header)
    for i, row in enumerate(joint_probs):
        row_str = "".join([f"{val*100:>8.1f}" for val in row])
        print(f"{int(classes_home[i]):>3}g local |{row_str}")
    print("------------------------------------\n")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(joint_probs * 100, 
                annot=True, 
                fmt=".1f", 
                cmap="Reds", 
                xticklabels=[int(c) for c in classes_away], 
                yticklabels=[int(c) for c in classes_home])

    plt.title(f"Mapa de Probabilidades: {equipo_a} vs {equipo_b}")
    plt.xlabel(f"Goles de {equipo_b} (Visitante)")
    plt.ylabel(f"Goles de {equipo_a} (Local)")
    plt.gca().invert_yaxis()
    plt.show()

# Nota importante: Usa los nombres en inglés como están en el dataset ("Spain", no "España")
predecir_partido('France', 'Iran')
