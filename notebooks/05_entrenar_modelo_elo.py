import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

# Función para calcular Sistema ELO
def update_elo(home_elo, away_elo, home_goals, away_goals, k=30):
    gd = abs(home_goals - away_goals)
    if gd <= 1:
        g = 1
    elif gd == 2:
        g = 1.5
    else:
        g = (11 + gd) / 8.0
        
    expected_home = 1 / (10 ** ((away_elo - home_elo) / 400) + 1)
    expected_away = 1 - expected_home
    
    if home_goals > away_goals:
        result_home, result_away = 1, 0
    elif home_goals < away_goals:
        result_home, result_away = 0, 1
    else:
        result_home, result_away = 0.5, 0.5
        
    home_elo_new = home_elo + k * g * (result_home - expected_home)
    away_elo_new = away_elo + k * g * (result_away - expected_away)
    
    return home_elo_new, away_elo_new

# Rutas de datos
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, '..', 'data', 'raw', 'real_results.csv')
if not os.path.exists(csv_path):
    csv_path = '../data/raw/real_results.csv'

# Cargar dataset y prepararlo
df = pd.read_csv(csv_path)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)
df = df.dropna(subset=['home_score', 'away_score'])

# Diccionarios y listas para ELO
elo_dict = {}
home_elos = []
away_elos = []

print("1. Calculando ELO histórico desde 1872 para todos los partidos...")
for idx, row in df.iterrows():
    home = row['home_team']
    away = row['away_team']
    h_score = row['home_score']
    a_score = row['away_score']
    
    h_elo = elo_dict.get(home, 1500)
    a_elo = elo_dict.get(away, 1500)
    
    # ELO ANTES del partido
    home_elos.append(h_elo)
    away_elos.append(a_elo)
    
    # Actualizar ELO para el próximo partido
    h_elo_new, a_elo_new = update_elo(h_elo, a_elo, h_score, a_score)
    elo_dict[home] = h_elo_new
    elo_dict[away] = a_elo_new

df['home_elo'] = home_elos
df['away_elo'] = away_elos

print("2. Calculando estado de forma reciente (últimos 5 partidos)...")
def compute_recent_form(df):
    home_df = df[['date', 'home_team', 'home_score', 'away_score']].copy()
    home_df.columns = ['date', 'team', 'goals_scored', 'goals_conceded']
    home_df['is_home'] = 1
    home_df['original_index'] = df.index
    
    away_df = df[['date', 'away_team', 'away_score', 'home_score']].copy()
    away_df.columns = ['date', 'team', 'goals_scored', 'goals_conceded']
    away_df['is_home'] = 0
    away_df['original_index'] = df.index
    
    long_df = pd.concat([home_df, away_df]).sort_values(['team', 'date']).reset_index(drop=True)
    
    long_df['form_goals_scored'] = long_df.groupby('team')['goals_scored'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    long_df['form_goals_conceded'] = long_df.groupby('team')['goals_conceded'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    
    long_df['form_goals_scored'] = long_df['form_goals_scored'].fillna(1.0)
    long_df['form_goals_conceded'] = long_df['form_goals_conceded'].fillna(1.0)
    
    return long_df

long_df = compute_recent_form(df)
df_home_form = long_df[long_df['is_home'] == 1].set_index('original_index')
df_away_form = long_df[long_df['is_home'] == 0].set_index('original_index')

df['home_form_scored'] = df_home_form['form_goals_scored']
df['home_form_conceded'] = df_home_form['form_goals_conceded']
df['away_form_scored'] = df_away_form['form_goals_scored']
df['away_form_conceded'] = df_away_form['form_goals_conceded']

# Filtramos solo partidos modernos para que el modelo no aprenda de tácticas de 1930
train_df = df[df['date'].dt.year >= 2000].copy()

# Decaimiento Temporal
max_date = train_df['date'].max()
train_df['days_since'] = (max_date - train_df['date']).dt.days
half_life_days = 1500 
train_df['weight'] = np.exp(-np.log(2) * train_df['days_since'] / half_life_days)
weights = train_df['weight']

# Topes de goles para simplificar clasificación
train_df['home_score_capped'] = train_df['home_score'].clip(upper=5)
train_df['away_score_capped'] = train_df['away_score'].clip(upper=5)

# Ya NO usamos clusters, usamos ELO que es infinitamente mejor
features = ['home_team', 'away_team', 'neutral',
            'home_elo', 'away_elo',
            'home_form_scored', 'home_form_conceded', 
            'away_form_scored', 'away_form_conceded']

X = train_df[features]
y_home = train_df['home_score_capped']
y_away = train_df['away_score_capped']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['home_team', 'away_team'])
    ],
    remainder='passthrough'
)

print("3. Entrenando modelos (HistGradientBoosting)...")
X_transformed = preprocessor.fit_transform(X)

home_clf = HistGradientBoostingClassifier(random_state=42, max_depth=5, learning_rate=0.05, max_iter=200)
away_clf = HistGradientBoostingClassifier(random_state=42, max_depth=5, learning_rate=0.05, max_iter=200)

home_clf.fit(X_transformed, y_home, sample_weight=weights)
away_clf.fit(X_transformed, y_away, sample_weight=weights)

print("4. Guardando modelos e inteligencia para la interfaz web...")
# Crear directorio models
models_dir = os.path.join(script_dir, '..', 'models')
os.makedirs(models_dir, exist_ok=True)

joblib.dump(home_clf, os.path.join(models_dir, 'home_clf.joblib'))
joblib.dump(away_clf, os.path.join(models_dir, 'away_clf.joblib'))
joblib.dump(preprocessor, os.path.join(models_dir, 'preprocessor.joblib'))
joblib.dump(elo_dict, os.path.join(models_dir, 'elo_dict.joblib'))

# Extraer el último estado de forma conocido de todos los equipos
latest_form = long_df.groupby('team').last().reset_index()[['team', 'form_goals_scored', 'form_goals_conceded']]
latest_form_dict = latest_form.set_index('team').to_dict('index')
joblib.dump(latest_form_dict, os.path.join(models_dir, 'latest_form_dict.joblib'))

# Guardar la lista de países para el Dropdown de la web
equipos = sorted(list(elo_dict.keys()))
joblib.dump(equipos, os.path.join(models_dir, 'equipos.joblib'))

print("¡Proceso completado! Archivos listos en la carpeta /models/")
