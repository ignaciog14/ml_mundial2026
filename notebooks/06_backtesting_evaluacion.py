import pandas as pd
import numpy as np
import os
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

# 1. Definir funciones clave (mismas que usamos en producción)
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

# 2. Cargar datos
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, '..', 'data', 'raw', 'real_results.csv')
if not os.path.exists(csv_path):
    csv_path = '../data/raw/real_results.csv'

df = pd.read_csv(csv_path)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)
df = df.dropna(subset=['home_score', 'away_score'])

# 3. Calcular ELO Histórico (hasta la fecha de cada partido)
elo_dict = {}
home_elos = []
away_elos = []

for idx, row in df.iterrows():
    home, away = row['home_team'], row['away_team']
    h_score, a_score = row['home_score'], row['away_score']
    
    h_elo = elo_dict.get(home, 1500)
    a_elo = elo_dict.get(away, 1500)
    
    home_elos.append(h_elo)
    away_elos.append(a_elo)
    
    h_elo_new, a_elo_new = update_elo(h_elo, a_elo, h_score, a_score)
    elo_dict[home] = h_elo_new
    elo_dict[away] = a_elo_new

df['home_elo'] = home_elos
df['away_elo'] = away_elos

# 4. Estado de Forma Reciente
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

# --- AQUÍ EMPIEZA LA MAGIA DEL BACKTESTING ---

# Dividimos el tiempo:
# Entrenamos con datos modernos (>= 2000) pero SOLO hasta el 4 de Julio 2026.
train_df = df[(df['date'].dt.year >= 2000) & (df['date'] <= '2026-07-04')].copy()

# El Examen Sorpresa: Partidos reales del 5 de Julio al 11 de Julio
test_df = df[df['date'] >= '2026-07-05'].copy()

print(f"[{train_df.shape[0]} partidos para Entrenamiento (hasta 4 Jul)]")
print(f"[{test_df.shape[0]} partidos para el Examen Final (5 Jul en adelante)]\n")

if test_df.empty:
    print("No hay partidos en el Test Set. Verifica la base de datos.")
    exit()

# Decaimiento temporal para entrenamiento
max_date_train = train_df['date'].max()
train_df['days_since'] = (max_date_train - train_df['date']).dt.days
train_df['weight'] = np.exp(-np.log(2) * train_df['days_since'] / 1500)
weights = train_df['weight']

# Preparar variables
train_df['home_score_capped'] = train_df['home_score'].clip(upper=5)
train_df['away_score_capped'] = train_df['away_score'].clip(upper=5)

features = ['home_team', 'away_team', 'neutral',
            'home_elo', 'away_elo',
            'home_form_scored', 'home_form_conceded', 
            'away_form_scored', 'away_form_conceded']

X_train = train_df[features]
y_home_train = train_df['home_score_capped']
y_away_train = train_df['away_score_capped']

X_test = test_df[features]

preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['home_team', 'away_team'])],
    remainder='passthrough'
)

X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Entrenar modelo ciego al futuro
home_clf = HistGradientBoostingClassifier(random_state=42, max_depth=5, learning_rate=0.05, max_iter=200)
away_clf = HistGradientBoostingClassifier(random_state=42, max_depth=5, learning_rate=0.05, max_iter=200)

home_clf.fit(X_train_transformed, y_home_train, sample_weight=weights)
away_clf.fit(X_train_transformed, y_away_train, sample_weight=weights)

# Evaluar el modelo (Obtener marcador más probable del Heatmap)
def get_most_probable_score(row_features):
    ph = home_clf.predict_proba(row_features)[0]
    pa = away_clf.predict_proba(row_features)[0]
    
    joint = np.outer(ph, pa)
    max_idx = np.unravel_index(np.argmax(joint), joint.shape)
    
    pred_h = home_clf.classes_[max_idx[0]]
    pred_a = away_clf.classes_[max_idx[1]]
    
    return pred_h, pred_a

print("--- EXAMEN FINAL: RESULTADOS ---")
print(f"{'FECHA':<12} | {'PARTIDO':<30} | {'PREDICCIÓN MODELO':<18} | {'MARCADOR REAL':<15} | {'¿ACERTÓ TENDENCIA?':<20}")
print("-" * 105)

aciertos_exactos = 0
aciertos_tendencia = 0
total = test_df.shape[0]

for idx, row in test_df.iterrows():
    real_h = row['home_score']
    real_a = row['away_score']
    
    # Determinar tendencia real (1 = Local, X = Empate, 2 = Visitante)
    if real_h > real_a: real_trend = 1
    elif real_h < real_a: real_trend = 2
    else: real_trend = 0
    
    # Extraer features y predecir
    row_df = pd.DataFrame([row[features]])
    row_transformed = preprocessor.transform(row_df)
    
    pred_h, pred_a = get_most_probable_score(row_transformed)
    
    if pred_h > pred_a: pred_trend = 1
    elif pred_h < pred_a: pred_trend = 2
    else: pred_trend = 0
    
    es_exacto = (pred_h == real_h and pred_a == real_a)
    es_tendencia = (pred_trend == real_trend)
    
    if es_exacto:
        aciertos_exactos += 1
    if es_tendencia:
        aciertos_tendencia += 1
        
    fecha = row['date'].strftime('%Y-%m-%d')
    partido = f"{row['home_team']} vs {row['away_team']}"
    pred_str = f"{int(pred_h)} - {int(pred_a)}"
    real_str = f"{int(real_h)} - {int(real_a)}"
    
    tendencia_str = "SI" if es_tendencia else "NO"
    if es_exacto:
        tendencia_str = "SI (Marcador Exacto!)"
        
    print(f"{fecha:<12} | {partido:<30} | {pred_str:<18} | {real_str:<15} | {tendencia_str:<20}")

print("\n--- RESUMEN CIENTÍFICO ---")
print(f"Total Partidos Evaluados (Test Set): {total}")
print(f"Aciertos de Tendencia (Ganador/Empate): {aciertos_tendencia} de {total} ({aciertos_tendencia/total*100:.1f}%)")
print(f"Aciertos de Marcador Exacto Perfecto: {aciertos_exactos} de {total} ({aciertos_exactos/total*100:.1f}%)")
print("Nota: En el fútbol profesional y las casas de apuestas, predecir el marcador exacto perfecto tiene un éxito base (azar) de apenas un ~3-5%, y acertar ganador ronda el ~33% al azar.")
