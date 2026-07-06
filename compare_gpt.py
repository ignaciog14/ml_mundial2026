import os
import joblib
import pandas as pd
import numpy as np

# Load models
models_dir = os.path.join(os.path.dirname(__file__), 'models')
home_clf = joblib.load(os.path.join(models_dir, 'home_clf.joblib'))
away_clf = joblib.load(os.path.join(models_dir, 'away_clf.joblib'))
preprocessor = joblib.load(os.path.join(models_dir, 'preprocessor.joblib'))
elo_dict = joblib.load(os.path.join(models_dir, 'elo_dict.joblib'))
latest_form_dict = joblib.load(os.path.join(models_dir, 'latest_form_dict.joblib'))
equipos = joblib.load(os.path.join(models_dir, 'equipos.joblib'))

def predict_match(home_team, away_team):
    home_elo = elo_dict.get(home_team, 1500)
    away_elo = elo_dict.get(away_team, 1500)
    
    hf = latest_form_dict.get(home_team, {'form_goals_scored': 1.0, 'form_goals_conceded': 1.0})
    af = latest_form_dict.get(away_team, {'form_goals_scored': 1.0, 'form_goals_conceded': 1.0})
    
    home_form_scored = hf['form_goals_scored']
    home_form_conceded = hf['form_goals_conceded']
    away_form_scored = af['form_goals_scored']
    away_form_conceded = af['form_goals_conceded']
    
    input_data = pd.DataFrame([{
        'home_team': home_team if home_team in equipos else 'Other',
        'away_team': away_team if away_team in equipos else 'Other',
        'neutral': 1,
        'home_elo': home_elo,
        'away_elo': away_elo,
        'home_form_scored': home_form_scored,
        'home_form_conceded': home_form_conceded,
        'away_form_scored': away_form_scored,
        'away_form_conceded': away_form_conceded
    }])
    
    X_transformed = preprocessor.transform(input_data)
    
    ph = home_clf.predict_proba(X_transformed)[0]
    pa = away_clf.predict_proba(X_transformed)[0]
    
    joint = np.outer(ph, pa)
    
    prob_home = np.sum(np.tril(joint, -1))
    prob_draw = np.sum(np.diag(joint))
    prob_away = np.sum(np.triu(joint, 1))
    
    max_idx = np.unravel_index(np.argmax(joint), joint.shape)
    pred_h = home_clf.classes_[max_idx[0]]
    pred_a = away_clf.classes_[max_idx[1]]
    
    return pred_h, pred_a, prob_home, prob_draw, prob_away

matches = [
    ("United States", "Belgium", "EE.UU. vs Bélgica (ChatGPT: 1-1, EE.UU. pasa)"),
    ("Argentina", "Egypt", "Argentina vs Egipto (ChatGPT: Argentina 2-0)"),
    ("Switzerland", "Colombia", "Suiza vs Colombia (ChatGPT: Colombia 1-0)"),
    ("France", "Morocco", "Francia vs Marruecos (ChatGPT: Francia 2-0)"),
    ("Spain", "United States", "España vs EE.UU. (ChatGPT: España 2-0)"),
    ("Norway", "England", "Noruega vs Inglaterra (ChatGPT: Inglaterra 2-1)"),
    ("Argentina", "Colombia", "Argentina vs Colombia (ChatGPT: Argentina 1-0)")
]

print("--- COMPARATIVA: NUESTRO MODELO VS CHATGPT ---")
for home, away, desc in matches:
    h_score, a_score, p_h, p_d, p_a = predict_match(home, away)
    print(f"\n{desc}")
    print(f"Nuestro Modelo: {home} {int(h_score)} - {int(a_score)} {away}")
    
    if p_h > p_a and p_h > p_d:
        fav = home
        pct = p_h * 100
    elif p_a > p_h and p_a > p_d:
        fav = away
        pct = p_a * 100
    else:
        fav = "Empate"
        pct = p_d * 100
        
    print(f"Probabilidad principal: {fav} al {pct:.1f}%")
