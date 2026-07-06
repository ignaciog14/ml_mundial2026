import sys
import os
import joblib
import pandas as pd
import numpy as np

def predict_match(home_team, away_team, tournament='FIFA World Cup', neutral=True):
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    
    try:
        home_pipeline = joblib.load(os.path.join(models_dir, 'home_pipeline.pkl'))
        away_pipeline = joblib.load(os.path.join(models_dir, 'away_pipeline.pkl'))
    except FileNotFoundError:
        print("Modelos no encontrados. Por favor, ejecuta train_model.py primero.")
        return
    
    input_data = pd.DataFrame([{
        'home_team': home_team,
        'away_team': away_team,
        'tournament': tournament,
        'neutral': neutral
    }])
    
    home_probs = home_pipeline.predict_proba(input_data)[0]
    away_probs = away_pipeline.predict_proba(input_data)[0]
    
    home_classes = home_pipeline.classes_
    away_classes = away_pipeline.classes_
    
    # Calcular la matriz de probabilidades conjuntas (asumiendo independencia condicional)
    joint_probs = np.outer(home_probs, away_probs)
    
    # Encontrar el índice del resultado más probable
    max_idx = np.unravel_index(np.argmax(joint_probs), joint_probs.shape)
    
    predicted_home_goals = home_classes[max_idx[0]]
    predicted_away_goals = away_classes[max_idx[1]]
    max_prob = joint_probs[max_idx]
    
    print(f"\nPredicción para: {home_team} vs {away_team}")
    print("-" * 40)
    print(f"Resultado más probable: {home_team} {predicted_home_goals} - {predicted_away_goals} {away_team}")
    print(f"Confianza (Probabilidad estimada): {max_prob * 100:.2f}%")
    
    # Mostrar top 3 de resultados probables
    print("\nOtros resultados probables:")
    flat_probs = joint_probs.flatten()
    top_indices = np.argsort(flat_probs)[::-1][1:4] # top 2 al 4
    
    for idx in top_indices:
        h_idx, a_idx = np.unravel_index(idx, joint_probs.shape)
        p_h = home_classes[h_idx]
        p_a = away_classes[a_idx]
        prob = joint_probs[h_idx, a_idx]
        print(f"{home_team} {p_h} - {p_a} {away_team}  ({prob * 100:.2f}%)")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Uso: python predict.py <Equipo Local> <Equipo Visitante>")
        print("Ejemplo: python predict.py Argentina France")
    else:
        h_team = sys.argv[1]
        a_team = sys.argv[2]
        predict_match(h_team, a_team)
