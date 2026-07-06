import pandas as pd
import os

def preprocess_data():
    raw_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'matches.csv')
    processed_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'matches_processed.csv')
    
    print(f"Leyendo datos desde {raw_data_path}...")
    df = pd.read_csv(raw_data_path)
    
    # Eliminar filas con valores nulos en columnas clave
    df = df.dropna(subset=['home_team', 'away_team', 'home_score', 'away_score'])
    
    # Crear una variable objetivo para saber quién ganó (opcional, útil para análisis)
    # 1: Local gana, 0: Empate, -1: Visitante gana
    conditions = [
        (df['home_score'] > df['away_score']),
        (df['home_score'] == df['away_score']),
        (df['home_score'] < df['away_score'])
    ]
    choices = [1, 0, -1]
    df['match_outcome'] = np.select(conditions, choices, default=0)
    
    # Guardar los datos procesados
    df.to_csv(processed_data_path, index=False)
    print(f"Datos preprocesados guardados en {processed_data_path}")

if __name__ == '__main__':
    import numpy as np # import here to avoid error if missing globally
    preprocess_data()
