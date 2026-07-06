import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'matches_processed.csv')
    df = pd.read_csv(data_path)
    
    # Agrupar goles mayores a 5 como 5 para simplificar la clasificación
    df['home_score'] = df['home_score'].clip(upper=5)
    df['away_score'] = df['away_score'].clip(upper=5)
    
    X = df[['home_team', 'away_team', 'tournament', 'neutral']]
    y_home = df['home_score']
    y_away = df['away_score']
    
    # Preprocesador: OneHotEncoding para las variables categóricas
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['home_team', 'away_team', 'tournament'])
        ],
        remainder='passthrough' # deja 'neutral' como está si es booleana/numérica
    )
    
    # Pipelines
    home_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10))
    ])
    
    away_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10))
    ])
    
    print("Entrenando modelo de Goles Locales...")
    home_pipeline.fit(X, y_home)
    
    print("Entrenando modelo de Goles Visitantes...")
    away_pipeline.fit(X, y_away)
    
    # Guardar los modelos
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(home_pipeline, os.path.join(models_dir, 'home_pipeline.pkl'))
    joblib.dump(away_pipeline, os.path.join(models_dir, 'away_pipeline.pkl'))
    
    print("Modelos guardados exitosamente en /models.")

if __name__ == '__main__':
    train()
