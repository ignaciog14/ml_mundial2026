import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_mock_data():
    np.random.seed(42)
    
    teams = ['Argentina', 'France', 'Brazil', 'Germany', 'Spain', 'England', 
             'Portugal', 'Italy', 'Netherlands', 'Croatia', 'Uruguay', 'Belgium']
    
    # Base strength (higher is better)
    strength = {
        'Argentina': 90, 'France': 91, 'Brazil': 89, 'Germany': 85,
        'Spain': 88, 'England': 87, 'Portugal': 86, 'Italy': 84,
        'Netherlands': 85, 'Croatia': 82, 'Uruguay': 80, 'Belgium': 83
    }
    
    # Generamos 1000 partidos históricos
    data = []
    start_date = datetime(2010, 1, 1)
    
    for _ in range(1000):
        t1, t2 = np.random.choice(teams, 2, replace=False)
        
        # Probabilidad de goles basada en la diferencia de fuerza y un factor aleatorio (Poisson)
        diff = strength[t1] - strength[t2]
        
        lambda_t1 = max(0.5, 1.5 + diff * 0.05)
        lambda_t2 = max(0.5, 1.5 - diff * 0.05)
        
        home_score = np.random.poisson(lambda_t1)
        away_score = np.random.poisson(lambda_t2)
        
        match_date = start_date + timedelta(days=np.random.randint(0, 5000))
        
        data.append({
            'date': match_date.strftime('%Y-%m-%d'),
            'home_team': t1,
            'away_team': t2,
            'home_score': home_score,
            'away_score': away_score,
            'tournament': np.random.choice(['Friendly', 'FIFA World Cup qualification', 'FIFA World Cup'], p=[0.6, 0.3, 0.1]),
            'neutral': np.random.choice([True, False])
        })
        
    df = pd.DataFrame(data)
    df = df.sort_values('date')
    
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'matches.csv')
    df.to_csv(output_path, index=False)
    print(f"Mock data generada en: {os.path.abspath(output_path)}")

if __name__ == '__main__':
    generate_mock_data()
