import pandas as pd
import os
import requests

def fetch_data():
    url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'real_results.csv')
    
    print(f"Descargando datos desde {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
            
        print(f"Datos descargados exitosamente en: {os.path.abspath(output_path)}")
        
        # Validar leyendo con pandas
        df = pd.read_csv(output_path)
        print(f"Dataset cargado con éxito. Contiene {len(df)} partidos históricos.")
        
    except Exception as e:
        print(f"Error al descargar los datos: {e}")

if __name__ == "__main__":
    fetch_data()
