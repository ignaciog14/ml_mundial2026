# 🏆 Predicción Mundial de la FIFA - Machine Learning ⚽

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-scikit--learn-orange.svg)
![Streamlit](https://img.shields.io/badge/Web%20App-Streamlit-red.svg)

Una aplicación interactiva impulsada por **Inteligencia Artificial** que predice el resultado exacto (marcador y probabilidades) de cualquier enfrentamiento entre selecciones nacionales de fútbol.

## 🧠 ¿Cómo funciona la Inteligencia Artificial?

Este modelo no se basa en el azar ni en encuestas. Utiliza un motor predictivo robusto construido con **HistGradientBoostingClassifier** que analiza millones de datos. Su inteligencia se apoya en tres pilares fundamentales:

1. **Sistema ELO Histórico (Desde 1872):**
   El modelo incluye una simulación cronológica de todos los partidos internacionales de la historia. Utiliza un sistema ELO (similar al del ajedrez y al ranking oficial de la FIFA) para calcular la verdadera fuerza relativa de una selección. Esto evita el problema de equipos débiles con "buenas rachas" frente a rivales pequeños (ej. *El problema del Pez Grande en Estanque Pequeño*).
   
2. **Estado de Forma Reciente:**
   El algoritmo viaja en el tiempo y calcula el promedio móvil de goles a favor y en contra de cada selección en sus **últimos 5 partidos** exactos antes del encuentro a predecir, capturando el "momento de forma" del equipo.

3. **Decaimiento Temporal (Amnesia Selectiva):**
   A los partidos jugados en la década de los 2000 se les asigna matemáticamente un peso cercano a cero, mientras que los partidos de los últimos 2 años tienen un peso del ~100% durante el entrenamiento del modelo.

## 📂 Arquitectura y Flujo de Trabajo

El sistema entrena **dos modelos independientes**:
* Un modelo predictivo exclusivo para los Goles del Equipo Local.
* Un modelo predictivo exclusivo para los Goles del Equipo Visitante.

Posteriormente, crea una **Matriz de Probabilidad Conjunta (Heatmap)** cruzando ambos modelos para descubrir matemáticamente el marcador exacto con mayor porcentaje de probabilidad.

## 🚀 Cómo ejecutarlo localmente

Si deseas probar el modelo en tu propia computadora:

1. **Clona el repositorio:**
   ```bash
   git clone https://github.com/TU_USUARIO/ml_mundial2026.git
   cd ml_mundial2026
   ```

2. **Instala las dependencias necesarias:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Inicia la aplicación web interactiva:**
   ```bash
   python -m streamlit run app.py
   ```

4. El servidor se iniciará localmente y se abrirá una pestaña en tu navegador web en `http://localhost:8501`.

## 📁 Estructura del Proyecto

```text
📁 ml_mundial2026/
├── 📄 app.py                        # Interfaz gráfica de la Web App en Streamlit
├── 📄 requirements.txt              # Librerías necesarias (pandas, scikit-learn, etc)
├── 📁 models/                       # Modelos .joblib ya entrenados e inteligencia ELO
├── 📁 notebooks/                    # Notebooks de Jupyter y scripts de experimentación
│   └── 📄 05_entrenar_modelo_elo.py # Script matemático principal que simula y entrena el ELO
└── 📁 data/                         # Conjunto de datos históricos en CSV
```

---
*Desarrollado con pasión, matemáticas y mucho fútbol.* 🏟️
