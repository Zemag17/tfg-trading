# TFG Trading — Sistema de Backtesting para Estrategias de Trading con ML y RL

Sistema integral de backtesting que compara modelos de aprendizaje supervisado (Regresión Logística, Random Forest, MLP) con un agente DQN (Deep Q-Network) para trading automático de múltiples activos.

## 📋 Requisitos Previos

### Python
- **Python 3.9+** (recomendado 3.10 o superior)
- **pip** (gestor de paquetes de Python)

### Dependencias del Proyecto

Instala todas las dependencias ejecutando:

    pip install -r requirements.txt

**Paquetes principales:**
- `numpy>=1.24.0` — Operaciones numéricas
- `pandas>=1.5.0` — Manipulación de datos
- `scikit-learn>=1.2.0` — Modelos supervisados (LR, RF)
- `tensorflow>=2.12.0` — Red neuronal MLP
- `stable-baselines3>=1.7.0` — Agente DQN
- `gymnasium>=0.27.0` — Entorno de RL
- `matplotlib>=3.6.0` — Visualización
- `seaborn>=0.12.0` — Estilos de gráficos
- `yfinance>=0.2.0` — Descarga de datos financieros (opcional, si usas descarga automática)

## 🚀 Cómo Ejecutar el Proyecto

### 1. Preparar los Datos

Los datos deben estar en formato CSV en la carpeta `datos/`. Alternativamente, puedes descargar automáticamente desde Yahoo Finance:

    python descargar_datos.py

Esto generará archivos CSV en `datos/` para los activos configurados.

### 2. Procesar y Alinear Datos

Prepara los datos para los modelos (estandarización, construcción de features):

    python procesar_datos.py
    python alinear_datos.py

Genera:
- `features/` — Matrices de features estandarizadas
- Particiones de entrenamiento, validación y prueba

### 3. Entrenar Modelos Supervisados

Entrena Regresión Logística, Random Forest, SVM y MLP:

    python modelos_supervisados.py

Salida:
- `datos_features/<ACTIVO>_features.csv` — Matriz de variables estacionarias por activo
- `resultados/metricas_supervisado.csv` — Métricas de clasificación (exactitud, precisión, recall, F1)
- `resultados/senales_supervisado.csv` — Senales generadas (entrada para backtesting)

### 4. Evaluar Modelos Supervisados (Backtesting)

Ejecuta el backtesting descontando costes de transacción (5 puntos básicos):

    python backtest_supervisado.py

Salida:
- `resultados/backtest_supervisado.csv` — Métricas financieras por activo y modelo (Sharpe, retorno total, caida máxima, etc.)
- `resultados/curvas_capital_supervisado.csv` — Evolución diaria del capital en el tramo de prueba

### 5. Entrenar Agente DQN

Entrena un agente Deep Q-Learning con 3 semillas independientes:

    python agente_dqn.py

Opciones:
- `--activos SPY BTC_USD` — Procesar solo ciertos activos
- `--semillas 0 1 2` — Semillas de entrenamiento (defecto: 0, 1, 2)
- `--pasos 30000` — Pasos máximos de entrenamiento por semilla

Salida:
- `resultados/metricas_dqn.csv` — Métricas por semilla (una fila por activo, semilla y partición)
- `resultados/backtest_dqn.csv` — Media y desviación típica entre semillas
- `resultados/senales_dqn.csv` — Senales del agente (validación y prueba)
- `resultados/curvas_capital_dqn.csv` — Curvas de capital por semilla
- `resultados/modelos_dqn/<ACTIVO>_semilla<N>/best_model.zip` — Pesos entrenados

### 6. Generar Gráficos Comparativos

Visualiza los resultados en gráficos SVG y PDF:

    python visualizador.py

Genera en `resultados/`:
- `grafico_supervisado.svg/pdf` — Evolución de capital por activo (modelos supervisados)
- `grafico_dqn.svg/pdf` — Evolución de capital por activo (agentes DQN)
- `grafico_comparativo.svg/pdf` — Ranking de Ratio Sharpe por modelo

### 4. Entrenar Agente DQN

Entrena el agente de Deep Q-Learning con diferentes semillas:

    python agente_dqn.py

Salida:
- Agentes guardados en `modelos/`
- Resultados en `resultados/backtest_dqn.csv`
- Curvas de capital en `resultados/curvas_capital_dqn.csv`

### 5. Generar Gráficos Comparativos

Visualiza los resultados en gráficos SVG y PDF:

    python visualizador.py

Genera en `resultados/`:
- `grafico_supervisado.svg/pdf` — Evolución de capital por activo (modelos supervisados)
- `grafico_dqn.svg/pdf` — Evolución de capital por activo (agentes DQN)
- `grafico_comparativo.svg/pdf` — Ranking de Ratio Sharpe por modelo

## 📁 Estructura del Proyecto

tfg_trading/
├── datos/                          # Datos CSV descargados o ingresados
├── features/                       # Features procesadas y estandarizadas
├── modelos/                        # Modelos entrenados guardados
├── resultados/                     # Resultados de backtests y gráficos
│   ├── backtest_supervisado.csv
│   ├── backtest_dqn.csv
│   ├── curvas_capital_supervisado.csv
│   ├── curvas_capital_dqn.csv
│   └── (gráficos SVG/PDF)
├── descargar_datos.py              # Descarga datos de Yahoo Finance
├── procesar_datos.py               # Limpieza, normalización, features
├── alinear_datos.py                # Alineación temporal de datos
├── modelos_supervisados.py         # Entrenamiento LR, RF, MLP
├── agente_dqn.py                   # Entrenamiento y evaluación del DQN
├── visualizador.py                 # Generación de gráficos
├── requirements.txt                # Dependencias del proyecto
└── README.md                       # Este archivo

## ⚙️ Configuración

Todos los parámetros configurables están centralizados. Edita los valores según tu caso:

**Activos a analizar:** Modifica `ACTIVOS` en `descargar_datos.py`
**Fechas de datos:** Cambia `START_DATE` y `END_DATE`
**Parámetros de RL:** En `agente_dqn.py` ajusta `LEARNING_RATE`, `GAMMA`, `EPSILON_DECAY`
**Ventanas temporales:** En `procesar_datos.py` modifica `WINDOW_SIZE`
**Estilos de gráficos:** En `visualizador.py` personaliza colores, fuentes y tamaño

## 📊 Interpretación de Resultados

### Métricas de Rendimiento

- **Ratio Sharpe:** Rentabilidad ajustada por riesgo (mayor es mejor)
- **Máximo Drawdown:** Caída máxima desde pico anterior (menor es mejor)
- **Rentabilidad Total:** Ganancia/pérdida acumulada en el período
- **Win Rate:** Porcentaje de operaciones ganadoras

### Archivos Generados

**backtest_supervisado.csv:** Contiene para cada modelo y activo:
- Rentabilidad total
- Ratio Sharpe
- Máximo drawdown
- Número de operaciones
- Win rate

**backtest_dqn.csv:** Igual estructura pero para agentes DQN con diferentes semillas

**curvas_capital_*.csv:** Evolución diaria del capital, útil para graficar manualmente

## 🐛 Troubleshooting

**Error: ModuleNotFoundError (numpy, pandas, etc.)**
- Verifica que has ejecutado `pip install -r requirements.txt`
- Comprueba la versión de Python con `python --version` (debe ser 3.9+)

**Error: No such file or directory 'datos/'**
- Crea la carpeta manualmente: `mkdir datos`
- O ejecuta `descargar_datos.py` para descargarlos automáticamente

**Gráficos no se generan en visualizador.py**
- Verifica que matplotlib está correctamente instalado
- Comprueba que los archivos CSV en `resultados/` existen y contienen datos

**Agente DQN no mejora (reward estancado)**
- Aumenta el número de pasos de entrenamiento en `agente_dqn.py`
- Ajusta `LEARNING_RATE` y `GAMMA` en la configuración
- Prueba diferentes semillas (el agente es estocástico)

## 📈 Ejemplo de Ejecución Completa

    python descargar_datos.py
    python procesar_datos.py
    python alinear_datos.py
    python modelos_supervisados.py
    python backtest_supervisado.py
    python agente_dqn.py --activos SPY BTC_USD META PEP TLT GLD --semillas 0 1 2 --pasos 30000
    python visualizador.py

Tras completar estos pasos, abre `resultados/grafico_comparativo.svg` en tu navegador para ver el ranking de Ratio Sharpe.

## 📚 Referencias y Fuentes

- **Stable-Baselines3:** https://stable-baselines3.readthedocs.io/
- **Gymnasium:** https://gymnasium.farama.org/
- **Scikit-Learn:** https://scikit-learn.org/
- **TensorFlow/Keras:** https://tensorflow.org/

## 📝 Licencia

Este proyecto es parte de un Trabajo Fin de Grado (TFG). Hecho por Pablo Gámez Guerrero.