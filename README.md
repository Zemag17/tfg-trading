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
- `scikit-learn>=1.2.0` — Modelos supervisados (LR, RF, MLP)
- `stable-baselines3>=1.7.0` — Agente DQN
- `gymnasium>=0.27.0` — Entorno de RL
- `matplotlib>=3.6.0` — Visualización
- `seaborn>=0.12.0` — Estilos de gráficos
- `yfinance>=0.2.0` — Descarga de datos financieros (opcional, si usas descarga automática)

## 🚀 Cómo Ejecutar el Proyecto

### 1. Preparar los Datos

Los datos deben estar en formato CSV en la carpeta `datos/`. Alternativamente, puedes descargarlos automáticamente desde Yahoo Finance:

    python descargar_datos.py

Esto generará archivos CSV en `datos/` para los activos configurados.

### 2. Procesar y Alinear Datos

Prepara los datos para los modelos (estandarización, construcción de features):

    python procesar_datos.py
    python alinear_datos.py

Genera:
- `features/` — Matrices de features estandarizadas
- Particiones cronológicas de entrenamiento, validación y prueba

### 3. Entrenar Modelos Supervisados

Entrena Regresión Logística, Random Forest, SVM y MLP:

    python modelos_supervisados.py

Salida:
- `datos_features/<ACTIVO>_features.csv` — Matriz de variables estacionarias por activo
- `resultados/metricas_supervisado.csv` — Métricas de clasificación (exactitud, precisión, recall, F1)
- `resultados/senales_supervisado.csv` — Señales generadas (entrada para backtesting)

### 4. Evaluar Modelos Supervisados (Backtesting)

Ejecuta el backtesting descontando costes de transacción (5 puntos básicos):

    python backtest_supervisado.py

Salida:
- `resultados/backtest_supervisado.csv` — Métricas financieras por activo y modelo (Sharpe, retorno total, caída máxima, etc.)
- `resultados/curvas_capital_supervisado.csv` — Evolución diaria del capital en el tramo de prueba

### 5. Entrenar Agente DQN y Evaluar

Entrena un agente Deep Q-Learning con 3 semillas independientes y genera sus métricas de backtest:

    python agente_dqn.py

Opciones:
- `--activos SPY BTC_USD` — Procesar solo ciertos activos
- `--semillas 0 1 2` — Semillas de entrenamiento (defecto: 0, 1, 2)
- `--pasos 30000` — Pasos máximos de entrenamiento por semilla

Salida:
- `resultados/metricas_dqn.csv` — Métricas por semilla (una fila por activo, semilla y partición)
- `resultados/backtest_dqn.csv` — Media y desviación típica entre semillas
- `resultados/senales_dqn.csv` — Señales del agente (validación y prueba)
- `resultados/curvas_capital_dqn.csv` — Curvas de capital por semilla
- `resultados/modelos_dqn/<ACTIVO>_semilla<N>/best_model.zip` — Pesos entrenados

### 6. Generar Gráficos Comparativos

Visualiza los resultados en gráficos vectoriales SVG y PDF:

    python visualizador.py

Genera en `resultados/`:
- `grafico_supervisado.svg/pdf` — Evolución de capital por activo (modelos supervisados)
- `grafico_dqn.svg/pdf` — Evolución de capital por activo (agentes DQN)
- `grafico_comparativo.svg/pdf` — Ranking global de Ratio Sharpe por modelo

## 📁 Estructura del Proyecto

    tfg_trading/
    ├── datos/                          # Datos CSV descargados o ingresados
    ├── features/                       # Features procesadas y estandarizadas
    ├── resultados/                     # Resultados de backtests y gráficos
    │   ├── backtest_supervisado.csv
    │   ├── backtest_dqn.csv
    │   ├── curvas_capital_supervisado.csv
    │   ├── curvas_capital_dqn.csv
    │   └── (gráficos SVG/PDF)
    ├── descargar_datos.py              # Descarga datos de Yahoo Finance
    ├── procesar_datos.py               # Limpieza, normalización, features
    ├── alinear_datos.py                # Alineación temporal de datos
    ├── modelos_supervisados.py         # Entrenamiento LR, RF, SVM, MLP
    ├── agente_dqn.py                   # Entrenamiento y evaluación del DQN
    ├── visualizador.py                 # Generación de gráficos
    ├── requirements.txt                # Dependencias del proyecto
    └── README.md                       # Este archivo

## ⚙️ Configuración

Todos los parámetros configurables están centralizados:
- **Activos a analizar:** Modifica `ACTIVOS` en los scripts principales.
- **Fechas de datos:** Cambia los rangos temporales en los módulos de carga.
- **Parámetros de RL:** En `agente_dqn.py` se ajustan tasas de aprendizaje, gamma y epsilon.
- **Fricción de mercado:** El coste por operación está fijado por defecto en `COSTE_OPERACION = 0.0005` (5 puntos básicos).

## 📊 Interpretación de Resultados

### Métricas de Rendimiento
- **Ratio Sharpe:** Rentabilidad ajustada por riesgo total (mayor es mejor).
- **Ratio Sortino:** Rentabilidad ajustada exclusivamente por riesgo de bajada.
- **Máximo Drawdown:** Caída máxima relativa desde un pico anterior.
- **Exposición:** Porcentaje de tiempo que el modelo permanece invertido frente a liquidez.

## 🐛 Troubleshooting

**Error: ModuleNotFoundError**
- Verifica que has activado tu entorno virtual y ejecutado `pip install -r requirements.txt`.

**Gráficos no se generan en visualizador.py**
- Comprueba que los archivos CSV en la carpeta `resultados/` existen y han sido generados previamente ejecutando los scripts de backtest.

**Agente DQN sobreopera (pérdidas por comisiones)**
- El entorno incorpora un multiplicador de castigo en la función de recompensa para mitigar este comportamiento en las versiones base del TFG.

## 📈 Ejemplo de Ejecución Completa

    python descargar_datos.py
    python procesar_datos.py
    python alinear_datos.py
    python modelos_supervisados.py
    python backtest_supervisado.py
    python agente_dqn.py --activos SPY BTC_USD META PEP TLT GLD --semillas 0 1 2 --pasos 30000
    python visualizador.py

Tras completar estos pasos, abre `resultados/grafico_comparativo.svg` para ver el ranking global de rendimiento.

## 📚 Referencias y Fuentes

- **Stable-Baselines3:** https://stable-baselines3.readthedocs.io/
- **Gymnasium:** https://gymnasium.farama.org/
- **Scikit-Learn:** https://scikit-learn.org/

## 📝 Autor

Trabajo Fin de Grado (TFG) desarrollado por **Pablo Gámez Guerrero**.