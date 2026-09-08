# TFG Trading — Comparación de Aprendizaje Supervisado y DQN para Generación de Señales de Trading

Proyecto desarrollado como Trabajo de Fin de Grado en Ingeniería de Software en la Universidad de Málaga.

El objetivo del proyecto es comparar dos enfoques de Aprendizaje Automático para la generación de señales de trading sobre series financieras diarias:

- modelos de Aprendizaje Automático supervisado orientados a predecir el signo del retorno de la siguiente sesión;
- un agente de Aprendizaje por Refuerzo basado en Deep Q-Network (DQN), que aprende directamente una política de posicionamiento a partir de una recompensa económica.

La comparación se realiza mediante un procedimiento común de preparación de datos y backtesting sobre seis activos financieros, incorporando costes de transacción y utilizando una división cronológica fija de entrenamiento, validación y prueba.

---

## Objetivo del estudio

La rama supervisada formula el problema como una clasificación binaria:

- `1`: el retorno de la siguiente sesión es positivo;
- `0`: el retorno de la siguiente sesión es nulo o negativo.

Se estudian cuatro modelos:

- Regresión Logística;
- Random Forest;
- Máquinas de Vectores de Soporte (SVM);
- Perceptrón Multicapa (MLP).

El agente DQN aborda el problema desde una perspectiva secuencial. En cada instante puede:

- permanecer fuera del mercado (`0`);
- mantener una posición larga (`1`).

El objetivo del trabajo no es demostrar que un paradigma sea universalmente superior al otro, sino estudiar si optimizar una tarea de clasificación y optimizar una recompensa económica conducen a comportamientos financieros diferentes.

---

## Activos utilizados

El estudio se realiza de forma independiente sobre seis activos:

| Ticker | Activo | Tipo |
|---|---|---|
| `BTC-USD` | Bitcoin | Criptoactivo |
| `META` | Meta Platforms | Renta variable tecnológica |
| `SPY` | SPDR S&P 500 ETF Trust | Renta variable estadounidense |
| `PEP` | PepsiCo | Renta variable defensiva |
| `TLT` | iShares 20+ Year Treasury Bond ETF | Renta fija de larga duración |
| `GLD` | SPDR Gold Shares | Oro |

Los datos diarios se descargan mediante `yfinance` desde el 1 de enero de 2015.

---

## Variables utilizadas

Ambas ramas utilizan las mismas doce características financieras construidas a partir de datos OHLCV:

1. retorno logarítmico de una sesión;
2. retorno acumulado de 5 sesiones;
3. retorno acumulado de 10 sesiones;
4. retorno normalizado por volatilidad;
5. volatilidad histórica de 20 sesiones;
6. cambio relativo de la volatilidad;
7. distancia respecto a la SMA de 20 sesiones;
8. distancia respecto a la SMA de 50 sesiones;
9. cruce relativo entre SMA20 y SMA50;
10. RSI de 14 sesiones;
11. rango intradía relativo;
12. volumen relativo respecto a su media móvil.

El DQN incorpora además la posición actual del agente, por lo que su estado tiene 13 componentes.

Los niveles absolutos de precio no se utilizan directamente como variables de entrada.

---

## División temporal de los datos

Los datos se dividen cronológicamente en:

- 70 % entrenamiento;
- 15 % validación;
- 15 % prueba.

No se utiliza división aleatoria ni validación `walk-forward`.

El ajuste del `StandardScaler` se realiza únicamente sobre la partición de entrenamiento para evitar fuga de información. Las variables estandarizadas se limitan posteriormente al intervalo `[-5, 5]`.

---

## Costes de transacción

La evaluación financiera utiliza un coste nominal de:

```text
0.0005 = 5 puntos básicos
```

por cada cambio completo de posición.

El coste se aplica proporcionalmente a:

```text
|posición_actual - posición_anterior|
```

Por tanto:

- mantener una posición no genera un nuevo coste;
- entrar al mercado genera un coste;
- salir del mercado genera un coste.

La estrategia de comprar y mantener también incorpora el coste correspondiente a la entrada inicial.

### Penalización adicional durante el entrenamiento del DQN

Durante el entrenamiento del agente DQN se utiliza una penalización amplificada por rotación:

```text
lambda = 15
```

La recompensa utilizada por el entorno es:

```text
recompensa =
    posicion_nueva * retorno
    - rotacion * coste_operacion * lambda
```

Por tanto, un cambio de posición recibe internamente durante el entrenamiento una penalización equivalente a:

```text
15 × 0.0005 = 0.0075
```

es decir, 75 puntos básicos.

Este valor **no representa un coste real de transacción**. Se utiliza únicamente como mecanismo de *reward shaping* para desincentivar una operativa excesivamente frecuente.

La evaluación final de todas las estrategias continúa utilizando el coste nominal común de 5 puntos básicos.

---

## Requisitos

### Python

Se recomienda utilizar Python 3.10 o superior.

Instala las dependencias con:

```bash
pip install -r requirements.txt
```

Entre las principales librerías utilizadas se encuentran:

- NumPy;
- pandas;
- scikit-learn;
- Stable-Baselines3;
- Gymnasium;
- PyTorch;
- matplotlib;
- yfinance.

---

## Ejecución del proyecto

El proyecto está organizado como un pipeline secuencial.

### 1. Descarga de datos

```bash
python descargar_datos.py
```

Genera:

```text
datos_mercado/<TICKER>.csv
```

### 2. Procesamiento de datos

```bash
python procesar_datos.py
```

Genera:

```text
datos_procesados/<ACTIVO>_procesado.csv
```

### 3. Alineamiento temporal

```bash
python alinear_datos.py
```

Alinea todos los activos sobre un calendario temporal común.

### 4. Entrenamiento de modelos supervisados

```bash
python modelos_supervisados.py
```

Entrena:

- Regresión Logística;
- Random Forest;
- SVM;
- MLP.

Genera:

```text
datos_features/<ACTIVO>_features.csv
resultados/metricas_supervisado.csv
resultados/senales_supervisado.csv
```

### 5. Backtesting de los modelos supervisados

```bash
python backtest_supervisado.py
```

Genera:

```text
resultados/backtest_supervisado.csv
resultados/curvas_capital_supervisado.csv
```

### 6. Entrenamiento y evaluación del agente DQN

```bash
python agente_dqn.py
```

La arquitectura contiene dos capas ocultas:

```text
64 -> 32
```

Parámetros principales:

```text
learning_rate = 5e-4
buffer_size = 50000
learning_starts = 1000
batch_size = 64
gamma = 0.99
train_freq = 4
target_update_interval = 1000
exploration_fraction = 0.30
exploration_final_eps = 0.05
max_grad_norm = 10
```

Se utilizan tres semillas:

```text
0, 1, 2
```

y un máximo de:

```text
30000 pasos
```

por activo y semilla.

La selección del modelo se realiza sobre validación. Se conserva el modelo con mejor recompensa media de evaluación y se aplica parada anticipada con:

```text
min_evals = 8
max_no_improvement_evals = 6
```

La partición de prueba no participa en el entrenamiento ni en la selección del modelo.

Ejemplo de ejecución:

```bash
python agente_dqn.py \
    --activos SPY BTC_USD META PEP TLT GLD \
    --semillas 0 1 2 \
    --pasos 30000
```

Genera:

```text
resultados/metricas_dqn.csv
resultados/backtest_dqn.csv
resultados/senales_dqn.csv
resultados/curvas_capital_dqn.csv
resultados/modelos_dqn/
```

### 7. Generación de gráficos

```bash
python visualizador.py
```

Genera:

```text
resultados/grafico_supervisado.svg
resultados/grafico_supervisado.pdf
resultados/grafico_dqn.svg
resultados/grafico_dqn.pdf
resultados/grafico_comparativo.svg
resultados/grafico_comparativo.pdf
```

---

## Estructura del proyecto

```text
tfg-trading/
│
├── datos_mercado/
├── datos_procesados/
├── datos_features/
├── resultados/
│   ├── backtest_supervisado.csv
│   ├── backtest_dqn.csv
│   ├── metricas_supervisado.csv
│   ├── metricas_dqn.csv
│   ├── senales_supervisado.csv
│   ├── senales_dqn.csv
│   ├── curvas_capital_supervisado.csv
│   ├── curvas_capital_dqn.csv
│   ├── modelos_dqn/
│   └── gráficos SVG/PDF
│
├── descargar_datos.py
├── procesar_datos.py
├── alinear_datos.py
├── modelos_supervisados.py
├── backtest_supervisado.py
├── entorno_trading.py
├── agente_dqn.py
├── metricas_financieras.py
├── visualizador.py
├── requirements.txt
└── README.md
```

---

## Métricas de evaluación

Las estrategias se comparan mediante:

- retorno total;
- CAGR;
- volatilidad anualizada;
- ratio de Sharpe;
- ratio de Sortino;
- máxima caída (*Maximum Drawdown*);
- ratio de Calmar;
- exposición al mercado;
- número de operaciones;
- proporción de sesiones ganadoras mientras existe exposición.

Para los clasificadores supervisados también se analiza la capacidad predictiva mediante métricas de clasificación.

---

## Estrategia de referencia

Se utiliza una estrategia de `Buy & Hold` como referencia financiera.

Esta estrategia permanece invertida durante la partición evaluada y se somete al mismo modelo nominal de costes de transacción.

---

## Ejecución completa

```bash
python descargar_datos.py
python procesar_datos.py
python alinear_datos.py
python modelos_supervisados.py
python backtest_supervisado.py
python agente_dqn.py --activos SPY BTC_USD META PEP TLT GLD --semillas 0 1 2 --pasos 30000
python visualizador.py
```

Los resultados se almacenan en:

```text
resultados/
```

---

## Consideraciones metodológicas

Este repositorio corresponde a un estudio experimental retrospectivo.

Los resultados deben interpretarse dentro de las condiciones concretas del experimento:

- seis activos;
- frecuencia diaria;
- una única división cronológica 70/15/15;
- ausencia de posiciones cortas;
- posición binaria: fuera del mercado o posición larga;
- coste nominal fijo de 5 puntos básicos;
- ausencia de dividendos;
- ausencia de impacto de mercado;
- ausencia de spread variable;
- ausencia de validación `walk-forward`;
- tres semillas para el agente DQN.

Los resultados no deben interpretarse como evidencia de superioridad general de una técnica ni como una recomendación de inversión.

---

## Reproducibilidad

Para reproducir el experimento se recomienda:

1. crear un entorno virtual de Python;
2. instalar `requirements.txt`;
3. ejecutar los scripts en el orden indicado;
4. mantener las semillas especificadas;
5. utilizar los mismos datos y parámetros experimentales.

---

## Referencias técnicas

- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [Gymnasium](https://gymnasium.farama.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [yfinance](https://github.com/ranaroussi/yfinance)

---

## Autor

**Pablo Gámez Guerrero**

Trabajo de Fin de Grado — Grado en Ingeniería de Software  
Escuela Técnica Superior de Ingeniería Informática  
Universidad de Málaga

**Tutor:** Francisco de Asís Fernández Navarro
