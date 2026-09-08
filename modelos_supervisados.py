"""
Entrenamiento y evaluacion de modelos de Aprendizaje Automatico supervisado
para la generacion de senales de trading.

Puntos clave del diseno experimental:

1. Todas las variables de entrada son ESTACIONARIAS (retornos, ratios y
   distancias relativas). No se usan niveles absolutos de precio como las
   medias moviles SMA_20 / SMA_50, porque su rango crece con el tiempo y la
   estandarizacion ajustada en el tramo de entrenamiento genera valores
   desbordados en el tramo de prueba (origen de los RuntimeWarning de
   overflow / invalid value en las multiplicaciones matriciales).

2. La variable objetivo es el signo del retorno logaritmico del dia SIGUIENTE.
   Las variables se calculan con informacion disponible hasta el cierre del dia t
   y predicen el retorno de t+1, de modo que no existe fuga de informacion futura.

3. Division estrictamente cronologica en entrenamiento, validacion y prueba.
   El escalador se ajusta unicamente con el tramo de entrenamiento.

Salidas generadas:
  datos_features/<ACTIVO>_features.csv          matriz de variables reutilizable
  resultados/metricas_supervisado.csv           metricas de clasificacion
  resultados/senales_supervisado.csv            senales para la evaluacion retrospectiva
"""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CARPETA_DATOS = "datos_procesados"
CARPETA_FEATURES = "datos_features"
CARPETA_RESULTADOS = "resultados"

ACTIVOS = ["BTC_USD", "META", "SPY", "PEP", "TLT", "GLD"]

PROPORCION_TRAIN = 0.70
PROPORCION_VALIDACION = 0.15
SEMILLA = 42

# Limite de recorte de las variables ya estandarizadas (en desviaciones tipicas).
# Protege frente a valores extremos del tramo de prueba y evita desbordamientos.
LIMITE_Z = 5.0

VARIABLES = [
    "ret_1",
    "ret_5",
    "ret_10",
    "ret_norm",
    "vol_20",
    "cambio_vol",
    "dist_sma20",
    "dist_sma50",
    "cruce_sma",
    "rsi_14",
    "rango_rel",
    "volumen_rel",
]


def calcular_rsi(cierre, periodo=14):
    """Indice de fuerza relativa (RSI) con suavizado exponencial de Wilder.
    
    Oscila entre 0 y 100. Detecta condiciones de sobrecompra (RSI > 70)
    y sobreventa (RSI < 30). Devuelve valores como fraccion [0, 1] (dividido por 100).
    """
    diferencia = cierre.diff()
    ganancia = diferencia.clip(lower=0.0)
    perdida = -diferencia.clip(upper=0.0)

    media_ganancia = ganancia.ewm(alpha=1.0 / periodo, adjust=False,
                                  min_periods=periodo).mean()
    media_perdida = perdida.ewm(alpha=1.0 / periodo, adjust=False,
                                min_periods=periodo).mean()

    # Se evita la division por cero sustituyendo el denominador nulo por NaN.
    fuerza_relativa = media_ganancia / media_perdida.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + fuerza_relativa))

    # Denominador nulo significa ausencia de perdidas: RSI saturado en 100.
    return rsi.where(media_perdida > 0.0, 100.0)


def construir_features(df):
    """Genera la matriz de variables estacionarias y la variable objetivo.
    
    Entrada: DataFrame con OHLCV sin procesar.
    Salida: DataFrame limpio sin NaN, lista para entrenamiento.
    
    Las 12 variables se construyen a partir de:
    - Retornos logaritmicos acumulados (multiples horizontes)
    - Volatilidad historica y su cambio
    - Distancias relativas a medias moviles
    - RSI normalizado
    - Rango y volumen relativos
    
    Objetivo: signo del retorno logaritmico del dia siguiente (0/1).
    """
    datos = df.copy()

    # Retornos logaritmicos acumulados a distintos horizontes.
    datos["ret_1"] = datos["Retorno_Log"]
    datos["ret_5"] = datos["Retorno_Log"].rolling(window=5).sum()
    datos["ret_10"] = datos["Retorno_Log"].rolling(window=10).sum()

    # Volatilidad historica y su variacion relativa.
    datos["vol_20"] = datos["Volatilidad_20"]
    volatilidad_previa = datos["vol_20"].shift(5).replace(0.0, np.nan)
    datos["cambio_vol"] = datos["vol_20"] / volatilidad_previa - 1.0

    # Retorno normalizado por volatilidad (momento ajustado al riesgo).
    datos["ret_norm"] = datos["ret_1"] / datos["vol_20"].replace(0.0, np.nan)

    # Distancias relativas a las medias moviles: estacionarias, a diferencia
    # de las propias medias moviles en nivel.
    datos["dist_sma20"] = (datos["Close"] - datos["SMA_20"]) / datos["SMA_20"]
    datos["dist_sma50"] = datos["Distancia_SMA50"]
    datos["cruce_sma"] = datos["SMA_20"] / datos["SMA_50"] - 1.0

    datos["rsi_14"] = calcular_rsi(datos["Close"], periodo=14) / 100.0

    # Amplitud diaria relativa al cierre.
    datos["rango_rel"] = (datos["High"] - datos["Low"]) / datos["Close"]

    # Volumen relativo a su media movil de 20 sesiones.
    media_volumen = datos["Volume"].rolling(window=20).mean().replace(0.0, np.nan)
    datos["volumen_rel"] = datos["Volume"] / media_volumen - 1.0

    # Objetivo: signo del retorno logaritmico del dia siguiente.
    datos["retorno_futuro"] = datos["Retorno_Log"].shift(-1)
    datos["objetivo"] = (datos["retorno_futuro"] > 0).astype("int64")

    columnas = VARIABLES + ["retorno_futuro", "objetivo", "Close"]
    datos = datos[columnas]
    datos = datos.replace([np.inf, -np.inf], np.nan).dropna()
    return datos


def crear_modelos():
    """Instancias nuevas de cada modelo (evita reutilizar estado entre activos).
    
    Devuelve un diccionario con modelos de clasificacion supervisada:
    - Regresion Logistica: baseline lineal
    - Random Forest: captura interacciones
    - SVM: separacion no lineal
    - MLP: red neuronal (misma arquitectura que la rama de RL)
    
    Todos usan class_weight="balanced" para manejar desbalance de clases.
    """
    return {
        "Regresion Logistica": LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=SEMILLA
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=SEMILLA,
            n_jobs=-1,
        ),
        "SVM": SVC(
            C=1.0, kernel="rbf", gamma="scale",
            class_weight="balanced", random_state=SEMILLA
        ),
        # Mas iteraciones y mayor penalizacion L2: la superficie de error es muy
        # plana (senal cercana al azar) y con 1000 iteraciones el optimizador
        # estocastico no alcanzaba el criterio de parada.
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            alpha=1e-2,
            max_iter=3000,
            tol=1e-4,
            n_iter_no_change=30,
            random_state=SEMILLA,
        ),
    }


def dividir_cronologicamente(datos):
    """Division temporal sin mezclar observaciones.
    
    Respeta el orden cronologico: entrenamiento -> validacion -> prueba.
    No hay fuga de informacion futura. Las proporciones son:
    - 70% entrenamiento
    - 15% validacion
    - 15% prueba
    """
    total = len(datos)
    fin_train = int(total * PROPORCION_TRAIN)
    fin_validacion = int(total * (PROPORCION_TRAIN + PROPORCION_VALIDACION))
    return (
        datos.iloc[:fin_train],
        datos.iloc[fin_train:fin_validacion],
        datos.iloc[fin_validacion:],
    )


def evaluar(y_real, y_predicho):
    """Calcula metricas de clasificacion binaria.
    
    Devuelve exactitud global, exactitud equilibrada (macro),
    y metricas de calidad sobre la clase positiva (alza):
    precision, sensibilidad (recall) y F1-score.
    """
    return {
        "exactitud": accuracy_score(y_real, y_predicho),
        "exactitud_equilibrada": balanced_accuracy_score(y_real, y_predicho),
        "precision_alza": precision_score(y_real, y_predicho, pos_label=1,
                                          zero_division=0),
        "sensibilidad_alza": recall_score(y_real, y_predicho, pos_label=1,
                                          zero_division=0),
        "f1_alza": f1_score(y_real, y_predicho, pos_label=1, zero_division=0),
    }


def main():
    """Punto de entrada: entrena modelos supervisados y genera senales.
    
    Para cada activo:
    1. Carga datos procesados y construye matriz de features estacionarias
    2. Divide cronologicamente en train/val/test
    3. Estandariza con el tramo de entrenamiento (sin fuga de informacion)
    4. Entrena cada modelo
    5. Genera senales para evaluacion retrospectiva
    6. Reporta metricas de clasificacion
    """
    os.makedirs(CARPETA_FEATURES, exist_ok=True)
    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

    filas_metricas = []
    filas_senales = []

    print("Entrenamiento de modelos supervisados")
    print(f"Division cronologica: {int(PROPORCION_TRAIN * 100)}% entrenamiento / "
          f"{int(PROPORCION_VALIDACION * 100)}% validacion / "
          f"{int((1 - PROPORCION_TRAIN - PROPORCION_VALIDACION) * 100)}% prueba")

    for activo in ACTIVOS:
        ruta = os.path.join(CARPETA_DATOS, f"{activo}_procesado.csv")
        if not os.path.exists(ruta):
            print(f"\n[AVISO] No se encontro {ruta}. Omitido.")
            continue

        df = pd.read_csv(ruta, index_col=0, parse_dates=True)
        datos = construir_features(df)
        datos.to_csv(os.path.join(CARPETA_FEATURES, f"{activo}_features.csv"))

        train, validacion, prueba = dividir_cronologicamente(datos)

        print(f"\nActivo {activo}: {len(datos)} observaciones "
              f"({train.index[0].date()} a {datos.index[-1].date()})")
        print(f"  entrenamiento {len(train)} | validacion {len(validacion)} "
              f"| prueba {len(prueba)}")

        escalador = StandardScaler().fit(train[VARIABLES])

        def preparar(particion):
            matriz = escalador.transform(particion[VARIABLES])
            return np.clip(matriz, -LIMITE_Z, LIMITE_Z)

        X_train = preparar(train)
        X_validacion = preparar(validacion)
        X_prueba = preparar(prueba)

        y_train = train["objetivo"].to_numpy()
        y_validacion = validacion["objetivo"].to_numpy()
        y_prueba = prueba["objetivo"].to_numpy()

        # Referencia ingenua: mantener siempre posicion larga.
        proporcion_alzas = float(y_prueba.mean())
        print(f"  referencia 'siempre alcista' en prueba: {proporcion_alzas:.4f}")
        filas_metricas.append({
            "activo": activo, "modelo": "Siempre alcista", "particion": "prueba",
            "n": len(y_prueba),
            **evaluar(y_prueba, np.ones_like(y_prueba)),
        })

        for nombre, modelo in crear_modelos().items():
            modelo.fit(X_train, y_train)

            predicciones = {
                "validacion": (validacion, y_validacion,
                               modelo.predict(X_validacion)),
                "prueba": (prueba, y_prueba, modelo.predict(X_prueba)),
            }

            metricas_por_particion = {}

            for particion, (marco, y_real, y_predicho) in predicciones.items():
                metricas = evaluar(y_real, y_predicho)
                metricas_por_particion[particion] = metricas
                filas_metricas.append({
                    "activo": activo, "modelo": nombre, "particion": particion,
                    "n": len(y_real), **metricas,
                })

                senales = pd.DataFrame({
                    "fecha": marco.index,
                    "activo": activo,
                    "modelo": nombre,
                    "particion": particion,
                    "senal": y_predicho,
                    "retorno_futuro": marco["retorno_futuro"].to_numpy(),
                    "precio_cierre": marco["Close"].to_numpy(),
                })
                filas_senales.append(senales)

            metricas_prueba = metricas_por_particion["prueba"]
            metricas_validacion = metricas_por_particion["validacion"]
            print(f"  {nombre:<20} val {metricas_validacion['exactitud']:.4f} | "
                  f"prueba {metricas_prueba['exactitud']:.4f} | "
                  f"equilibrada {metricas_prueba['exactitud_equilibrada']:.4f}")

    if not filas_metricas:
        print("\nNo se proceso ningun activo. Revisa la carpeta de datos.")
        return

    tabla_metricas = pd.DataFrame(filas_metricas)
    ruta_metricas = os.path.join(CARPETA_RESULTADOS, "metricas_supervisado.csv")
    tabla_metricas.to_csv(ruta_metricas, index=False)

    tabla_senales = pd.concat(filas_senales, ignore_index=True)
    ruta_senales = os.path.join(CARPETA_RESULTADOS, "senales_supervisado.csv")
    tabla_senales.to_csv(ruta_senales, index=False)

    resumen = (
        tabla_metricas[tabla_metricas["particion"] == "prueba"]
        .groupby("modelo")[["exactitud", "exactitud_equilibrada", "f1_alza"]]
        .mean()
        .sort_values("exactitud_equilibrada", ascending=False)
    )
    print("\nMedia sobre los seis activos (particion de prueba):")
    print(resumen.round(4).to_string())

    print(f"\nMetricas guardadas en {ruta_metricas}")
    print(f"Senales guardadas en {ruta_senales}")


if __name__ == "__main__":
    main()