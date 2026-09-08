"""
Procesamiento de series temporales y cálculo de variables técnicas.

Convierte datos OHLCV brutos en una matriz de variables estacionarias:
- Retornos logarítmicos (base de cálculos posteriores)
- Volatilidad histórica (riesgo de corto plazo)
- Medias móviles simples (referencia de tendencia)
- Distancias relativas a medias (variables técnicas estacionarias)

Todas las variables se calculan con datos disponibles al cierre del día t,
sin fuga de información futura.

Archivos de entrada:
  datos_mercado/<TICKER>.csv       series OHLCV descargadas

Archivos de salida:
  datos_procesados/<TICKER>_procesado.csv  matriz de variables técnicas
"""

import pandas as pd
import numpy as np
import os

carpeta_entrada = "datos_mercado"
carpeta_salida = "datos_procesados"

if not os.path.exists(carpeta_salida):
    os.makedirs(carpeta_salida)

# Los mismos activos que en descargar_datos.py para mantener consistencia
lista_activos = ["BTC-USD", "META", "SPY", "PEP", "TLT", "GLD"]

print("Iniciando procesamiento y cálculo de variables técnicas...")

for ticker in lista_activos:
    ruta_csv = os.path.join(carpeta_entrada, f"{ticker}.csv")
    
    if not os.path.exists(ruta_csv):
        print(f"No se encontró el archivo: {ruta_csv}")
        continue
    
    # Lectura con manejo de doble cabecera (formato típico de yfinance)
    try:
        df = pd.read_csv(ruta_csv, header=[0, 1], index_col=0, parse_dates=True)
        df.columns = df.columns.get_level_values(0)
    except Exception:
        # Si no tiene doble cabecera, lectura simple
        df = pd.read_csv(ruta_csv, index_col=0, parse_dates=True)
    
    # Conversión a numérico: elimina artefactos textuales de descarga defectuosa
    columnas_base = ["Close", "Open", "High", "Low", "Volume"]
    for col in columnas_base:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # === VARIABLES DE RENTABILIDAD Y RIESGO ===
    
    # Retorno logarítmico: base de todas las métricas financieras posteriores
    df["Retorno_Log"] = np.log(df["Close"] / df["Close"].shift(1))
    
    # Volatilidad histórica: desviación típica de retornos en ventana móvil de 20 días
    # (aproximadamente un mes de sesiones de mercado)
    df["Volatilidad_20"] = df["Retorno_Log"].rolling(window=20).std()
    
    # === VARIABLES DE TENDENCIA ===
    
    # Media móvil simple (SMA): suavizadora de corto plazo (20 días)
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    
    # Media móvil simple: suavizadora de largo plazo (50 días)
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    
    # Distancia relativa a la SMA de 50: medida estacionaria de posición respecto tendencia
    df["Distancia_SMA50"] = (df["Close"] - df["SMA_50"]) / df["SMA_50"]
    
    # Eliminación de valores nulos generados durante cálculos (periodos de warm-up)
    df = df.dropna()
    
    # Guardar con nombre normalizado (guiones sustituidos por guiones bajos)
    nombre_limpio = ticker.replace("-", "_")
    ruta_guardado = os.path.join(carpeta_salida, f"{nombre_limpio}_procesado.csv")
    df.to_csv(ruta_guardado)
    
    print(f"Procesado con éxito: {nombre_limpio}_procesado.csv ({len(df)} filas)")

print("Procesamiento completado. Carpeta lista.")