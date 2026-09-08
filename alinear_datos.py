"""
Alineación temporal de series multivariantes.

Las series OHLCV descargadas de diferentes activos pueden tener fechas
ligeramente distintas debido a feriados, suspensiones, ajustes corporativos.

Este script encuentra las fechas exactas que comparten TODOS los activos
y recorta todas las series al mismo rango de fechas operativas comunes.

Resultado: matrices simétricas, listas para entrenar modelos sin
incompatibilidades de índices.

Archivos de entrada:
  datos_procesados/<TICKER>_procesado.csv

Archivos de salida:
  datos_procesados/<TICKER>_procesado.csv  (sobrescrito, alineado)
"""

import pandas as pd
import os

carpeta = "datos_procesados"
lista_activos = ["BTC_USD", "META", "SPY", "PEP", "TLT", "GLD"]

print("Iniciando alineación de matrices de datos...")

# === CARGA Y DESCUBRIMIENTO DE FECHAS COMUNES ===

diccionario_dfs = {}

# Cargar todos los archivos procesados en memoria
for ticker in lista_activos:
    ruta = os.path.join(carpeta, f"{ticker}_procesado.csv")
    diccionario_dfs[ticker] = pd.read_csv(ruta, index_col=0, parse_dates=True)

# Buscar la intersección de fechas: solo días que existen en TODOS los activos
fechas_comunes = diccionario_dfs[lista_activos[0]].index
for ticker in lista_activos[1:]:
    fechas_comunes = fechas_comunes.intersection(diccionario_dfs[ticker].index)

print(f"Se han encontrado {len(fechas_comunes)} días operativos comunes.")

# === RECORTE Y SOBRESCRITURA ===

# Aplicar el rango común a todos los activos y guardar
for ticker in lista_activos:
    df_alineado = diccionario_dfs[ticker].loc[fechas_comunes]
    ruta = os.path.join(carpeta, f"{ticker}_procesado.csv")
    df_alineado.to_csv(ruta)
    print(f"Activo {ticker} ajustado a {len(df_alineado)} filas.")

print("Alineación completada. Matrices simétricas listas para entrenar.")