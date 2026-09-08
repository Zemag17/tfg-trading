"""
Descarga de series temporales OHLCV desde Yahoo Finance.

Descarga datos históricos de seis activos (Bitcoin, ETF de renta variable,
renta fija y acciones individuales) desde 2015 hasta la fecha actual.
Los datos se guardan en formato CSV para procesamiento posterior.

Archivo de salida:
  datos_mercado/<TICKER>.csv   series temporales OHLCV sin procesar
"""

import yfinance as yf
import pandas as pd
import os

# Directorio de almacenamiento para datos sin procesar
carpeta_salida = "datos_mercado"
if not os.path.exists(carpeta_salida):
    os.makedirs(carpeta_salida)

# Activos seleccionados: cobertura multisectorial y correlaciones diversas
lista_activos = ["BTC-USD", "META", "SPY", "PEP", "TLT", "GLD"]
fecha_inicio = "2015-01-01"

print("Iniciando descarga de series temporales OHLCV...")

for ticker in lista_activos:
    print(f"Obteniendo datos de {ticker}...")
    
    # Descarga desde Yahoo Finance: genera OHLCV (Open, High, Low, Close, Volume)
    dataframe = yf.download(ticker, start=fecha_inicio)
    
    # Eliminación de filas sin datos: pueden existir fechas sin sesión de mercado
    dataframe = dataframe.dropna()
    
    # Guardar en CSV para etapas posteriores de procesamiento
    ruta_csv = os.path.join(carpeta_salida, f"{ticker}.csv")
    dataframe.to_csv(ruta_csv)
    
    print(f"Archivo guardado correctamente: {ruta_csv}")

print("Extracción finalizada. Series temporales listas para procesar.")