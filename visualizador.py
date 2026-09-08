"""
Visualización de resultados de backtesting para modelos supervisados y DQN.

Genera gráficos comparativos de la evolución del capital, curvas de rentabilidad
y métricas de rendimiento (Ratio Sharpe) entre diferentes estrategias de trading.

Puntos clave del diseño:

1. Entrada: archivos CSV generados por los scripts de backtesting:
   curvas_capital_supervisado.csv: evolución diaria del capital para modelos
     de aprendizaje supervisado (Regresión Logística, Random Forest, MLP).
   curvas_capital_dqn.csv: evolución diaria del capital para agentes DQN
     entrenados con diferentes semillas.
   backtest_supervisado.csv: métricas agregadas de supervisado.
   backtest_dqn.csv: métricas agregadas de DQN.

2. Salidas: gráficos SVG y PDF en la carpeta 'resultados/':
   grafico_supervisado.svg/pdf: grid 3x2 con evolución de capital por activo
     para modelos supervisados.
   grafico_dqn.svg/pdf: grid 3x2 con evolución de capital por activo para DQN.
   grafico_comparativo.svg/pdf: ranking horizontal de Ratio Sharpe medio.

3. Todas las líneas incluyen "Comprar y mantener" como referencia, representada
   con una línea negra punteada.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuración global de estilo y rutas
sns.set_style("whitegrid")
RESULTADOS_DIR = "resultados"
ACTIVOS = ["SPY", "GLD", "TLT", "META", "PEP", "BTC_USD"]

# Paletas de colores
COLORES_SUPERVISADO = {
    "MLP": "#EE3333",
    "Regresión Logística": "#33AA33",
    "Random Forest": "#3366FF",
}

COLORES_DQN = {
    "DQN (seed 0)": "#FF9933",
    "DQN (seed 1)": "#9933FF",
    "DQN (seed 2)": "#996633",
}

COLOR_BYH = "#000000"


def crear_carpeta_resultados():
    """Crea la carpeta de resultados si no existe."""
    Path(RESULTADOS_DIR).mkdir(exist_ok=True)


def cargar_csv(ruta, nombre_archivo):
    """Carga un CSV con manejo de errores."""
    try:
        df = pd.read_csv(ruta)
        print(f"✓ Cargado {nombre_archivo}: {len(df)} filas")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Archivo no encontrado: {nombre_archivo}")
    except Exception as e:
        raise Exception(f"Error al leer {nombre_archivo}: {e}")


def ordenar_por_fecha(df):
    """Ordena el dataframe por fecha."""
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"])
        df = df.sort_values("fecha").reset_index(drop=True)
    return df


def grafico_supervisado(curvas_df):
    """Genera el gráfico de supervisado (3x2 grid)."""
    curvas_df = ordenar_por_fecha(curvas_df.copy())
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, activo in enumerate(ACTIVOS):
        ax = axes[idx]
        datos_activo = curvas_df[curvas_df["activo"] == activo]
        
        # Separar Comprar y mantener del resto
        byh = datos_activo[datos_activo["modelo"] == "Comprar y mantener"]
        otros = datos_activo[datos_activo["modelo"] != "Comprar y mantener"]
        
        # Plotear modelos normales
        for modelo in otros["modelo"].unique():
            datos_modelo = otros[otros["modelo"] == modelo]
            color = COLORES_SUPERVISADO.get(modelo, "#666666")
            ax.plot(
                datos_modelo["fecha"],
                datos_modelo["capital"],
                label=modelo,
                color=color,
                linewidth=1.5,
                alpha=0.8,
            )
        
        # Plotear Comprar y mantener (línea negra punteada gruesa)
        if len(byh) > 0:
            ax.plot(
                byh["fecha"],
                byh["capital"],
                label="Comprar y mantener",
                color=COLOR_BYH,
                linewidth=2.5,
                linestyle="--",
                alpha=0.9,
            )
        
        ax.set_title(f"Evolución de capital: {activo}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Fecha", fontsize=10)
        ax.set_ylabel("Capital", fontsize=10)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar
    svg_path = os.path.join(RESULTADOS_DIR, "grafico_supervisado.svg")
    pdf_path = os.path.join(RESULTADOS_DIR, "grafico_supervisado.pdf")
    fig.savefig(svg_path, format="svg", dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"✓ Gráfico supervisado guardado en {svg_path} y {pdf_path}")


def grafico_dqn(curvas_df):
    """Genera el gráfico de DQN (3x2 grid)."""
    curvas_df = ordenar_por_fecha(curvas_df.copy())
    
    # Fusionar nombre de modelo y semilla para dibujar lineas separadas
    if "semilla" in curvas_df.columns:
        curvas_df["modelo_plot"] = curvas_df.apply(
            lambda r: f"{r['modelo']} (seed {int(r['semilla'])})" if pd.notna(r['semilla']) and r['modelo'] != "Comprar y mantener" else r['modelo'],
            axis=1
        )
    else:
        curvas_df["modelo_plot"] = curvas_df["modelo"]
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    # Extraer dinámicamente los modelos DQN únicos combinados
    modelos_dqn = curvas_df[curvas_df["modelo_plot"] != "Comprar y mantener"]["modelo_plot"].unique()
    colores_dinamicos = {}
    colores_predefinidos = list(COLORES_DQN.values())
    for i, modelo in enumerate(sorted(modelos_dqn)):
        colores_dinamicos[modelo] = colores_predefinidos[i % len(colores_predefinidos)]
    
    for idx, activo in enumerate(ACTIVOS):
        ax = axes[idx]
        datos_activo = curvas_df[curvas_df["activo"] == activo]
        
        # Separar Comprar y mantener del resto
        byh = datos_activo[datos_activo["modelo_plot"] == "Comprar y mantener"]
        otros = datos_activo[datos_activo["modelo_plot"] != "Comprar y mantener"]
        
        # Plotear modelos DQN separados
        for modelo in sorted(otros["modelo_plot"].unique()):
            datos_modelo = otros[otros["modelo_plot"] == modelo]
            color = colores_dinamicos.get(modelo, "#666666")
            ax.plot(
                datos_modelo["fecha"],
                datos_modelo["capital"],
                label=modelo,
                color=color,
                linewidth=1.5,
                alpha=0.8,
            )
        
        # Plotear Comprar y mantener (línea negra punteada gruesa)
        if len(byh) > 0:
            ax.plot(
                byh["fecha"],
                byh["capital"],
                label="Comprar y mantener",
                color=COLOR_BYH,
                linewidth=2.5,
                linestyle="--",
                alpha=0.9,
            )
        
        ax.set_title(f"Evolución de capital: {activo}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Fecha", fontsize=10)
        ax.set_ylabel("Capital", fontsize=10)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar
    svg_path = os.path.join(RESULTADOS_DIR, "grafico_dqn.svg")
    pdf_path = os.path.join(RESULTADOS_DIR, "grafico_dqn.pdf")
    fig.savefig(svg_path, format="svg", dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"✓ Gráfico DQN guardado en {svg_path} y {pdf_path}")


def grafico_comparativo(backtest_sup, backtest_dqn):
    """Genera el gráfico comparativo horizontal (barras ordenadas)."""
    # Filtrar solo partición de prueba
    sup_prueba = backtest_sup[backtest_sup["particion"] == "prueba"].copy()
    dqn_prueba = backtest_dqn[backtest_dqn["particion"] == "prueba"].copy()
    
    # Normalizar nombres de columnas: detectar dinámicamente el nombre de Sharpe
    # en cada dataframe y renombrarlo a "ratio_sharpe"
    
    # Para supervisado
    if "ratio_sharpe_media" in sup_prueba.columns:
        sup_prueba.rename(columns={"ratio_sharpe_media": "ratio_sharpe"}, inplace=True)
    elif "ratio_sharpe" not in sup_prueba.columns:
        raise ValueError("No se encontró columna de Sharpe en backtest_supervisado")
    
    # Para DQN
    if "ratio_sharpe_media" in dqn_prueba.columns:
        dqn_prueba.rename(columns={"ratio_sharpe_media": "ratio_sharpe"}, inplace=True)
    elif "ratio_sharpe" not in dqn_prueba.columns:
        raise ValueError("No se encontró columna de Sharpe en backtest_dqn")
    
    # Combinar ambos con columnas normalizadas
    combined = pd.concat([sup_prueba, dqn_prueba], ignore_index=True)
    
    # Desduplicar "Comprar y mantener" manteniendo registros únicos por activo y modelo
    combined = combined.drop_duplicates(subset=["modelo", "activo", "particion"])
    
    # Agrupar por modelo y calcular media del ratio_sharpe normalizado
    media_sharpe = combined.groupby("modelo")["ratio_sharpe"].mean().sort_values(ascending=True)
    # Sanitizar NaN a 0 para que las barras se representen correctamente
    media_sharpe = media_sharpe.fillna(0.0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Colores para la barra comparativa
    colores_barras = []
    for modelo in media_sharpe.index:
        if modelo == "Comprar y mantener":
            colores_barras.append(COLOR_BYH)
        elif modelo in COLORES_SUPERVISADO:
            colores_barras.append(COLORES_SUPERVISADO[modelo])
        elif modelo in COLORES_DQN:
            colores_barras.append(COLORES_DQN[modelo])
        else:
            colores_barras.append("#999999")
    
    # Gráfico de barras horizontal
    bars = ax.barh(
        range(len(media_sharpe)),
        media_sharpe.values,
        color=colores_barras,
        alpha=0.8,
        edgecolor="black",
        linewidth=1.2,
    )
    
    ax.set_yticks(range(len(media_sharpe)))
    ax.set_yticklabels(media_sharpe.index, fontsize=11)
    ax.set_xlabel("Ratio Sharpe (media entre activos)", fontsize=12, fontweight="bold")
    ax.set_title("Comparativa Global: Ratio Sharpe por Modelo (Partición Prueba)", 
                fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    
    # Añadir valores en las barras
    for i, v in enumerate(media_sharpe.values):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=10, fontweight="bold")
    
    plt.tight_layout()
    
    # Guardar
    svg_path = os.path.join(RESULTADOS_DIR, "grafico_comparativo.svg")
    pdf_path = os.path.join(RESULTADOS_DIR, "grafico_comparativo.pdf")
    fig.savefig(svg_path, format="svg", dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"✓ Gráfico comparativo guardado en {svg_path} y {pdf_path}")
    print(f"\nRanking (media Sharpe en partición prueba):")
    for modelo, sharpe in media_sharpe.items():
        print(f"  {modelo:30s} {sharpe:7.4f}")


def main():
    """Función principal."""
    print("="*60)
    print("TFG Trading — Visualizador de Resultados")
    print("="*60)
    
    try:
        # Crear carpeta de resultados
        crear_carpeta_resultados()
        
        # Cargar archivos CSV
        print("\n1. Cargando archivos CSV...")
        curvas_sup = cargar_csv(
            os.path.join(RESULTADOS_DIR, "curvas_capital_supervisado.csv"),
            "curvas_capital_supervisado.csv"
        )
        curvas_dqn = cargar_csv(
            os.path.join(RESULTADOS_DIR, "curvas_capital_dqn.csv"),
            "curvas_capital_dqn.csv"
        )
        backtest_sup = cargar_csv(
            os.path.join(RESULTADOS_DIR, "backtest_supervisado.csv"),
            "backtest_supervisado.csv"
        )
        backtest_dqn = cargar_csv(
            os.path.join(RESULTADOS_DIR, "backtest_dqn.csv"),
            "backtest_dqn.csv"
        )
        
        # Generar gráficos
        print("\n2. Generando gráficos...")
        grafico_supervisado(curvas_sup)
        grafico_dqn(curvas_dqn)
        grafico_comparativo(backtest_sup, backtest_dqn)
        
        print("\n" + "="*60)
        print("✓ Visualización completada exitosamente")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nArchivos requeridos en la carpeta 'resultados/':")
        print("  resultados/curvas_capital_supervisado.csv")
        print("  resultados/curvas_capital_dqn.csv")
        print("  resultados/backtest_supervisado.csv")
        print("  resultados/backtest_dqn.csv")
        exit(1)
    except Exception as e:
        print(f"\nError inesperado: {e}")
        exit(1)


if __name__ == "__main__":
    main()