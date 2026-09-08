"""
Evaluacion retrospectiva de las senales generadas por los modelos supervisados.

Lee resultados/senales_supervisado.csv, simula la estrategia derivada de cada
senal descontando costes de transaccion y calcula el cuadro de metricas
financieras (ratio de Sharpe, ratio de Sortino, caida maxima del capital).

Se incluye siempre la referencia de comprar y mantener sobre el mismo tramo y
el mismo activo, que es el punto de comparacion honesto para un inversor.

Salidas:
  resultados/backtest_supervisado.csv   una fila por activo, modelo y particion
  resultados/curvas_capital_supervisado.csv  curvas de capital del tramo de prueba
"""

import os

import pandas as pd

from metricas_financieras import (
    COSTE_OPERACION,
    calcular_metricas,
    curva_capital,
    retornos_estrategia,
)

CARPETA_RESULTADOS = "resultados"
RUTA_SENALES = os.path.join(CARPETA_RESULTADOS, "senales_supervisado.csv")

COLUMNAS_RESUMEN = [
    "retorno_total", "cagr", "volatilidad_anual", "ratio_sharpe",
    "ratio_sortino", "caida_maxima", "ratio_calmar", "exposicion",
    "n_operaciones", "aciertos",
]


def main():
    """Punto de entrada: calcula metricas retrospectivas para modelos supervisados.
    
    Evalua cada modelo sobre cada activo y particion usando exactamente el mismo
    coste de transaccion que los modelos supervisados. Genera dos salidas CSV:
    - backtest_supervisado.csv: metricas por activo, modelo y particion
    - curvas_capital_supervisado.csv: curvas de equity para el tramo de prueba
    """
    if not os.path.exists(RUTA_SENALES):
        print(f"No se encontro {RUTA_SENALES}. Ejecuta antes modelos_supervisados.py")
        return

    senales = pd.read_csv(RUTA_SENALES, parse_dates=["fecha"])
    senales = senales.sort_values(["activo", "modelo", "particion", "fecha"])

    filas = []
    curvas = []

    # Agrupa por (activo, particion) para incluir referencia comprar y mantener
    for (activo, particion), bloque_particion in senales.groupby(
        ["activo", "particion"], sort=True
    ):
        # Referencia comprar y mantener: posicion larga permanente en el tramo.
        # Sirve como benchmark neutral para evaluar el valor anadido de cada modelo.
        referencia = (
            bloque_particion.drop_duplicates(subset="fecha").sort_values("fecha")
        )
        posiciones_bh = pd.Series(1.0, index=range(len(referencia)))
        netos_bh, pos_bh, rot_bh = retornos_estrategia(
            posiciones_bh.to_numpy(),
            referencia["retorno_futuro"].to_numpy(),
            coste_operacion=COSTE_OPERACION,
        )
        filas.append({
            "activo": activo,
            "modelo": "Comprar y mantener",
            "particion": particion,
            **calcular_metricas(netos_bh, pos_bh, rot_bh),
        })
        # Guarda la curva de capital solo para la particion de prueba
        if particion == "prueba":
            curvas.append(pd.DataFrame({
                "fecha": referencia["fecha"].to_numpy(),
                "activo": activo,
                "modelo": "Comprar y mantener",
                "capital": curva_capital(netos_bh),
            }))

        # Procesa cada modelo entrenado para este activo y particion
        for modelo, bloque in bloque_particion.groupby("modelo", sort=True):
            bloque = bloque.sort_values("fecha")
            netos, posiciones, rotacion = retornos_estrategia(
                bloque["senal"].to_numpy(),
                bloque["retorno_futuro"].to_numpy(),
                coste_operacion=COSTE_OPERACION,
            )
            filas.append({
                "activo": activo,
                "modelo": modelo,
                "particion": particion,
                **calcular_metricas(netos, posiciones, rotacion),
            })
            # Guarda la curva de capital solo para la particion de prueba
            if particion == "prueba":
                curvas.append(pd.DataFrame({
                    "fecha": bloque["fecha"].to_numpy(),
                    "activo": activo,
                    "modelo": modelo,
                    "capital": curva_capital(netos),
                }))

    tabla = pd.DataFrame(filas)
    ruta_tabla = os.path.join(CARPETA_RESULTADOS, "backtest_supervisado.csv")
    tabla.to_csv(ruta_tabla, index=False)

    tabla_curvas = pd.concat(curvas, ignore_index=True)
    ruta_curvas = os.path.join(CARPETA_RESULTADOS, "curvas_capital_supervisado.csv")
    tabla_curvas.to_csv(ruta_curvas, index=False)

    prueba = tabla[tabla["particion"] == "prueba"]

    print(f"Coste de transaccion aplicado: {COSTE_OPERACION * 10000:.1f} puntos basicos "
          "por unidad de rotacion")
    print("\nDetalle por activo (particion de prueba)")
    for activo, bloque in prueba.groupby("activo", sort=True):
        print(f"\n{activo}")
        detalle = bloque.set_index("modelo")[
            ["retorno_total", "ratio_sharpe", "ratio_sortino",
             "caida_maxima", "exposicion", "n_operaciones"]
        ]
        print(detalle.sort_values("ratio_sharpe", ascending=False).round(4).to_string())

    resumen = (
        prueba.groupby("modelo")[COLUMNAS_RESUMEN]
        .mean()
        .sort_values("ratio_sharpe", ascending=False)
    )
    print("\nMedia sobre los seis activos (particion de prueba)")
    print(resumen.round(4).to_string())

    print(f"\nTabla guardada en {ruta_tabla}")
    print(f"Curvas de capital guardadas en {ruta_curvas}")


if __name__ == "__main__":
    main()