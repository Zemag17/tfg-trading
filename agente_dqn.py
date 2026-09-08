"""
Entrenamiento y evaluacion de un agente DQN de decisiones de trading.

Protocolo experimental
----------------------
1. Entrenamiento en la particion de entrenamiento (70 %).
2. La particion de validacion (15 %) se usa SOLO como criterio de parada: se
   conserva el modelo con mayor recompensa acumulada en validacion y se detiene
   el entrenamiento cuando deja de mejorar.
3. Evaluacion final unica en la particion de prueba (15 %).
4. Tres semillas por activo. Se informa la media y la desviacion tipica entre
   semillas, porque una sola ejecucion de DQN no es evidencia.

Las senales del agente se evaluan con metricas_financieras, el mismo modulo que
usa la rama supervisada, con el mismo coste de 5 puntos basicos por unidad de
rotacion. Por eso las tablas de ambas ramas son directamente comparables.

Salidas generadas:
  resultados/metricas_dqn.csv          una fila por activo, semilla y particion
  resultados/backtest_dqn.csv          medias entre semillas y referencia
  resultados/senales_dqn.csv           senales del agente (validacion y prueba)
  resultados/curvas_capital_dqn.csv    curvas de capital del tramo de prueba
  resultados/modelos_dqn/              mejores pesos por activo y semilla

Uso:
  ./entorno_tfg/bin/python agente_dqn.py
  ./entorno_tfg/bin/python agente_dqn.py --activos SPY BTC_USD --pasos 20000
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.monitor import Monitor

from entorno_trading import EntornoTrading, cargar_particiones, simular_politica
from metricas_financieras import (
    COSTE_OPERACION,
    calcular_metricas,
    curva_capital,
    retornos_estrategia,
)
from modelos_supervisados import ACTIVOS

CARPETA_RESULTADOS = "resultados"
CARPETA_MODELOS = os.path.join(CARPETA_RESULTADOS, "modelos_dqn")

SEMILLAS = (0, 1, 2)
PASOS_ENTRENAMIENTO = 30_000

# Red identica a la del perceptron supervisado (64, 32): asi la comparacion
# entre ramas no depende de la capacidad del modelo, solo del objetivo optimizado.
HIPERPARAMETROS = {
    "learning_rate": 5e-4,
    "buffer_size": 50_000,
    "learning_starts": 1_000,
    "batch_size": 64,
    "tau": 1.0,
    "gamma": 0.99,
    "train_freq": 4,
    "gradient_steps": 1,
    "target_update_interval": 1_000,
    "exploration_fraction": 0.30,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
    "max_grad_norm": 10.0,
    "policy_kwargs": {"net_arch": [64, 32]},
}

COLUMNAS_METRICAS = [
    "retorno_total", "cagr", "volatilidad_anual", "ratio_sharpe",
    "ratio_sortino", "caida_maxima", "ratio_calmar", "exposicion",
    "n_operaciones", "aciertos",
]


def metricas_de_senales(senales, retornos_log):
    """Cuadro de metricas y curva de capital de una secuencia de senales."""
    netos, posiciones, rotacion = retornos_estrategia(
        senales, retornos_log, coste_operacion=COSTE_OPERACION
    )
    return calcular_metricas(netos, posiciones, rotacion), netos


def entrenar_semilla(particiones, semilla, pasos, verbose=0):
    """Entrena un DQN y devuelve el modelo con mejor recompensa en validacion."""
    entorno_train = Monitor(EntornoTrading(particiones["entrenamiento"]))
    entorno_validacion = Monitor(EntornoTrading(particiones["validacion"]))

    activo = particiones["entrenamiento"].activo
    carpeta_mejor = os.path.join(CARPETA_MODELOS, f"{activo}_semilla{semilla}")
    os.makedirs(carpeta_mejor, exist_ok=True)

    # Una evaluacion por episodio completo de entrenamiento.
    frecuencia = max(1, len(particiones["entrenamiento"]))

    parada = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=6, min_evals=8, verbose=verbose
    )
    evaluacion = EvalCallback(
        entorno_validacion,
        best_model_save_path=carpeta_mejor,
        log_path=carpeta_mejor,
        eval_freq=frecuencia,
        n_eval_episodes=1,       # el episodio es deterministico y completo
        deterministic=True,
        render=False,
        warn=False,
        verbose=verbose,
        callback_after_eval=parada,
    )

    modelo = DQN(
        "MlpPolicy",
        entorno_train,
        seed=semilla,
        device="cpu",            # red pequena: la CPU es mas rapida que la GPU
        verbose=verbose,
        **HIPERPARAMETROS,
    )
    modelo.learn(total_timesteps=pasos, callback=evaluacion, progress_bar=False)

    ruta_mejor = os.path.join(carpeta_mejor, "best_model.zip")
    if os.path.exists(ruta_mejor):
        modelo = DQN.load(ruta_mejor, device="cpu")

    entorno_train.close()
    entorno_validacion.close()
    return modelo, float(evaluacion.best_mean_reward)


def procesar_activo(activo, semillas, pasos, verbose=0):
    """Entrena todas las semillas de un activo y recopila metricas y senales."""
    particiones = cargar_particiones(activo)

    print(f"\nActivo {activo}: "
          f"entrenamiento {len(particiones['entrenamiento'])} | "
          f"validacion {len(particiones['validacion'])} | "
          f"prueba {len(particiones['prueba'])}")

    filas_metricas = []
    filas_senales = []
    curvas = []

    # Referencia comprar y mantener sobre el tramo de prueba, con costes.
    prueba = particiones["prueba"]
    metricas_bh, netos_bh = metricas_de_senales(
        np.ones(len(prueba)), prueba.retornos_log
    )
    filas_metricas.append({
        "activo": activo, "modelo": "Comprar y mantener", "semilla": np.nan,
        "particion": "prueba", **metricas_bh,
    })
    curvas.append(pd.DataFrame({
        "fecha": prueba.fechas,
        "activo": activo,
        "modelo": "Comprar y mantener",
        "semilla": np.nan,
        "capital": curva_capital(netos_bh),
    }))

    for semilla in semillas:
        modelo, mejor_validacion = entrenar_semilla(
            particiones, semilla, pasos, verbose=verbose
        )

        resumen_semilla = {}
        for nombre in ("validacion", "prueba"):
            registro = simular_politica(modelo, particiones[nombre])
            metricas, netos = metricas_de_senales(
                registro["senal"].to_numpy(), registro["retorno_futuro"].to_numpy()
            )

            filas_metricas.append({
                "activo": activo, "modelo": "DQN", "semilla": semilla,
                "particion": nombre, **metricas,
            })
            registro.insert(3, "modelo", "DQN")
            registro.insert(4, "semilla", semilla)
            filas_senales.append(registro)
            resumen_semilla[nombre] = metricas

            if nombre == "prueba":
                curvas.append(pd.DataFrame({
                    "fecha": registro["fecha"].to_numpy(),
                    "activo": activo,
                    "modelo": "DQN",
                    "semilla": semilla,
                    "capital": curva_capital(netos),
                }))

        print(f"  semilla {semilla}: recompensa validacion "
              f"{mejor_validacion:+.4f} | prueba retorno "
              f"{resumen_semilla['prueba']['retorno_total']:+.4f} "
              f"Sharpe {resumen_semilla['prueba']['ratio_sharpe']:+.3f} "
              f"caida {resumen_semilla['prueba']['caida_maxima']:+.4f} "
              f"exposicion {resumen_semilla['prueba']['exposicion']:.3f} "
              f"operaciones {int(resumen_semilla['prueba']['n_operaciones'])}")

    return filas_metricas, filas_senales, curvas


def agregar_por_semillas(tabla):
    """Media y desviacion tipica entre semillas por activo, modelo y particion."""
    agrupado = tabla.groupby(["activo", "modelo", "particion"], dropna=False)
    medias = agrupado[COLUMNAS_METRICAS].mean().add_suffix("_media")
    desviaciones = agrupado[COLUMNAS_METRICAS].std(ddof=0).add_suffix("_desv")
    n_semillas = agrupado.size().rename("n_semillas")
    return pd.concat([medias, desviaciones, n_semillas], axis=1).reset_index()


def main():
    analizador = argparse.ArgumentParser(
        description="Entrena y evalua el agente DQN de trading."
    )
    analizador.add_argument("--activos", nargs="+", default=ACTIVOS,
                            help="Activos a procesar.")
    analizador.add_argument("--semillas", nargs="+", type=int, default=list(SEMILLAS),
                            help="Semillas de entrenamiento.")
    analizador.add_argument("--pasos", type=int, default=PASOS_ENTRENAMIENTO,
                            help="Pasos maximos de entrenamiento por semilla.")
    analizador.add_argument("--verbose", type=int, default=0, choices=(0, 1),
                            help="Detalle de stable-baselines3.")
    argumentos = analizador.parse_args()

    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    os.makedirs(CARPETA_MODELOS, exist_ok=True)

    print("Entrenamiento del agente DQN")
    print(f"Coste de transaccion: {COSTE_OPERACION * 10000:.1f} puntos basicos "
          "por unidad de rotacion")
    print(f"Semillas: {argumentos.semillas} | pasos maximos por semilla: "
          f"{argumentos.pasos}")

    filas_metricas = []
    filas_senales = []
    curvas = []

    for activo in argumentos.activos:
        try:
            metricas, senales, curvas_activo = procesar_activo(
                activo, argumentos.semillas, argumentos.pasos, argumentos.verbose
            )
        except FileNotFoundError as error:
            print(f"\n[AVISO] {error} Omitido.")
            continue

        filas_metricas.extend(metricas)
        filas_senales.extend(senales)
        curvas.extend(curvas_activo)

    if not filas_metricas:
        print("\nNo se procesó ningún activo. Revisa datos_features y datos_procesados.")
        return

    tabla_metricas = pd.DataFrame(filas_metricas)
    ruta_metricas = os.path.join(CARPETA_RESULTADOS, "metricas_dqn.csv")
    tabla_metricas.to_csv(ruta_metricas, index=False)

    tabla_senales = pd.concat(filas_senales, ignore_index=True)
    ruta_senales = os.path.join(CARPETA_RESULTADOS, "senales_dqn.csv")
    tabla_senales.to_csv(ruta_senales, index=False)

    tabla_curvas = pd.concat(curvas, ignore_index=True)
    ruta_curvas = os.path.join(CARPETA_RESULTADOS, "curvas_capital_dqn.csv")
    tabla_curvas.to_csv(ruta_curvas, index=False)

    agregado = agregar_por_semillas(tabla_metricas)
    ruta_backtest = os.path.join(CARPETA_RESULTADOS, "backtest_dqn.csv")
    agregado.to_csv(ruta_backtest, index=False)

    prueba = agregado[agregado["particion"] == "prueba"]
    columnas_vista = [
        "activo", "modelo", "retorno_total_media", "ratio_sharpe_media",
        "ratio_sortino_media", "caida_maxima_media", "exposicion_media",
        "n_operaciones_media",
    ]
    print("\nParticion de prueba, media entre semillas")
    print(prueba[columnas_vista].round(4).to_string(index=False))

    resumen = (
        prueba.groupby("modelo")[
            ["retorno_total_media", "ratio_sharpe_media",
             "ratio_sortino_media", "caida_maxima_media"]
        ]
        .mean()
        .sort_values("ratio_sharpe_media", ascending=False)
    )
    print("\nMedia sobre los activos procesados (particion de prueba)")
    print(resumen.round(4).to_string())

    print(f"\nMetricas por semilla guardadas en {ruta_metricas}")
    print(f"Agregado por semillas guardado en {ruta_backtest}")
    print(f"Senales guardadas en {ruta_senales}")
    print(f"Curvas de capital guardadas en {ruta_curvas}")


if __name__ == "__main__":
    main()