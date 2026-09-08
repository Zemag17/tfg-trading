"""
Entorno de decision secuencial para la rama de Aprendizaje por Refuerzo.

Formulacion del problema
------------------------
Estado      : 13 componentes -> las 12 variables estacionarias estandarizadas
              (las mismas que usa la rama supervisada) mas la posicion actual.
              Incluir la posicion es imprescindible: sin ella el agente no puede
              saber si mantener es gratis o si cambiar cuesta 5 puntos basicos,
              y el problema deja de ser markoviano.
Acciones    : 0 = fuera del mercado, 1 = posicion larga.
Recompensa  : accion * retorno_simple(t+1) - coste * |accion - posicion_previa|
Transicion  : avance de una sesion; el episodio termina al agotar la particion.

Convencion temporal (identica a la rama supervisada): las variables de la fila t
se calculan con informacion disponible al cierre de t y la decision tomada en t
se aplica al retorno de t+1. No hay fuga de informacion futura.

El coste y la conversion de retornos logaritmicos a simples se toman de
metricas_financieras, de modo que el agente se evalua con exactamente el mismo
procedimiento que los modelos supervisados.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from sklearn.preprocessing import StandardScaler

from metricas_financieras import COSTE_OPERACION
from modelos_supervisados import (
    CARPETA_DATOS,
    CARPETA_FEATURES,
    LIMITE_Z,
    VARIABLES,
    construir_features,
    dividir_cronologicamente,
)

NOMBRES_PARTICIONES = ("entrenamiento", "validacion", "prueba")


@dataclass(frozen=True)
class Particion:
    """Tramo cronologico ya estandarizado y listo para el entorno."""

    activo: str
    nombre: str
    variables: np.ndarray      # (n, 12) estandarizadas y recortadas a +-LIMITE_Z
    retornos_log: np.ndarray   # (n,) retorno logaritmico del dia siguiente
    fechas: pd.DatetimeIndex   # (n,) fecha de la decision
    precios: np.ndarray        # (n,) cierre del dia de la decision

    def __len__(self) -> int:
        return int(self.variables.shape[0])


def _matriz_features(activo, carpeta_features=CARPETA_FEATURES,
                     carpeta_datos=CARPETA_DATOS):
    """Carga la matriz de variables, reconstruyendola si aun no existe."""
    ruta_features = os.path.join(carpeta_features, f"{activo}_features.csv")
    if os.path.exists(ruta_features):
        return pd.read_csv(ruta_features, index_col=0, parse_dates=True)

    ruta_procesado = os.path.join(carpeta_datos, f"{activo}_procesado.csv")
    if not os.path.exists(ruta_procesado):
        raise FileNotFoundError(
            f"No existen {ruta_features} ni {ruta_procesado}. "
            "Ejecuta procesar_datos.py y alinear_datos.py primero."
        )
    datos = construir_features(
        pd.read_csv(ruta_procesado, index_col=0, parse_dates=True)
    )
    os.makedirs(carpeta_features, exist_ok=True)
    datos.to_csv(ruta_features)
    return datos


def cargar_particiones(activo, carpeta_features=CARPETA_FEATURES,
                       carpeta_datos=CARPETA_DATOS):
    """Devuelve las tres particiones cronologicas de un activo.

    El escalador se ajusta unicamente con el tramo de entrenamiento y las
    variables se recortan a +-LIMITE_Z desviaciones tipicas, igual que en
    modelos_supervisados.py.
    """
    datos = _matriz_features(activo, carpeta_features, carpeta_datos)

    faltantes = [columna for columna in VARIABLES if columna not in datos.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en la matriz de {activo}: {faltantes}")

    tramos = dict(zip(NOMBRES_PARTICIONES, dividir_cronologicamente(datos)))

    vacias = [nombre for nombre, tramo in tramos.items() if len(tramo) == 0]
    if vacias:
        raise ValueError(f"Particiones vacias en {activo}: {vacias}")

    escalador = StandardScaler().fit(tramos["entrenamiento"][VARIABLES])

    particiones = {}
    for nombre, tramo in tramos.items():
        matriz = escalador.transform(tramo[VARIABLES])
        matriz = np.clip(matriz, -LIMITE_Z, LIMITE_Z).astype(np.float32)
        particiones[nombre] = Particion(
            activo=activo,
            nombre=nombre,
            variables=matriz,
            retornos_log=tramo["retorno_futuro"].to_numpy(dtype=np.float64),
            fechas=pd.DatetimeIndex(tramo.index),
            precios=tramo["Close"].to_numpy(dtype=np.float64),
        )
    return particiones


class EntornoTrading(gym.Env):
    """Entorno de una sola posicion larga con coste por cambio de posicion."""

    metadata = {"render_modes": []}

    def __init__(self, particion: Particion, coste_operacion=COSTE_OPERACION):
        super().__init__()

        if len(particion) == 0:
            raise ValueError("La particion no contiene observaciones.")
        if particion.retornos_log.shape[0] != len(particion):
            raise ValueError("Variables y retornos tienen longitudes distintas.")
        if not np.isfinite(particion.variables).all():
            raise ValueError("Las variables contienen valores no finitos.")
        if not np.isfinite(particion.retornos_log).all():
            raise ValueError("Los retornos contienen valores no finitos.")

        self.particion = particion
        self.coste_operacion = float(coste_operacion)

        self.n_pasos = len(particion)
        self.n_variables = int(particion.variables.shape[1])

        # Retornos simples: los logaritmicos no son aditivos al multiplicarlos
        # por el tamano de la posicion.
        self._retornos_simples = np.expm1(particion.retornos_log)

        self.action_space = spaces.Discrete(2)
        limite_inferior = np.concatenate(
            (np.full(self.n_variables, -LIMITE_Z, dtype=np.float32),
             np.zeros(1, dtype=np.float32))
        )
        limite_superior = np.concatenate(
            (np.full(self.n_variables, LIMITE_Z, dtype=np.float32),
             np.ones(1, dtype=np.float32))
        )
        self.observation_space = spaces.Box(
            low=limite_inferior, high=limite_superior, dtype=np.float32
        )

        self.indice = 0
        self.posicion = 0.0
        self.senales: list[int] = []
        self.recompensas: list[float] = []
        self.rotaciones: list[float] = []

    # ------------------------------------------------------------------ API

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.indice = 0
        self.posicion = 0.0   # se parte siempre de cartera vacia
        self.senales = []
        self.recompensas = []
        self.rotaciones = []
        return self._observacion(), {}

    def step(self, accion):
        if self.indice >= self.n_pasos:
            raise RuntimeError("El episodio ya termino: llama a reset() antes.")

        accion = int(accion)
        if accion not in (0, 1):
            raise ValueError(f"Accion no valida: {accion}. Se admite 0 o 1.")

        posicion_nueva = float(accion)
        rotacion = abs(posicion_nueva - self.posicion)

        # Penalizacion artificial multiplicada para evitar la sobreoperacion
        multiplicador_castigo = 15.0
        recompensa = (
            posicion_nueva * float(self._retornos_simples[self.indice])
            - rotacion * (self.coste_operacion * multiplicador_castigo)
        )

        self.senales.append(accion)
        self.recompensas.append(recompensa)
        self.rotaciones.append(rotacion)

        self.posicion = posicion_nueva
        self.indice += 1

        terminado = self.indice >= self.n_pasos
        info = {
            "posicion": posicion_nueva,
            "rotacion": rotacion,
            "indice": self.indice - 1,
        }
        return self._observacion(), float(recompensa), terminado, False, info

    # ------------------------------------------------------------- internos

    def _observacion(self):
        # Al terminar el episodio se devuelve la ultima fila valida: la
        # observacion terminal no se usa para decidir, pero debe respetar el
        # espacio declarado.
        fila = min(self.indice, self.n_pasos - 1)
        return np.concatenate(
            (self.particion.variables[fila],
             np.array([self.posicion], dtype=np.float32))
        ).astype(np.float32)

    def episodio_a_dataframe(self):
        """Registro del episodio recorrido, listo para el analisis financiero."""
        n = len(self.senales)
        return pd.DataFrame({
            "fecha": self.particion.fechas[:n],
            "activo": self.particion.activo,
            "particion": self.particion.nombre,
            "senal": np.asarray(self.senales, dtype=int),
            "retorno_futuro": self.particion.retornos_log[:n],
            "precio_cierre": self.particion.precios[:n],
            "recompensa": np.asarray(self.recompensas, dtype=float),
            "rotacion": np.asarray(self.rotaciones, dtype=float),
        })


def simular_politica(modelo, particion, coste_operacion=COSTE_OPERACION):
    """Recorre una particion completa con la politica determinista del agente."""
    entorno = EntornoTrading(particion, coste_operacion=coste_operacion)
    observacion, _ = entorno.reset(seed=0)

    terminado = False
    while not terminado:
        accion, _ = modelo.predict(observacion, deterministic=True)
        observacion, _, terminado, _, _ = entorno.step(int(accion))

    return entorno.episodio_a_dataframe()