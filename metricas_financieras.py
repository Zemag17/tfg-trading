"""
Metricas de rendimiento financiero para la evaluacion retrospectiva.

Modulo compartido por la rama supervisada y por la rama de Aprendizaje por
Refuerzo, de forma que ambas se midan exactamente con el mismo procedimiento.

Convencion temporal: la senal del dia t se decide con informacion disponible
al cierre de t y se aplica al retorno del dia t+1. La columna de retornos que
se pasa a estas funciones debe estar ya alineada con esa convencion.

Convencion de costes: coste proporcional aplicado sobre la rotacion de la
cartera, es decir sobre el valor absoluto del cambio de posicion. Un paso de
posicion 0 a 1 supone una rotacion de 1 y por tanto un coste completo.
"""

import numpy as np

DIAS_ANUALES = 252

# Coste por operacion sobre el nominal negociado (5 puntos basicos).
# Cubre comision y horquilla de compraventa en activos liquidos.
COSTE_OPERACION = 0.0005


def retornos_estrategia(senales, retornos_log, coste_operacion=COSTE_OPERACION):
    """Convierte una secuencia de senales en retornos simples netos de costes.

    Parametros
    ----------
    senales : secuencia numerica
        Posicion mantenida durante el periodo siguiente. Se admite 0/1 para
        estrategias solo largas y valores continuos en [-1, 1] para
        estrategias con posiciones cortas.
    retornos_log : secuencia numerica
        Retorno logaritmico del periodo al que se aplica cada senal.
    coste_operacion : float
        Coste proporcional por unidad de rotacion.

    Devuelve
    --------
    (retornos_netos, posiciones, rotacion) como arrays de numpy.
    """
    posiciones = np.asarray(senales, dtype=float)
    logaritmicos = np.asarray(retornos_log, dtype=float)

    if posiciones.shape != logaritmicos.shape:
        raise ValueError(
            "Las senales y los retornos deben tener la misma longitud: "
            f"{posiciones.shape} frente a {logaritmicos.shape}"
        )
    if posiciones.size == 0:
        vacio = np.empty(0, dtype=float)
        return vacio, vacio, vacio
    if not np.isfinite(logaritmicos).all():
        raise ValueError("Los retornos contienen valores no finitos.")

    # Los retornos logaritmicos no son aditivos al multiplicarlos por una
    # posicion: se convierten a retornos simples antes de aplicar la senal.
    simples = np.expm1(logaritmicos)

    # Se parte de cartera vacia, por lo que la primera operacion tambien paga coste.
    posicion_previa = np.concatenate(([0.0], posiciones[:-1]))
    rotacion = np.abs(posiciones - posicion_previa)

    netos = posiciones * simples - rotacion * coste_operacion
    return netos, posiciones, rotacion


def curva_capital(retornos_netos, capital_inicial=1.0):
    """Evolucion del capital por capitalizacion compuesta."""
    netos = np.asarray(retornos_netos, dtype=float)
    if netos.size == 0:
        return np.empty(0, dtype=float)
    return capital_inicial * np.cumprod(1.0 + netos)


def caida_maxima(curva):
    """Maxima caida relativa desde un maximo previo (valor negativo o cero)."""
    valores = np.asarray(curva, dtype=float)
    if valores.size == 0:
        return np.nan
    maximo_previo = np.maximum.accumulate(valores)
    caidas = valores / maximo_previo - 1.0
    return float(caidas.min())


def calcular_metricas(retornos_netos, posiciones=None, rotacion=None,
                      dias_anuales=DIAS_ANUALES):
    """Cuadro de metricas financieras de una serie de retornos netos."""
    netos = np.asarray(retornos_netos, dtype=float)
    n = netos.size

    if n == 0:
        return {clave: np.nan for clave in (
            "n_periodos", "retorno_total", "cagr", "volatilidad_anual",
            "ratio_sharpe", "ratio_sortino", "caida_maxima", "ratio_calmar",
            "exposicion", "n_operaciones", "aciertos",
        )}

    curva = curva_capital(netos)
    capital_final = float(curva[-1])
    anios = n / dias_anuales

    retorno_total = capital_final - 1.0
    if capital_final > 0.0 and anios > 0.0:
        cagr = capital_final ** (1.0 / anios) - 1.0
    else:
        # Capital agotado: la tasa compuesta no esta definida.
        cagr = np.nan

    media_diaria = float(netos.mean())
    if n > 1:
        volatilidad = float(netos.std(ddof=1)) * np.sqrt(dias_anuales)
    else:
        volatilidad = np.nan

    retorno_anualizado = media_diaria * dias_anuales
    sharpe = retorno_anualizado / volatilidad if volatilidad and volatilidad > 0 else np.nan

    # Sortino: solo penaliza la desviacion de los retornos negativos.
    negativos = netos[netos < 0.0]
    if negativos.size > 0:
        desviacion_baja = float(np.sqrt(np.mean(negativos ** 2))) * np.sqrt(dias_anuales)
        sortino = retorno_anualizado / desviacion_baja if desviacion_baja > 0 else np.nan
    else:
        # Ninguna perdida registrada: el ratio no esta acotado.
        sortino = np.inf if retorno_anualizado > 0 else np.nan

    max_caida = caida_maxima(curva)
    calmar = cagr / abs(max_caida) if max_caida and max_caida < 0 and np.isfinite(cagr) else np.nan

    exposicion = float(np.mean(np.abs(posiciones) > 0)) if posiciones is not None else np.nan
    n_operaciones = float(np.sum(np.asarray(rotacion) > 0)) if rotacion is not None else np.nan

    activos = netos[np.asarray(posiciones) != 0.0] if posiciones is not None else netos
    aciertos = float(np.mean(activos > 0)) if activos.size > 0 else np.nan

    return {
        "n_periodos": n,
        "retorno_total": retorno_total,
        "cagr": cagr,
        "volatilidad_anual": volatilidad,
        "ratio_sharpe": sharpe,
        "ratio_sortino": sortino,
        "caida_maxima": max_caida,
        "ratio_calmar": calmar,
        "exposicion": exposicion,
        "n_operaciones": n_operaciones,
        "aciertos": aciertos,
    }


def evaluar_senales(senales, retornos_log, coste_operacion=COSTE_OPERACION,
                    dias_anuales=DIAS_ANUALES):
    """Atajo: de senales a cuadro de metricas en una sola llamada."""
    netos, posiciones, rotacion = retornos_estrategia(
        senales, retornos_log, coste_operacion=coste_operacion
    )
    metricas = calcular_metricas(netos, posiciones, rotacion, dias_anuales)
    return metricas, netos