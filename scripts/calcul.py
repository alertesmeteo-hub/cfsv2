"""Calculs purs sur les prévisions CFSv2 : semaines, mois, moyennes sur une zone, anomalies. Aucun accès réseau (testé dans tests/)."""
from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np

SECONDES_PAR_JOUR = 86400.0
KELVIN = 273.15


def ajouter_mois(annee: int, mois: int, n: int) -> tuple[int, int]:
    """(année, mois) décalé de `n` mois."""
    total = annee * 12 + (mois - 1) + n
    return total // 12, total % 12 + 1


def mois_cibles(init: date, n: int = 9) -> list[str]:
    """Les `n` mois qui suivent le mois d'initialisation, au format AAAAMM (le mois de l'initialisation est partiel : on ne le garde pas)."""
    return [f"{a}{m:02d}" for a, m in (ajouter_mois(init.year, init.month, k) for k in range(1, n + 1))]


def semaines(init: date, n: int = 8) -> list[tuple[date, date]]:
    """Les `n` semaines de 7 jours qui suivent le jour d'initialisation : la 1re va de J+1 à J+7, la 2e de J+8 à J+14, etc."""
    return [(init + timedelta(days=7 * k + 1), init + timedelta(days=7 * k + 7)) for k in range(n)]


HEURES_RETENUES = (0, 12)


def pas_de_6h(debut: date, fin: date, heures: tuple[int, ...] = HEURES_RETENUES) -> list[tuple[date, int]]:
    """Les pas retenus de chaque jour de `debut` à `fin` inclus : 00 et 12 UTC suffisent pour une moyenne hebdomadaire (même échantillonnage
    pour la prévision et pour la référence, donc un écart sans biais d'heure), et divisent le téléchargement par deux."""
    jours = (fin - debut).days + 1
    return [(debut + timedelta(days=j), h) for j in range(jours) for h in heures]


def memes_jour_annee(init: date, annee: int) -> date:
    """Le même jour du calendrier dans `annee` (le 29 février devient le 28)."""
    try:
        return init.replace(year=annee)
    except ValueError:
        return init.replace(year=annee, day=28)


def moyenne_zone(lat: np.ndarray, valeurs: np.ndarray, masque: np.ndarray) -> float:
    """Moyenne des points du masque, pondérée par cos(latitude) (surface d'une maille)."""
    if not masque.any():
        return float("nan")
    poids = np.cos(np.radians(lat[masque]))
    return float(np.average(valeurs[masque], weights=poids))


def masque_zone(lat: np.ndarray, lon: np.ndarray, zone: tuple[float, float, float, float]) -> np.ndarray:
    lat_min, lat_max, lon_min, lon_max = zone
    return (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)


def anomalie_temperature(prevu_k: np.ndarray, reference_k: np.ndarray) -> np.ndarray:
    """Écart de température (°C = K) entre la prévision et la référence, point par point."""
    return prevu_k - reference_k


def pluie_mm_par_jour(prate: np.ndarray) -> np.ndarray:
    """kg m-2 s-1 → mm par jour (1 kg d'eau par m² = 1 mm)."""
    return prate * SECONDES_PAR_JOUR


def anomalie_pluie_pct(prevu_mm_j: np.ndarray, reference_mm_j: np.ndarray, plancher: float = 0.1) -> np.ndarray:
    """Écart de pluie en % de la référence ; NaN là où la référence est sous `plancher` mm/jour (pourcentage sans signification)."""
    ref = np.where(reference_mm_j >= plancher, reference_mm_j, np.nan)
    return (prevu_mm_j / ref - 1.0) * 100.0


def arrondir(v: float | None, n: int = 1) -> float | None:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    return round(float(v), n)


def tercile_pluie(anom_pct: float | None) -> str | None:
    """Lecture simple d'un écart de pluie : « sec » (≤ −25 %), « normal », « humide » (≥ +25 %)."""
    if anom_pct is None:
        return None
    return "sec" if anom_pct <= -25 else "humide" if anom_pct >= 25 else "normal"


def lecture_temperature(anom_c: float | None) -> str | None:
    """Lecture simple d'un écart de température : « frais » (≤ −1 °C), « normal », « chaud » (≥ +1 °C)."""
    if anom_c is None:
        return None
    return "frais" if anom_c <= -1 else "chaud" if anom_c >= 1 else "normal"
