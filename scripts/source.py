"""Lecture des prévisions NOAA CFSv2 sur AWS (bucket public noaa-cfs-pds) : listes, index .idx, champs GRIB2 par plage d'octets.

Chaque fichier GRIB2 pèse plusieurs mégaoctets pour une quarantaine de champs ; l'index (.idx) donne le décalage de chaque champ, on ne
télécharge donc que la température à 2 m (TMP:2 m above ground) et le taux de précipitation (PRATE:surface), soit une centaine de
kilooctets par pas de temps.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date

import numpy as np
import requests

BASE = "https://noaa-cfs-pds.s3.amazonaws.com"
CHAMPS = {"t2m": ("TMP", "2 m above ground"), "prate": ("PRATE", "surface")}

_session = requests.Session()
_session.headers["User-Agent"] = "alertes-meteo-cfsv2/1.0 (+https://github.com/alertesmeteo-hub/cfsv2)"


def _get(url: str, headers: dict[str, str] | None = None, essais: int = 4) -> requests.Response:
    """GET avec nouvelles tentatives espacées sur les coupures passagères (429, 5xx, connexion)."""
    derniere: Exception | None = None
    for i in range(essais):
        try:
            r = _session.get(url, headers=headers, timeout=60)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
            return r
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            derniere = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"Échec de {url} : {derniere}")


def existe(cle: str) -> bool:
    """Vrai si l'objet existe dans le bucket (requête HEAD)."""
    for i in range(3):
        try:
            r = _session.head(f"{BASE}/{cle}", timeout=30)
            if r.status_code in (200, 404, 403):
                return r.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(2 * (i + 1))
    return False


def cle_6h(init: date, cycle: int, membre: int, cible: date, heure: int) -> str:
    """Fichier des champs de surface (flxf) d'un pas de 6 h : cfs.AAAAMMJJ/HH/6hrly_grib_0M/flxfAAAAMMJJHH.0M.AAAAMMJJHH.grb2."""
    i = f"{init:%Y%m%d}{cycle:02d}"
    return f"cfs.{init:%Y%m%d}/{cycle:02d}/6hrly_grib_{membre:02d}/flxf{cible:%Y%m%d}{heure:02d}.{membre:02d}.{i}.grb2"


def cle_mensuelle(init: date, cycle: int, membre: int, mois: str) -> str:
    """Moyenne mensuelle (flxf) du mois `mois` (AAAAMM) : cfs.AAAAMMJJ/HH/monthly_grib_0M/flxf.0M.AAAAMMJJHH.AAAAMM.avrg.grib.grb2."""
    i = f"{init:%Y%m%d}{cycle:02d}"
    return f"cfs.{init:%Y%m%d}/{cycle:02d}/monthly_grib_{membre:02d}/flxf.{membre:02d}.{i}.{mois}.avrg.grib.grb2"


@dataclass(frozen=True)
class Champ:
    """Un champ décodé, restreint à la zone : valeurs aux points (lat, lon en degrés, longitudes de -180 à 180)."""

    lat: np.ndarray
    lon: np.ndarray
    valeurs: np.ndarray


def parser_index(texte: str, taille: int) -> dict[tuple[str, str], tuple[int, int]]:
    """Index NCEP (« n:octet:d=…:VAR:niveau:échéance: ») → {(variable, niveau): (début, fin exclue)} en octets."""
    lignes = [ligne.split(":") for ligne in texte.strip().splitlines() if ligne.strip()]
    sortie: dict[tuple[str, str], tuple[int, int]] = {}
    for k, p in enumerate(lignes):
        debut = int(p[1])
        fin = int(lignes[k + 1][1]) if k + 1 < len(lignes) else taille
        sortie.setdefault((p[3], p[4]), (debut, fin))
    return sortie


def lire_champs(cle: str, zone: tuple[float, float, float, float]) -> dict[str, Champ] | None:
    """Température à 2 m (K) et précipitation (kg m-2 s-1) du fichier `cle`, restreintes à `zone` = (lat_min, lat_max, lon_min, lon_max).

    Renvoie None si le fichier n'existe pas (échéance non publiée) ; lève une erreur si le fichier existe mais ne se lit pas.
    """
    from eccodes import codes_get_array, codes_get_values, codes_new_from_message, codes_release

    r = _get(f"{BASE}/{cle}.idx")
    if r.status_code != 200:
        return None
    taille_r = _session.head(f"{BASE}/{cle}", timeout=30)
    taille = int(taille_r.headers.get("Content-Length", "0"))
    index = parser_index(r.text, taille)
    lat_min, lat_max, lon_min, lon_max = zone
    sortie: dict[str, Champ] = {}
    for nom, cle_champ in CHAMPS.items():
        if cle_champ not in index:
            raise RuntimeError(f"{cle} : champ {cle_champ} absent de l'index")
        a, b = index[cle_champ]
        g = _get(f"{BASE}/{cle}", headers={"Range": f"bytes={a}-{b - 1}"})
        if g.status_code not in (200, 206):
            raise RuntimeError(f"{cle} : HTTP {g.status_code} pour {nom}")
        gid = codes_new_from_message(g.content)
        try:
            lat = np.asarray(codes_get_array(gid, "latitudes"), dtype=float)
            lon = np.asarray(codes_get_array(gid, "longitudes"), dtype=float)
            val = np.asarray(codes_get_values(gid), dtype=float)
        finally:
            codes_release(gid)
        lon = np.where(lon > 180, lon - 360, lon)
        m = (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)
        sortie[nom] = Champ(lat[m], lon[m], val[m])
    return sortie


def nom_cycle(init: date, cycle: int) -> str:
    return f"{init:%Y-%m-%d} {cycle:02d} UTC"



def dernier_run_complet(aujourdhui: date, cycle: int = 0, jours_max: int = 4) -> date | None:
    """Le run (date d'initialisation) le plus récent dont le dernier fichier de 6 h utile (J+56) et le dernier mois (M+9) sont publiés."""
    from datetime import timedelta

    from calcul import mois_cibles

    for k in range(jours_max + 1):
        d = aujourdhui - timedelta(days=k)
        if existe(cle_6h(d, cycle, 1, d + timedelta(days=56), 18) + ".idx") and existe(cle_mensuelle(d, cycle, 1, mois_cibles(d, 9)[-1]) + ".idx"):
            return d
    return None
