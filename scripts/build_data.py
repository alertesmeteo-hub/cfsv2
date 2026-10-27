"""Produit les prévisions saisonnières CFSv2 pour l'Occitanie et les Pyrénées-Orientales : 8 semaines et 9 mois.

Pour chaque échéance : moyenne de 4 membres (membres 01 à 04 du cycle 00 UTC) de la température à 2 m et de la pluie, et écart à la
climatologie du modèle = moyenne des mêmes échéances des runs du même jour du calendrier des années passées (2019 à l'année dernière,
membre 01), seules années disponibles sur AWS. Sortie : build/data/index.json, occitanie.json et france.json (8 semaines et 9 mois, métropole), publiés sur la branche data.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calcul as c  # noqa: E402
import source as s  # noqa: E402

# (lat_min, lat_max, lon_min, lon_max) : Occitanie, et Pyrénées-Orientales (mailles autour du département)
ZONE_OCCITANIE = (41.9, 45.3, -0.9, 5.4)
ZONE_66 = (42.0, 43.0, 1.4, 3.6)
# France métropolitaine (Corse comprise) : cartes France
ZONE_FRANCE = (41.0, 51.5, -5.8, 9.9)
CYCLE = 0
MEMBRES = (1, 2, 3, 4)
ANNEES_CLIM = list(range(2019, date.today().year))
SEMAINES = 8
MOIS = 9
PARALLELE = 8


def moyenne_pas(cles: list[str], zone) -> tuple[dict[str, np.ndarray], tuple[np.ndarray, np.ndarray]] | None:
    """Moyenne champ par champ de plusieurs fichiers (les pas d'une fenêtre). None si un fichier manque."""
    with ThreadPoolExecutor(PARALLELE) as ex:
        champs = list(ex.map(lambda k: s.lire_champs(k, zone), cles))
    if any(ch is None for ch in champs):
        return None
    ref = champs[0]["t2m"]
    for ch in champs:
        if not (np.allclose(ch["t2m"].lat, ref.lat) and np.allclose(ch["t2m"].lon, ref.lon)):
            raise RuntimeError("La grille change d'un fichier à l'autre.")
    return {nom: np.mean([ch[nom].valeurs for ch in champs], axis=0) for nom in ("t2m", "prate")}, (ref.lat, ref.lon)


def fenetre_6h(init: date, membre: int, debut: date, fin: date, zone):
    cles = [s.cle_6h(init, CYCLE, membre, j, h) for j, h in c.pas_de_6h(debut, fin)]
    return moyenne_pas(cles, zone)


def resume_zone(lat, lon, zone, prevu: dict[str, np.ndarray], ref: dict[str, np.ndarray], jours: int) -> dict:
    """Écarts moyens sur la zone pour une échéance : température (°C), pluie (mm sur la période et % de la normale du modèle)."""
    m = c.masque_zone(lat, lon, zone)
    t_prevu = c.moyenne_zone(lat, prevu["t2m"] - c.KELVIN, m)
    t_ref = c.moyenne_zone(lat, ref["t2m"] - c.KELVIN, m)
    p_prevu = c.moyenne_zone(lat, c.pluie_mm_par_jour(prevu["prate"]), m)
    p_ref = c.moyenne_zone(lat, c.pluie_mm_par_jour(ref["prate"]), m)
    anom_p = (p_prevu / p_ref - 1) * 100 if p_ref >= 0.1 else None
    anom_t = t_prevu - t_ref
    return {
        "t2m": {"prevu": c.arrondir(t_prevu), "reference": c.arrondir(t_ref), "anomalie": c.arrondir(anom_t), "lecture": c.lecture_temperature(anom_t)},
        "pluie": {"mm": c.arrondir(p_prevu * jours, 0), "reference_mm": c.arrondir(p_ref * jours, 0), "anomalie_pct": c.arrondir(anom_p, 0), "lecture": c.tercile_pluie(anom_p)},
    }


def cellules(lat, lon, prevu, ref, zone=None) -> list[list[float | None]]:
    """Écarts point par point sur `zone` (toute la grille si None), pour la carte : [lat, lon, écart de température °C, écart de pluie %]."""
    if zone is not None:
        m = c.masque_zone(lat, lon, zone)
        lat, lon = lat[m], lon[m]
        prevu = {n: v[m] for n, v in prevu.items()}
        ref = {n: v[m] for n, v in ref.items()}
    at = c.anomalie_temperature(prevu["t2m"], ref["t2m"])
    ap = c.anomalie_pluie_pct(c.pluie_mm_par_jour(prevu["prate"]), c.pluie_mm_par_jour(ref["prate"]))
    return [[round(float(a), 2), round(float(b), 2), c.arrondir(float(t)), c.arrondir(float(p), 0)] for a, b, t, p in zip(lat, lon, at, ap)]


def construire(init: date) -> dict:
    zone_occ = ZONE_OCCITANIE
    hebdo, mensuel, hebdo_france, mensuel_france = [], [], [], []

    # Semaines (fichiers de 6 h), tout l'ensemble puis les années de référence
    for k, (debut, fin) in enumerate(c.semaines(init, SEMAINES), start=1):
        jours = (fin - debut).days + 1
        membres = []
        grille = None
        for mb in MEMBRES:
            r = fenetre_6h(init, mb, debut, fin, ZONE_FRANCE)
            if r:
                membres.append(r[0])
                grille = r[1]
        if not membres or grille is None:
            print(f"Semaine {k} ({debut} → {fin}) : fichiers manquants, ignorée.", flush=True)
            continue
        prevu = {n: np.mean([m[n] for m in membres], axis=0) for n in ("t2m", "prate")}
        refs = []
        annees_ok = set()
        for an in ANNEES_CLIM:
            d0 = c.memes_jour_annee(init, an)
            for mb in MEMBRES:
                r = fenetre_6h(d0, mb, d0 + (debut - init), d0 + (fin - init), ZONE_FRANCE)
                if r:
                    refs.append(r[0])
                    annees_ok.add(an)
        if len(annees_ok) < 3:
            print(f"Semaine {k} : moins de 3 années de référence, ignorée.", flush=True)
            continue
        ref = {n: np.mean([m[n] for m in refs], axis=0) for n in ("t2m", "prate")}
        lat, lon = grille
        t_membres = [c.moyenne_zone(lat, m["t2m"] - c.KELVIN, c.masque_zone(lat, lon, ZONE_66)) for m in membres]
        entree = {
            "n": k,
            "debut": debut.isoformat(),
            "fin": fin.isoformat(),
            "membres": len(membres),
            "annees_reference": len(annees_ok),
            "occitanie": resume_zone(lat, lon, zone_occ, prevu, ref, jours),
            "po": resume_zone(lat, lon, ZONE_66, prevu, ref, jours),
            "po_dispersion_t2m": c.arrondir(float(np.std(t_membres))),
            "cellules": cellules(lat, lon, prevu, ref, zone_occ),
        }
        hebdo.append(entree)
        hebdo_france.append(
            {
                "n": k,
                "debut": debut.isoformat(),
                "fin": fin.isoformat(),
                "membres": len(membres),
                "annees_reference": len(annees_ok),
                "france": resume_zone(lat, lon, ZONE_FRANCE, prevu, ref, jours),
                "cellules": cellules(lat, lon, prevu, ref),
            }
        )
        print(f"Semaine {k} : écart PO {entree['po']['t2m']['anomalie']} °C, pluie {entree['po']['pluie']['anomalie_pct']} %", flush=True)

    # Mois (fichiers mensuels)
    for mois in c.mois_cibles(init, MOIS):
        jours_mois = (date(int(mois[:4]) + (int(mois[4:]) == 12), int(mois[4:]) % 12 + 1, 1) - date(int(mois[:4]), int(mois[4:]), 1)).days
        membres = []
        grille = None
        for mb in MEMBRES:
            ch = s.lire_champs(s.cle_mensuelle(init, CYCLE, mb, mois), ZONE_FRANCE)
            if ch:
                membres.append({"t2m": ch["t2m"].valeurs, "prate": ch["prate"].valeurs})
                grille = (ch["t2m"].lat, ch["t2m"].lon)
        if not membres or grille is None:
            print(f"Mois {mois} : fichiers manquants, ignoré.", flush=True)
            continue
        prevu = {n: np.mean([m[n] for m in membres], axis=0) for n in ("t2m", "prate")}
        refs = []
        annees_ok = set()
        for an in ANNEES_CLIM:
            d0 = c.memes_jour_annee(init, an)
            decal = (int(mois[:4]) - init.year, int(mois[4:]) - init.month)
            a2, m2 = c.ajouter_mois(an, init.month, decal[0] * 12 + decal[1])
            for mb in MEMBRES:
                ch = s.lire_champs(s.cle_mensuelle(d0, CYCLE, mb, f"{a2}{m2:02d}"), ZONE_FRANCE)
                if ch:
                    refs.append({"t2m": ch["t2m"].valeurs, "prate": ch["prate"].valeurs})
                    annees_ok.add(an)
        if len(annees_ok) < 3:
            print(f"Mois {mois} : moins de 3 années de référence, ignoré.", flush=True)
            continue
        ref = {n: np.mean([m[n] for m in refs], axis=0) for n in ("t2m", "prate")}
        lat, lon = grille
        entree = {
            "mois": f"{mois[:4]}-{mois[4:]}",
            "membres": len(membres),
            "annees_reference": len(annees_ok),
            "occitanie": resume_zone(lat, lon, zone_occ, prevu, ref, jours_mois),
            "po": resume_zone(lat, lon, ZONE_66, prevu, ref, jours_mois),
            "cellules": cellules(lat, lon, prevu, ref, zone_occ),
        }
        mensuel.append(entree)
        mensuel_france.append(
            {
                "mois": entree["mois"],
                "membres": len(membres),
                "annees_reference": len(annees_ok),
                "france": resume_zone(lat, lon, ZONE_FRANCE, prevu, ref, jours_mois),
                "cellules": cellules(lat, lon, prevu, ref),
            }
        )
        print(f"Mois {mois} : écart PO {entree['po']['t2m']['anomalie']} °C, pluie {entree['po']['pluie']['anomalie_pct']} %", flush=True)

    return {
        "schema_version": 1,
        "init": init.isoformat(),
        "cycle_utc": CYCLE,
        "genere_le": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "NOAA NCEP CFSv2 (domaine public), via le bucket AWS noaa-cfs-pds ; champs flxf TMP 2 m et PRATE",
        "methode": (
            f"Moyenne des membres {', '.join(str(m) for m in MEMBRES)} du cycle {CYCLE:02d} UTC. Écart = prévision moins la moyenne des runs du même jour du "
            f"calendrier de {ANNEES_CLIM[0]} à {ANNEES_CLIM[-1]} (4 membres par année), seules années archivées sur AWS : une référence courte, à lire avec prudence. Semaines : pas de 00 et 12 UTC."
        ),
        "zones": {"occitanie": list(ZONE_OCCITANIE), "po": list(ZONE_66)},
        "hebdo": hebdo,
        "mensuel": mensuel,
        "france": {"zone": list(ZONE_FRANCE), "hebdo": hebdo_france, "mensuel": mensuel_france},
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--init", help="date d'initialisation AAAA-MM-JJ (défaut : le dernier run complet)")
    p.add_argument("--sortie", default=str(Path(__file__).resolve().parent.parent / "build" / "data"))
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    init = date.fromisoformat(a.init) if a.init else s.dernier_run_complet(datetime.now(timezone.utc).date())
    if init is None:
        print("Aucun run CFSv2 complet trouvé.")
        return 0
    sortie = Path(a.sortie)
    if not a.force and (sortie / "occitanie.json").exists():
        print("Déjà construit.")
        return 0
    data = construire(init)
    if len(data["hebdo"]) < 4 or len(data["mensuel"]) < 5:
        print(f"Publication incomplète ({len(data['hebdo'])} semaines, {len(data['mensuel'])} mois) : rien n'est publié.")
        return 1
    sortie.mkdir(parents=True, exist_ok=True)
    france = {k: data[k] for k in ("schema_version", "init", "cycle_utc", "genere_le", "source", "methode")}
    france.update(data.pop("france"))
    (sortie / "france.json").write_text(json.dumps(france, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    (sortie / "occitanie.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    index = {k: data[k] for k in ("schema_version", "init", "cycle_utc", "genere_le", "source", "methode", "zones")}
    index["semaines"] = len(data["hebdo"])
    index["mois"] = len(data["mensuel"])
    index["zones"]["france"] = list(ZONE_FRANCE)
    (sortie / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Écrit {sortie} : init {init}, {index['semaines']} semaines, {index['mois']} mois.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
