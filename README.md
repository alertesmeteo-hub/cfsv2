# cfsv2

Prévisions saisonnières NOAA CFSv2 sur l'Occitanie et les Pyrénées-Orientales : **tendance à 8 semaines** (semaine par semaine) et **prévision à 9 mois** (mois par mois), pour le site [app.alertes-meteo.com](https://app.alertes-meteo.com/previsions-saisonnieres) et GitHub Actions.

## Ce que fait le dépôt

Un workflow GitHub Actions (`build-cfsv2.yml`, chaque jour vers 09 h 40 UTC, une fois le run de 00 UTC publié) :

1. lit les prévisions CFSv2 du bucket public AWS `noaa-cfs-pds` (domaine public) : température à 2 m (`TMP`) et taux de précipitation (`PRATE`) du fichier `flxf`, par plages d'octets grâce aux index `.idx` (une centaine de kilooctets par échéance au lieu de plusieurs mégaoctets) ;
2. moyenne les **4 membres** du run de 00 UTC (membres 01 à 04) ;
3. calcule l'**écart à l'habituel** : prévision moins la moyenne des prévisions du même modèle pour le même jour du calendrier des années passées (2019 jusqu'à l'année dernière, 4 membres par année) : seules années archivées sur AWS, donc une référence courte à lire avec prudence ;
4. publie `index.json` et `occitanie.json` sur la branche **`data`** (réécrite à chaque run, un seul commit).

## Fichiers publiés (branche `data`)

- `index.json` : date du run, méthode, zones.
- `occitanie.json` : `hebdo` (8 semaines : J+1 à J+7, J+8 à J+14…) et `mensuel` (9 mois suivant le mois du run). Pour chaque échéance : écart de température (°C) et de pluie (% et mm) pour les Pyrénées-Orientales (`po`) et l'Occitanie (`occitanie`), avec la lecture « frais / normal / chaud » (±1 °C) et « sec / normal / humide » (±25 %), la dispersion des membres, et les écarts maille par maille (`cellules` : latitude, longitude, écart de température, écart de pluie en %).

## Limites

- Mailles d'environ 1° : les reliefs ne sont pas détaillés.
- Au-delà de quatre semaines, une tendance seulement ; la pluie est très irrégulière.
- Les écarts sont relatifs au modèle lui-même (le biais du CFSv2 est en grande partie retiré), pas à une normale climatique de station.

## Développement

```
python -m pip install -r requirements.txt
python -m unittest discover -s tests
python scripts/build_data.py --init AAAA-MM-JJ --force     # écrit build/data/
```

`eccodes` (décodage GRIB2) s'installe par `pip` sous Linux et macOS ; sous Windows, utiliser un environnement Linux.
