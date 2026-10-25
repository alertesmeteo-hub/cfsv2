import sys
import unittest
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import calcul as c  # noqa: E402
import source as s  # noqa: E402


class TestCalendrier(unittest.TestCase):
    def test_mois_cibles_saute_le_mois_de_l_initialisation(self):
        self.assertEqual(c.mois_cibles(date(2026, 10, 25), 9), ["202611", "202612", "202701", "202702", "202703", "202704", "202705", "202706", "202707"])

    def test_semaines_commencent_le_lendemain(self):
        sem = c.semaines(date(2026, 10, 25), 8)
        self.assertEqual(sem[0], (date(2026, 10, 26), date(2026, 11, 1)))
        self.assertEqual(sem[7][1], date(2026, 12, 20))
        self.assertEqual(len(sem), 8)

    def test_pas_de_6h(self):
        pas = c.pas_de_6h(date(2026, 10, 26), date(2026, 11, 1))
        self.assertEqual(len(pas), 14)
        self.assertEqual(pas[0], (date(2026, 10, 26), 0))
        self.assertEqual(pas[-1], (date(2026, 11, 1), 12))

    def test_meme_jour_annee_et_29_fevrier(self):
        self.assertEqual(c.memes_jour_annee(date(2026, 10, 25), 2021), date(2021, 10, 25))
        self.assertEqual(c.memes_jour_annee(date(2024, 2, 29), 2023), date(2023, 2, 28))


class TestAnomalies(unittest.TestCase):
    def test_pluie_kg_m2_s_en_mm_par_jour(self):
        self.assertAlmostEqual(float(c.pluie_mm_par_jour(np.array([1e-5]))[0]), 0.864)

    def test_anomalie_pluie_pct_et_plancher(self):
        r = c.anomalie_pluie_pct(np.array([3.0, 1.0, 0.5]), np.array([2.0, 2.0, 0.05]))
        self.assertAlmostEqual(float(r[0]), 50.0)
        self.assertAlmostEqual(float(r[1]), -50.0)
        self.assertTrue(np.isnan(r[2]))

    def test_moyenne_zone_ponderee(self):
        lat = np.array([0.0, 60.0])
        val = np.array([10.0, 20.0])
        m = np.array([True, True])
        # poids cos(0)=1 et cos(60°)=0,5 : (10 + 20 × 0,5) / 1,5
        self.assertAlmostEqual(c.moyenne_zone(lat, val, m), 20 / 1.5 * 1.0, places=5)
        self.assertTrue(np.isnan(c.moyenne_zone(lat, val, np.array([False, False]))))

    def test_lectures(self):
        self.assertEqual(c.lecture_temperature(1.4), "chaud")
        self.assertEqual(c.lecture_temperature(-1.0), "frais")
        self.assertEqual(c.lecture_temperature(0.2), "normal")
        self.assertEqual(c.tercile_pluie(-40), "sec")
        self.assertEqual(c.tercile_pluie(30), "humide")
        self.assertIsNone(c.tercile_pluie(None))
        self.assertIsNone(c.arrondir(float("nan")))


class TestSource(unittest.TestCase):
    def test_cles_des_fichiers(self):
        self.assertEqual(
            s.cle_6h(date(2026, 10, 25), 0, 1, date(2026, 11, 5), 12),
            "cfs.20261025/00/6hrly_grib_01/flxf2026110512.01.2026102500.grb2",
        )
        self.assertEqual(
            s.cle_mensuelle(date(2026, 10, 25), 0, 2, "202611"),
            "cfs.20261025/00/monthly_grib_02/flxf.02.2026102500.202611.avrg.grib.grb2",
        )

    def test_parser_index(self):
        texte = "1:0:d=2026102400:UFLX:surface:6 hour fcst:\n2:100:d=2026102400:TMP:2 m above ground:6 hour fcst:\n3:250:d=2026102400:PRATE:surface:6 hour fcst:\n"
        idx = s.parser_index(texte, 400)
        self.assertEqual(idx[("TMP", "2 m above ground")], (100, 250))
        self.assertEqual(idx[("PRATE", "surface")], (250, 400))


if __name__ == "__main__":
    unittest.main()
