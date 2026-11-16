"""Run from the repository root: python -m unittest discover -s tests -v."""
import hashlib
from pathlib import Path
import random
import unittest

import joblib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from profile_model import GROUPS, LABELS, display_category, predict_profile, risk_level, validate_profile

ROOT = Path(__file__).resolve().parents[1]


class InferenceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact = joblib.load(ROOT / "modelo_riesgo.joblib")

    def profile(self):
        return {**{c: self.artifact["opciones"][c][0] for c in self.artifact["cat"]}, "edad": 24}

    def test_trained_artifact_unchanged(self):
        self.assertEqual(hashlib.sha256((ROOT / "modelo_riesgo.joblib").read_bytes()).hexdigest(),
                         "1eb36636273236df90c16ffeb4c08486f6cde20d4116deb73499bf2a74d63813")

    def test_exactly_sixteen_original_predictors(self):
        grouped = [c for _, rows in GROUPS for row in rows for c in row]
        self.assertEqual(len(grouped), 16)
        self.assertEqual(len(set(grouped)), 16)
        self.assertEqual(set(grouped), set(self.artifact["cat"] + self.artifact["num"]))
        self.assertEqual(set(grouped), set(LABELS))

    def test_empty_invalid_and_boundary_ages(self):
        self.assertEqual(len(validate_profile({}, self.artifact)), 16)
        for age in [15, 71, 23.5, float("nan")]:
            self.assertIn("edad", validate_profile({**self.profile(), "edad": age}, self.artifact))
        for age in [16, 70]:
            self.assertEqual(validate_profile({**self.profile(), "edad": age}, self.artifact), {})
        bad = {**self.profile(), "estu_genero": "invented category"}
        self.assertIn("estu_genero", validate_profile(bad, self.artifact))
        with self.assertRaises(ValueError):
            predict_profile(bad, self.artifact)

    def test_original_risk_cut_boundaries(self):
        cuts = self.artifact["cortes"]
        self.assertEqual(tuple(cuts), (0.13, 0.50))
        for p, expected in [(0, "Bajo"), (np.nextafter(.13, 0), "Bajo"), (.13, "Medio"),
                            (np.nextafter(.5, 0), "Medio"), (.5, "Alto"), (1, "Alto")]:
            self.assertEqual(risk_level(p, cuts), expected)

    def test_probabilities_identical_to_original_calculation(self):
        rng = random.Random(42)
        for _ in range(6):
            profile = {c: rng.choice(self.artifact["opciones"][c]) for c in self.artifact["cat"]}
            profile["edad"] = rng.randint(16, 70)
            expected = float(self.artifact["modelo"].predict_proba(pd.DataFrame([profile]))[:, 1][0])
            self.assertEqual(predict_profile(profile, self.artifact), expected)
            self.assertTrue(0 <= expected <= 1)

    def test_form_labels_do_not_replace_categories(self):
        self.assertEqual(display_category("F"), "Femenino")
        self.assertEqual(display_category("DISTANCIA VITUAL"), "Distancia virtual")
        for c in self.artifact["cat"]:
            for option in self.artifact["opciones"][c]:
                self.assertIsInstance(display_category(option), str)

    def test_every_artifact_category_is_accepted_by_pipeline(self):
        options = self.artifact["opciones"]
        size = max(len(options[column]) for column in self.artifact["cat"])
        rows = [
            {**{column: options[column][i % len(options[column])]
                for column in self.artifact["cat"]}, "edad": 16 + i % 55}
            for i in range(size)
        ]
        for row in rows:
            self.assertEqual(validate_profile(row, self.artifact), {})
        probabilities = self.artifact["modelo"].predict_proba(pd.DataFrame(rows))[:, 1]
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())


class StreamlitFlowTests(unittest.TestCase):
    def test_empty_validation_prediction_and_retained_values(self):
        artifact = joblib.load(ROOT / "modelo_riesgo.joblib")
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertFalse(list(app.exception))
        self.assertEqual([tab.label for tab in app.tabs],
                         ["Evaluar perfil", "Hallazgos del modelo", "Acerca del proyecto"])
        self.assertEqual(len(app.selectbox), 16)  # 15 predictors + findings filter
        self.assertIsNone(app.number_input(key="profile_edad").value)
        for column in artifact["cat"]:
            self.assertIsNone(app.selectbox(key=f"profile_{column}").value)
        app.button[0].click().run()
        self.assertFalse(list(app.exception))
        self.assertEqual(len(app.session_state["profile_errors"]), 16)
        self.assertEqual(len(app.error), 1)
        profile = {c: artifact["opciones"][c][0] for c in artifact["cat"]}
        for column, value in profile.items():
            app.selectbox(key=f"profile_{column}").select(value)
        app.number_input(key="profile_edad").set_value(24)
        app.button[0].click().run()
        self.assertFalse(list(app.exception))
        self.assertFalse(list(app.error))
        self.assertFalse(list(app.warning))
        self.assertEqual(app.session_state["profile_errors"], {})
        profile["edad"] = 24
        self.assertEqual(app.session_state["prediction"]["probability"], predict_profile(profile, artifact))
        for column in artifact["cat"]:
            self.assertEqual(app.selectbox(key=f"profile_{column}").value, profile[column])
        # A findings filter rerun must retain both the profile and its result.
        finding = app.selectbox(key="finding_group")
        groups = pd.read_csv(ROOT / "resultados" / "tasa_riesgo_por_categoria.csv")["variable"].unique()
        finding.select(groups[1]).run()
        self.assertFalse(list(app.exception))
        self.assertEqual(app.session_state["prediction"]["profile"], profile)
        self.assertEqual(app.number_input(key="profile_edad").value, 24)


if __name__ == "__main__":
    unittest.main()
