"""Pruebas de humo para las API publicas principales del proyecto.

La suite es intencionalmente pequena: verifica reglas analiticas, invariantes
fisicos basicos y una integracion numerica corta sin regenerar los barridos de
parametros ni los resultados pesados de los notebooks.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import (  # noqa: E402
    BarotropicPowerDiffusionModel,
    BarotropicPowerDiffusionParams,
    CosCoshModel,
    CosCoshParams,
    PhantomQ0Model,
    PhantomQ0Params,
    PhantomQAlphaRhoModel,
    PhantomQAlphaRhoParams,
)
from src.parameter_space import build_parameter_grid  # noqa: E402


class FutureEndpointSmokeTests(unittest.TestCase):
    def test_linear_diffusion_has_finite_endpoint_only_above_half(self) -> None:
        for nu in (-0.25, 0.0, 0.25, 0.5):
            with self.subTest(nu=nu):
                model = PhantomQAlphaRhoModel(
                    PhantomQAlphaRhoParams(
                        A=0.2,
                        nu=nu,
                        alpha=0.3,
                        rho0=1.0,
                    )
                )
                self.assertIsNone(model.singular_time())

        for nu in (0.75, 1.0, 1.25):
            with self.subTest(nu=nu):
                model = PhantomQAlphaRhoModel(
                    PhantomQAlphaRhoParams(
                        A=0.2,
                        nu=nu,
                        alpha=0.3,
                        rho0=1.0,
                    )
                )
                self.assertGreater(model.singular_time(), 0.0)


class AnalyticModelSmokeTests(unittest.TestCase):
    def test_phantom_q0_background_obeys_friedmann_constraint(self) -> None:
        model = PhantomQ0Model(
            PhantomQ0Params(A=0.2, nu=0.75, rho0=1.0)
        )
        background = model.evaluate_background_t(np.linspace(0.0, 0.1, 9))

        self.assertEqual(len(background), 9)
        self.assertTrue(np.isfinite(background.to_numpy(dtype=float)).all())
        self.assertAlmostEqual(float(background.iloc[0]["rho"]), 1.0)
        self.assertAlmostEqual(float(background.iloc[0]["a"]), 1.0)
        self.assertTrue((np.diff(background["rho"]) > 0.0).all())
        self.assertTrue((np.diff(background["a"]) > 0.0).all())
        np.testing.assert_allclose(
            3.0 * background["H"].to_numpy() ** 2,
            background["rho"].to_numpy(),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_phantom_diffusion_background_obeys_effective_friedmann_constraint(
        self,
    ) -> None:
        alpha = 0.3
        model = PhantomQAlphaRhoModel(
            PhantomQAlphaRhoParams(
                A=0.2,
                nu=0.75,
                alpha=alpha,
                rho0=1.0,
            )
        )
        background = model.evaluate_background_t(np.linspace(0.0, 0.1, 9))

        self.assertTrue(np.isfinite(background.to_numpy(dtype=float)).all())
        np.testing.assert_allclose(
            background["Q"].to_numpy(),
            alpha * background["rho"].to_numpy(),
            rtol=1e-13,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            3.0 * background["H"].to_numpy() ** 2,
            background["rho_eff"].to_numpy(),
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertGreater(model.singular_time(), float(background["t"].iloc[-1]))

    def test_cos_cosh_background_preserves_initial_normalization(self) -> None:
        model = CosCoshModel(CosCoshParams())
        background = model.evaluate_background_x(np.linspace(0.0, 0.1, 11))

        self.assertEqual(len(background), 11)
        self.assertTrue(np.isfinite(background.to_numpy(dtype=float)).all())
        for column in ("a", "rho", "H"):
            self.assertAlmostEqual(float(background.iloc[0][column]), 1.0)
        self.assertAlmostEqual(float(background.iloc[0]["t"]), 0.0)
        self.assertTrue((np.diff(background["t"]) > 0.0).all())


class NumericalModelSmokeTests(unittest.TestCase):
    def test_short_barotropic_integration_is_finite_and_reaches_t_end(self) -> None:
        alpha = 0.2
        beta = 1.0
        model = BarotropicPowerDiffusionModel(
            BarotropicPowerDiffusionParams(
                omega=-1.1,
                alpha=alpha,
                beta=beta,
                rho0=1.0,
            )
        )
        background = model.solve_background_t(
            t_end=0.1,
            n_points=21,
            max_step=0.02,
        )

        self.assertEqual(list(background.columns), ["t", "rho", "a", "H", "Hdot"])
        self.assertEqual(len(background), 21)
        self.assertTrue(np.isfinite(background.to_numpy(dtype=float)).all())
        self.assertTrue(background.attrs["solver_success"])
        self.assertTrue(background.attrs["reached_t_end"])
        self.assertIsNone(background.attrs["termination_event"])
        self.assertAlmostEqual(float(background["t"].iloc[-1]), 0.1)
        self.assertTrue((np.diff(background["rho"]) > 0.0).all())
        self.assertTrue((np.diff(background["a"]) > 0.0).all())

        rho = background["rho"].to_numpy()
        expected_effective_density = rho + alpha * rho**beta
        np.testing.assert_allclose(
            3.0 * background["H"].to_numpy() ** 2,
            expected_effective_density,
            rtol=1e-10,
            atol=1e-12,
        )

    def test_parameter_grid_has_cartesian_size_and_marks_invalid_domain(self) -> None:
        grid = build_parameter_grid(
            alpha_values=[-2.0, 0.2],
            beta_values=[0.0, 1.0],
            omega_values=[-1.2, -0.9],
        )

        self.assertEqual(len(grid), 2 * 2 * 2)
        self.assertEqual(grid["point_id"].nunique(), len(grid))
        self.assertEqual(int((~grid["valid_initial_domain"]).sum()), 4)
        self.assertTrue(
            grid.loc[~grid["valid_initial_domain"], "invalid_reason"]
            .str.contains("F\\(rho0\\)<=0", regex=True)
            .all()
        )


if __name__ == "__main__":
    unittest.main()
