"""
Implementation of material classes for being used in dolfinx fem
"""
import ufl
import numpy as np
from   dolfinx import fem

class IsotropicLinearElastic:
    """
    Small-strain linear elastic isotropic material for dolfinx/ufl.

    Supports:
      - 2D or 3D
      - Plane strain (default in 2D)
      - Plane stress (optional in 2D)
    """

    def __init__(self, E, nu, dim, plane_stress=False, name="Elastic"):
        self.E = E
        self.nu = nu
        self.dim = dim
        self.plane_stress = plane_stress
        self.name = name

        if dim not in (2, 3):
            raise ValueError("dim must be 2 or 3")

        self._compute_lame_parameters()

    def _compute_lame_parameters(self):
        E  = self.E
        nu = self.nu

        if self.dim == 3:
            self.lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            self.mu = E / (2 * (1 + nu))

        elif self.dim == 2:
            mu = E / (2 * (1 + nu))

            if self.plane_stress:
                # plane stress correction
                lmbda = 2 * mu * nu / (1 - nu)
            else:
                # plane strain (3D law restricted to 2D)
                lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

            self.lmbda = lmbda
            self.mu = mu

    def eps(self, u):
        """Small strain tensor."""
        return ufl.sym(ufl.grad(u))

    def sigma(self, u):
        """Cauchy stress tensor σ = λ tr(ε) I + 2μ ε"""
        strain = self.eps(u)
        I = ufl.Identity(self.dim)
        return self.lmbda * ufl.tr(strain) * I + 2.0 * self.mu * strain

    def strain_energy_density(self, eps):
        return 0.5 * ufl.inner(self.sigma(eps), eps)
