"""
reynolds_solver v1.1.0 — CPU reference implementation.

Solves the Reynolds equation for journal bearings using SOR iteration
with half-Sommerfeld (cavitation) boundary condition.

Dynamic terms (squeeze film) are supported via xprime / yprime parameters.
"""

from .solver import solve_reynolds

__version__ = "1.1.0"
__all__ = ["solve_reynolds"]
