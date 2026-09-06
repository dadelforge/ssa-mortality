"""Quasi-Poisson dispersion modelling for excess-mortality z-scores.

Implements the Farrington et al. (1996) / Be-MOMO / EuroMOMO-style
quasi-Poisson approach to standardizing mortality residuals: variance is
assumed to scale with the baseline level, ``V(Y) = phi * mu``, with ``phi``
(overdispersion) estimated via iteratively reweighted Pearson residuals
that down-weight past excess-mortality weeks, so they don't bias the
estimate of "normal" background variability.

References
----------
Farrington, C. P., Andrews, N. J., Beale, A. D., & Catchpole, M. A. (1996).
A statistical algorithm for the early detection of outbreaks of infectious
disease. Journal of the Royal Statistical Society: Series A, 159(3),
547-563.

Cox, B., Wuillaume, F., Van Oyen, H., & Maes, S. (2010). Monitoring of
all-cause mortality in Belgium (Be-MOMO): a new and automated system for
the early detection and quantification of the mortality impact of public
health events. International Journal of Public Health, 55, 251-259.
"""
from __future__ import annotations

import warnings
from typing import Literal, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd

__all__ = [
    'fit_quasi_poisson_dispersion',
    'pearson_zscore',
    'quasi_poisson_bound',
]

# Return type mirrors whichever of ndarray / Series was passed in for mu.
ArrayOrSeries = TypeVar('ArrayOrSeries', np.ndarray, pd.Series)
Side = Literal['upper', 'lower']


def fit_quasi_poisson_dispersion(
        y: npt.ArrayLike,
        mu: npt.ArrayLike,
        n_iter: int = 15,
        tol: float = 1e-6,
) -> float:
    """Iteratively reweighted quasi-Poisson dispersion (Farrington et al.,
    1996; Cox et al., 2010, Be-MOMO). Assumes ``V(y) = phi * mu``.

    Weeks with a large positive standardized Pearson residual (past
    outbreaks pushing counts above the baseline) are down-weighted before
    re-estimating ``phi``, so they don't inflate the baseline noise
    estimate. Only positive deviations are down-weighted, matching the
    original method's focus on abnormally high rather than abnormally low
    counts. Floored at 1: the model can't claim less variance than plain
    Poisson.

    Parameters
    ----------
    y : array_like of float, shape (n,)
        Observed counts for the reference (non-wave) weeks.
    mu : array_like of float, shape (n,)
        Baseline (fitted mean) for the same weeks. Must be strictly
        positive.
    n_iter : int, default=15
        Maximum number of reweighting iterations.
    tol : float, default=1e-6
        Convergence tolerance on the change in ``phi`` between iterations.

    Returns
    -------
    float
        The (converged, or last) dispersion parameter ``phi``.

    Raises
    ------
    ValueError
        If `y` and `mu` have mismatched shapes, are empty, are not
        1-dimensional, contain NaN/infinite values, or if `mu` contains
        non-positive values; or if `n_iter` or `tol` are not valid.

    Warns
    -----
    RuntimeWarning
        If the iteration does not converge to `tol` within `n_iter` steps.
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    _validate_pair(y, mu, require_1d=True)
    if n_iter < 1:
        raise ValueError(f'n_iter must be >= 1, got {n_iter}')
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError(f'tol must be a finite positive number, got {tol}')

    phi = 1.0
    n = y.size
    phi_new = phi
    delta = np.inf
    for _ in range(n_iter):
        s = (y - mu) / np.sqrt(phi * mu)
        # np.where evaluates both branches eagerly, so s**-2 is computed
        # (and discarded) even where s <= 1, including s == 0; suppress
        # the resulting spurious divide-by-zero warning.
        with np.errstate(divide='ignore', invalid='ignore'):
            w = np.where(s > 1, s ** -2, 1.0)
        phi_new = max(np.sum(w * (y - mu) ** 2 / mu) / n, 1.0)
        delta = abs(phi_new - phi)
        if delta < tol:
            return phi_new
        phi = phi_new
    warnings.warn(
        f'fit_quasi_poisson_dispersion did not converge within {n_iter} '
        f'iterations (last change={delta:.2e}, tol={tol:.1e}); returning '
        f'the last estimate.',
        RuntimeWarning,
        stacklevel=2,
    )
    return phi_new


def pearson_zscore(
        y: npt.ArrayLike,
        mu: npt.ArrayLike,
        phi: float,
) -> np.ndarray:
    """Standardized Pearson residual under a quasi-Poisson variance model
    (Farrington et al., 1996; EuroMOMO-style).

    Parameters
    ----------
    y : array_like of float, shape (n,)
        Observed counts.
    mu : array_like of float, shape (n,)
        Baseline (fitted mean), same shape as `y`. Must be strictly
        positive.
    phi : float
        Dispersion parameter, e.g. from `fit_quasi_poisson_dispersion`.
        Must be positive.

    Returns
    -------
    ndarray
        Standardized residuals ``(y - mu) / sqrt(phi * mu)``.

    Raises
    ------
    ValueError
        If `y` and `mu` have mismatched shapes, contain NaN/infinite
        values, `mu` contains non-positive values, or `phi` is not a
        finite positive number.
    """
    y_arr = np.asarray(y, dtype=float)
    mu_arr = np.asarray(mu, dtype=float)
    _validate_pair(y_arr, mu_arr, require_1d=False)
    if not np.isfinite(phi) or phi <= 0:
        raise ValueError(f'phi must be a finite positive number, got {phi}')
    return (y_arr - mu_arr) / np.sqrt(phi * mu_arr)


def quasi_poisson_bound(
        mu: ArrayOrSeries,
        phi: float,
        z: float,
        side: Side = 'upper',
) -> ArrayOrSeries:
    """z-sigma prediction bound under a quasi-Poisson variance model
    ``V = phi * mu`` (Farrington et al., 1996), so the bound widens/narrows
    with the local baseline level instead of being a constant offset.

    Parameters
    ----------
    mu : ndarray or pandas.Series of float
        Baseline (fitted mean). Must be strictly positive. If a
        `pandas.Series` is passed, its index is preserved in the output.
    phi : float
        Dispersion parameter, e.g. from `fit_quasi_poisson_dispersion`.
        Must be positive.
    z : float
        Number of standard deviations for the bound (e.g. 1 or 2). Must be
        non-negative.
    side : {'upper', 'lower'}, default='upper'
        Which side of the baseline to compute. The lower bound is clipped
        at 0, since mortality counts can't be negative.

    Returns
    -------
    ndarray or pandas.Series
        ``mu + z*sqrt(phi*mu)`` for ``side='upper'``, or
        ``max(mu - z*sqrt(phi*mu), 0)`` for ``side='lower'``. Same type as
        `mu`.

    Raises
    ------
    ValueError
        If `mu` contains NaN/infinite or non-positive values, `phi` is not
        a finite positive number, `z` is not a finite non-negative number,
        or `side` is neither `'upper'` nor `'lower'`.
    """
    mu_arr = np.asarray(mu, dtype=float)
    if not np.all(np.isfinite(mu_arr)):
        raise ValueError('mu must not contain NaN or infinite values')
    if np.any(mu_arr <= 0):
        raise ValueError('mu must be strictly positive')
    if not np.isfinite(phi) or phi <= 0:
        raise ValueError(f'phi must be a finite positive number, got {phi}')
    if not np.isfinite(z) or z < 0:
        raise ValueError(f'z must be a finite non-negative number, got {z}')
    if side not in ('upper', 'lower'):
        raise ValueError(f"side must be 'upper' or 'lower', got {side!r}")

    # Operate on the original `mu` (not `mu_arr`) so a pandas Series input
    # keeps its index in the output.
    width = z * np.sqrt(phi * mu)
    if side == 'upper':
        return mu + width
    return np.clip(mu - width, 0, None)


def _validate_pair(
        y: np.ndarray,
        mu: np.ndarray,
        require_1d: bool,
) -> None:
    """Shared shape/positivity validation for `y`/`mu` pairs."""
    if y.shape != mu.shape:
        raise ValueError(
            f'y and mu must have the same shape, got {y.shape} and '
            f'{mu.shape}'
        )
    if require_1d and y.ndim != 1:
        raise ValueError(f'y and mu must be 1-dimensional, got ndim={y.ndim}')
    if y.size == 0:
        raise ValueError('y and mu must not be empty')
    if not np.all(np.isfinite(y)):
        raise ValueError('y must not contain NaN or infinite values')
    if not np.all(np.isfinite(mu)):
        raise ValueError('mu must not contain NaN or infinite values')
    if np.any(mu <= 0):
        raise ValueError('mu must be strictly positive')
