"""Functions for filling missing values using Singular Spectrum Analysis (SSA).

This module provides multiple strategies to fill missing values (NaN) in time
series data using SSA, including both basic reconstruction and Monte Carlo
significance testing approaches.
"""

import logging
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from ssalib import MonteCarloSSA, SingularSpectrumAnalysis

logger = logging.getLogger(__name__)


def fill_na(
        timeseries: ArrayLike,
        strategy: Literal['basic', 'monte_carlo'] = 'basic',
        n_components: Optional[int] = None,
        window: Optional[int] = None,
        svd_matrix_kind: str = 'bk_trajectory',
        standardize: bool = True,
        max_frequency: Optional[float] = None,
        tol: float = 1e-3,
        max_iter: int = 100,
        return_as: Literal['signal', 'ssa', 'filled'] = 'filled',
        # Monte Carlo specific parameters
        n_surrogates: int = 100,
        ar_order_max: int = 1,
        confidence_level: float = 0.95,
        two_tailed: bool = True,
        random_seed: int | None = None
) -> Union[np.ndarray, pd.Series, SingularSpectrumAnalysis, MonteCarloSSA]:
    """Fill missing values in a time series using Singular Spectrum Analysis.

    This function provides a unified interface to fill missing values using
    either basic SSA reconstruction or Monte Carlo SSA with significance
    testing.

    Parameters
    ----------
    timeseries : ArrayLike
        Input time series data containing NaN values to be filled
    strategy : Literal['basic', 'monte_carlo'], default='basic'
        The strategy to use for filling missing values:
        - 'basic': Uses fixed number of components
        - 'monte_carlo': Uses Monte Carlo significance testing
    n_components : Optional[int], default=None
        Number of components to use for reconstruction. Required for 'basic'
        strategy, serves as maximum components for 'monte_carlo' strategy. If
        None and using 'monte_carlo' strategy, defaults to
        min(50, len(timeseries)//4)
    window : Optional[int], default=None
        Window length for SSA. If None, uses half the series length
    svd_matrix_kind : str, default='bk_trajectory'
        Type of matrix for SVD ('bk_trajectory', 'bk_covariance',
         or 'vg_covariance')
    standardize : bool, default=True
        Whether to standardize the time series
    max_frequency : Optional[float], default=None
        Maximum frequency to consider for component selection
    tol : float, default=1e-3
        Tolerance for convergence of RMSE
    max_iter : int, default=100
        Maximum number of iterations
    return_as : Literal['signal', 'ssa', 'filled'], default='filled'
        Format of the returned result:
        - 'signal': Returns only the reconstructed signal
        - 'ssa': Returns the full SSA object
        - 'filled': Returns the original series with NaN values filled
    n_surrogates : int, default=100
        Number of surrogate series to generate (monte_carlo only)
    ar_order_max : int, default=1
        Maximum order for autoregressive model (monte_carlo only)
    confidence_level : float, default=0.95
        Confidence level for significance testing (monte_carlo only)
    two_tailed : bool, default=True
        Whether to use two-tailed significance testing (monte_carlo only)
    random_seed : int | None, default=None
        Random seed for reproducibility (monte_carlo only)

    Returns
    -------
    Union[np.ndarray, pd.Series, SingularSpectrumAnalysis, MonteCarloSSA]
        Filled time series or SSA object, depending on return_as

    Raises
    ------
    ValueError
        If input parameters are invalid or if the time series cannot be processed
    RuntimeError
        If the algorithm fails to converge within max_iter iterations

    Examples
    --------
    Basic strategy with fixed number of components:
    >>> from ssalib.datasets import load_sst
    >>> sst = load_sst()
    >>> sst_with_gaps = sst.copy()
    >>> sst_with_gaps.iloc[400:450] = np.nan
    >>> filled_basic = fill_na(sst_with_gaps, strategy='basic', n_components=10)

    Monte Carlo strategy with significance testing:
    >>> filled_mc = fill_na(sst_with_gaps, strategy='monte_carlo',
    ...                    n_surrogates=100, confidence_level=0.95)
    """
    # Input validation
    if strategy not in ['basic', 'monte_carlo']:
        raise ValueError("strategy must be either 'basic' or 'monte_carlo'")

    if strategy == 'basic' and n_components is None:
        raise ValueError("n_components must be specified for basic strategy")

    if n_components is None:
        n_components = min(50, len(timeseries) // 4)

    # Dispatch to appropriate strategy
    if strategy == 'basic':
        return fill_na_basic(
            timeseries=timeseries,
            n_components=n_components,
            window=window,
            svd_matrix_matrix=svd_matrix_kind,
            standardize=standardize,
            max_frequency=max_frequency,
            tol=tol,
            max_iter=max_iter,
            return_as=return_as
        )
    else:  # strategy == 'monte_carlo'
        return fill_na_monte_carlo(
            timeseries=timeseries,
            max_n_component=n_components,
            window=window,
            svd_matrix_kind=svd_matrix_kind,
            standardize=standardize,
            n_surrogates=n_surrogates,
            ar_order_max=ar_order_max,
            confidence_level=confidence_level,
            two_tailed=two_tailed,
            max_frequency=max_frequency,
            tol=tol,
            max_iter=max_iter,
            return_as=return_as
        )


def fill_na_basic(
        timeseries: ArrayLike,
        n_components: int,
        window: Optional[int] = None,
        svd_matrix_kind: str = 'bk_trajectory',
        standardize: bool = True,
        max_frequency: Optional[float] = None,
        tol: float = 1e-3,
        max_iter: int = 100,
        return_as: Literal['signal', 'ssa', 'filled'] = 'filled'
) -> Union[np.ndarray, pd.Series, SingularSpectrumAnalysis]:
    """Fill missing values using basic SSA reconstruction using a fixed number
    of components.

    This function implements an iterative algorithm to fill missing values using
    SSA with a fixed number of components. The algorithm iterates until the
    change in RMSE between iterations falls below a tolerance threshold.

    Parameters
    ----------
    timeseries : ArrayLike
        Input time series data containing NaN values to be filled
    n_components : int
        Number of components to use for reconstruction
    window : Optional[int], default=None
        Window length for SSA. If None, uses half the series length
    svd_matrix_kind : str, default='bk_trajectory'
        Type of matrix for SVD ('bk_trajectory' or 'vg_covariance')
    standardize : bool, default=True
        Whether to standardize the time series
    max_frequency : Optional[float], default=None
        Maximum frequency to consider for component selection. If provided,
        only components with frequencies below this threshold are used.
    tol : float, default=1e-3
        Tolerance for convergence of RMSE
    max_iter : int, default=100
        Maximum number of iterations
    return_as : Literal['signal', 'ssa', 'filled'], default='filled'
        Format of the returned result:
        - 'signal': Returns only the reconstructed signal
        - 'ssa': Returns the full SSA object
        - 'filled': Returns the original series with NaN values filled

    Returns
    -------
    np.ndarray | pd.Series | SingularSpectrumAnalysis
        Filled time series or SSA object, depending on return_as

    Raises
    ------
    ValueError
        If input parameters are invalid or if the time series cannot be processed
    RuntimeError
        If the algorithm fails to converge within max_iter iterations

    Notes
    -----
    The algorithm works by:
    1. Initializing missing values with the mean
    2. Performing SSA decomposition
    3. Reconstructing the signal using first n_components
    4. Optionally filtering by frequency
    5. Updating missing values and repeating until convergence

    The convergence is determined by the change in RMSE between iterations
    falling below the specified tolerance.

    Examples
    --------
    >>> import numpy as np
    >>> from ssalib.datasets import load_sst
    >>> # Create sample data with missing values
    >>> sst = load_sst()
    >>> sst_with_gaps = sst.copy()
    >>> sst_with_gaps.iloc[400:450] = np.nan
    >>> # Fill missing values using 10 components
    >>> filled_ts = fill_na_basic(sst_with_gaps, n_components=10, window=125)
    """
    # Input validation
    if not isinstance(n_components, int) or n_components <= 0:
        raise ValueError("n_components must be a positive integer")
    if not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    if not isinstance(tol, (int, float)) or tol <= 0:
        raise ValueError("tol must be a positive number")

    # Check for missing values before initialization
    original_na_mask = np.isnan(np.asarray(timeseries))
    if not np.any(original_na_mask):
        logger.warning("No missing values found in time series")
        return timeseries

    # Initialize SSA
    ssa = SingularSpectrumAnalysis(
        timeseries,
        window=window,
        svd_matrix_kind=svd_matrix_kind,
        standardize=standardize,
        svd_solver='sk_rsvd',  # Use faster SVD solver for large matrices
        na_strategy='fill_mean'
    )

    # Iterative filling algorithm
    rmse_last = float('inf')
    n_iter = 0

    while True:
        n_iter += 1
        logger.debug(f"Starting iteration {n_iter}")

        # Decompose
        ssa.decompose(n_components=n_components)

        # Select components based on frequency if requested
        if max_frequency is not None:
            frequencies = ssa.get_dominant_frequencies(
                n_components=n_components)
            selected_indices = [int(i) for i in
                                np.where(frequencies < max_frequency)[0]]
            if len(selected_indices) == 0:
                raise ValueError(
                    "No components found below the maximum frequency")
            signal = ssa[selected_indices]
        else:
            signal = ssa[:n_components]

        # Update missing values
        ts_pp = ssa._timeseries_pp.copy()  # Make a copy to prevent modifying the original
        ts_pp[original_na_mask] = signal[original_na_mask]
        ts_filled = ts_pp * ssa.std_ + ssa.mean_

        # Check convergence using original mask
        rmse = np.sqrt(((ts_filled[original_na_mask] - ssa._timeseries_pp[
            original_na_mask]) ** 2).mean())
        diff = abs(rmse - rmse_last)
        logger.debug(f"RMSE: {rmse:.6f}, Change: {diff:.6f}")

        if diff <= tol:
            logger.info(f"Converged after {n_iter} iterations")
            break

        if n_iter >= max_iter:
            raise RuntimeError(
                f"Failed to converge within {max_iter} iterations. "
                f"Last RMSE change: {diff:.6f}"
            )

        rmse_last = rmse
        ssa._timeseries_pp = ts_pp

    # Prepare return value
    if return_as == 'ssa':
        result = ssa
    elif return_as == 'signal':
        result = signal * ssa.std_ + ssa.mean_
    else:  # return_as == 'filled'
        result = ts_filled

    # Convert to pandas Series if input had an index
    if not isinstance(result, SingularSpectrumAnalysis) and ssa._ix is not None:
        result = pd.Series(result, index=ssa._ix)

    return result


def fill_na_monte_carlo(
        timeseries: ArrayLike,
        max_n_component: int = 50,
        window: Optional[int] = None,
        svd_matrix_kind: str = 'bk_trajectory',
        standardize: bool = True,
        n_surrogates: int = 100,
        ar_order_max: int = 1,
        confidence_level: float = 0.95,
        two_tailed: bool = True,
        max_frequency: Optional[float] = None,
        tol: float = 1e-3,
        max_iter: int = 100,
        return_as: Literal['signal', 'ssa', 'filled'] = 'filled',
        random_seed: int | None = None
) -> Union[np.ndarray, pd.Series, MonteCarloSSA]:
    """Fill missing values using Monte Carlo SSA with significance testing.

    This function implements an iterative algorithm to fill missing values using
    Monte Carlo SSA. The algorithm uses significance testing to automatically
    select components for reconstruction.

    Parameters
    ----------
    timeseries : ArrayLike
        Input time series data containing NaN values to be filled
    max_n_component : int, default=50
        Maximum number of components to consider for testing
    window : Optional[int], default=None
        Window length for SSA. If None, uses half the series length
    svd_matrix_kind : str, default='bk_trajectory'
        Type of matrix for SVD ('bk_trajectory', 'bk_covariance', or 'vg_covariance')
    standardize : bool, default=True
        Whether to standardize the time series
    n_surrogates : int, default=100
        Number of surrogate series to generate
    ar_order_max : int, default=1
        Maximum order for autoregressive model
    confidence_level : float, default=0.95
        Confidence level for significance testing
    two_tailed : bool, default=True
        Whether to use two-tailed significance testing
    max_frequency : Optional[float], default=None
        Maximum frequency to consider for component selection
    tol : float, default=1e-3
        Tolerance for convergence of RMSE
    max_iter : int, default=100
        Maximum number of iterations
    return_as : Literal['signal', 'ssa', 'filled'], default='filled'
        Format of the returned result
    random_seed : int | None, default=None
        Random seed for reproducibility

    Returns
    -------
    Union[np.ndarray, pd.Series, MonteCarloSSA]
        Filled time series or MonteCarloSSA object, depending on return_as

    Notes
    -----
    The algorithm performs these steps:
    1. Initialize missing values with mean
    2. Fit autoregressive model and generate surrogates
    3. Select significant components through Monte Carlo testing
    4. Reconstruct signal using selected components
    5. Update missing values and iterate until convergence
    """
    # Input validation
    if not isinstance(max_n_component, int) or max_n_component <= 0:
        raise ValueError("max_n_component must be a positive integer")
    if not isinstance(n_surrogates, int) or n_surrogates <= 0:
        raise ValueError("n_surrogates must be a positive integer")
    if not isinstance(ar_order_max, int) or ar_order_max < 0:
        raise ValueError("ar_order_max must be a non-negative integer")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    # Check for missing values before initialization
    original_na_mask = np.isnan(np.asarray(timeseries))
    if not np.any(original_na_mask):
        logger.warning("No missing values found in time series")
        return timeseries

    # Initialize Monte Carlo SSA
    mcssa = MonteCarloSSA(
        timeseries=timeseries,
        window=window,
        svd_matrix_kind=svd_matrix_kind,
        standardize=standardize,
        svd_solver='sklearn_randomized',
        # Use faster SVD solver for large matrices
        n_surrogates=n_surrogates,
        ar_order_max=ar_order_max,
        na_strategy='fill_mean',
        random_seed=random_seed
    )

    # Iterative filling algorithm
    rmse_last = float('inf')
    n_iter = 0

    while True:
        n_iter += 1
        logger.debug(f"Starting iteration {n_iter}")

        # Decompose and test significance
        mcssa.decompose(n_components=max_n_component, random_state=random_seed)
        is_significant = mcssa.test_significance(
            confidence_level=confidence_level,
            two_tailed=two_tailed
        )

        # Apply frequency filtering if requested
        if max_frequency is not None:
            frequencies = mcssa.get_dominant_frequencies(
                n_components=max_n_component)
            is_significant = np.logical_and(is_significant,
                                            frequencies < max_frequency)

        selected_indices = [int(i) for i in np.where(is_significant)[0]]
        if len(selected_indices) == 0:
            raise ValueError(
                "No significant components found. Consider adjusting parameters:"
                "\n- Decrease confidence_level"
                "\n- Increase max_n_component"
                "\n- Increase max_frequency"
                "\n- Switch to basic SSA strategy"
            )

        logger.debug(f"Selected {len(selected_indices)} significant components")

        # Reconstruct signal using significant components
        signal = mcssa[selected_indices]
        ts_pp = mcssa._timeseries_pp.copy()  # Make a copy to prevent modifying the original
        ts_pp[original_na_mask] = signal[original_na_mask]
        ts_filled = ts_pp * mcssa.std_ + mcssa.mean_

        # Check convergence using original mask
        rmse = np.sqrt(((ts_filled[original_na_mask] - mcssa._timeseries_pp[
            original_na_mask]) ** 2).mean())
        diff = abs(rmse - rmse_last)
        logger.debug(f"RMSE: {rmse:.6f}, Change: {diff:.6f}")

        if diff <= tol:
            logger.info(
                f"Converged after {n_iter} iterations with {len(selected_indices)} "
                f"significant components"
            )
            break

        if n_iter >= max_iter:
            logger.warning(
                f"Reached maximum iterations ({max_iter}) without convergence. "
                f"Last RMSE change: {diff:.6f}"
            )
            break

        rmse_last = rmse
        mcssa._timeseries_pp = ts_pp

    # Prepare return value based on requested format
    if return_as == 'ssa':
        result = mcssa
    elif return_as == 'signal':
        result = signal * mcssa.std_ + mcssa.mean_
    else:  # return_as == 'filled'
        result = ts_filled

    # Convert to pandas Series if input had an index
    if not isinstance(result, MonteCarloSSA) and mcssa._ix is not None:
        result = pd.Series(result, index=mcssa._ix)

    return result


if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    # Example usage comparing both strategies
    from ssalib.datasets import load_sst
    import matplotlib.pyplot as plt

    # Load data and create gaps
    gap_start_ix, gap_end_ix = 400, 450
    sst = load_sst()
    sst_with_gaps = sst.copy()
    sst_with_gaps.iloc[gap_start_ix:gap_end_ix] = np.nan

    # Fill gaps using both strategies
    filled_basic = fill_na(sst_with_gaps, strategy='basic', n_components=10)
    filled_mc = fill_na(sst_with_gaps, strategy='monte_carlo',
                        n_surrogates=100,
                        ar_order_max=1,
                        n_components=10,  # Try 0
                        random_seed=42)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Basic strategy plot
    sst.plot(ax=ax1, label='Original', zorder=1)
    filled_basic.iloc[gap_start_ix:gap_end_ix].plot(ax=ax1,
                                                    label='Filled with Basic SSA',
                                                    zorder=2)
    ax1.fill_betweenx(y=[sst.min(), sst.max()], x1=sst.index[gap_start_ix],
                      x2=sst.index[gap_end_ix], edgecolor='none',
                      facecolor='lightgrey', zorder=0)
    ax1.set_title('Basic SSA Strategy')
    ax1.legend()

    # Monte Carlo strategy plot
    # Note that in that case, the strategy is bad because the sst trend
    # in not captured as a significant AR1 component. Try ar_max_order set
    # to 0.
    sst.plot(ax=ax2, label='Original', zorder=1)
    # sst_with_gaps.plot(ax=ax2, label='With Gaps', linewidth=2, zorder=3)
    filled_mc.iloc[gap_start_ix:gap_end_ix].plot(
        ax=ax2,
        label='Filled with Monte Carlo SSA', zorder=2)
    ax2.fill_betweenx(y=[sst.min(), sst.max()], x1=sst.index[gap_start_ix],
                      x2=sst.index[gap_end_ix], edgecolor='none',
                      facecolor='lightgrey', zorder=0)
    ax2.set_title('Monte Carlo SSA Strategy')
    ax2.legend()

    plt.tight_layout()
    plt.show()
