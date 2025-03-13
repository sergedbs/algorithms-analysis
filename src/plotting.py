# Path: src/plotting.py
# Description: Module for plotting performance graphs for algorithm benchmarks.
# Acknowledgment: Documentation generated and code optimized using Claude 3.7 Sonnet.

import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any, Tuple

from matplotlib.figure import Figure

from src.utils import create_directory


def plot_performance_graphs(
    df: pd.DataFrame,
    path: str,
    param_name: str = "N",
    time_unit: str = "seconds",
    all_algorithms: Optional[List[str]] = None,
    fast_algorithms: Optional[List[str]] = None,
    title_prefix: str = "Algorithms",
    filename_prefix: str = "performance",
    small_threshold: int = 100,
    large_threshold: int = 200,
    convert_to_seconds: bool = True,
    rolling_window: int = 5,
    plot_formats: Optional[List[str]] = None,
    plot_dpi: int = 300,
    figsize: tuple = (12, 8),
    grid_style: Optional[Dict[str, Any]] = None,
    display_plots: bool = False,
    return_figures: bool = False
) -> list[Figure] | None:
    """
    Plot performance graphs for algorithm benchmarks.

    :param df: DataFrame containing benchmark results
    :param path: Directory path where plots will be saved
    :param param_name: Name of the parameter column (default: "N")
    :param time_unit: Unit for the time measurement (default: "seconds")
    :param all_algorithms: List of all algorithms to include (default: None, uses all columns except param_name)
    :param fast_algorithms: List of faster algorithms for large input plots (default: None, uses all columns except param_name)
    :param title_prefix: Prefix for plot titles (default: "Algorithms")
    :param filename_prefix: Prefix for saved files (default: "performance")
    :param small_threshold: Threshold for small input values (default: 100)
    :param large_threshold: Threshold for large input values (default: 200)
    :param convert_to_seconds: Whether to convert nanoseconds to seconds (default: True)
    :param rolling_window: Window size for rolling average (default: 5)
    :param plot_formats: List of formats to save plots in (default: ["png"])
    :param plot_dpi: DPI for saved plots (default: 300)
    :param figsize: Figure size (width, height) in inches (default: (12, 8))
    :param grid_style: Dictionary with grid style parameters (default: uses predefined style)
    :param display_plots: Whether to display plots in the notebook/output (default: False)
    :param return_figures: Whether to return figure objects (default: False)

    :return: List of paths to the saved plot files or tuple of (paths, figures) if return_figures is True
    """
    create_directory(path)

    # Set default values
    if plot_formats is None:
        plot_formats = ["png"]

    if grid_style is None:
        grid_style = {"which": "both", "linestyle": "--", "alpha": 0.3}

    # Determine algorithms to plot
    all_algorithms = all_algorithms or [col for col in df.columns if col != param_name]
    fast_algorithms = fast_algorithms or all_algorithms

    # Generate and save all plots
    saved_files = []
    all_figures = []

    # Small input plot
    small_result = _create_small_input_plot(
        df, path, param_name, time_unit, all_algorithms,
        title_prefix, filename_prefix, small_threshold,
        convert_to_seconds, rolling_window, plot_formats,
        plot_dpi, figsize, grid_style, display_plots
    )

    saved_files.extend(small_result[0])
    if return_figures:
        all_figures.append(small_result[1])

    # Large input plot
    large_result = _create_large_input_plot(
        df, path, param_name, time_unit, fast_algorithms,
        title_prefix, filename_prefix, large_threshold,
        convert_to_seconds, rolling_window, plot_formats,
        plot_dpi, figsize, grid_style, display_plots
    )

    saved_files.extend(large_result[0])
    if return_figures:
        all_figures.append(large_result[1])

    # Linear scale plot
    linear_result = _create_linear_scale_plot(
        df, path, param_name, time_unit, fast_algorithms,
        title_prefix, filename_prefix, convert_to_seconds,
        rolling_window, plot_formats, plot_dpi,
        figsize, grid_style, display_plots
    )

    saved_files.extend(linear_result[0])
    if return_figures:
        all_figures.append(linear_result[1])

    return all_figures if return_figures else None


def _prepare_data_for_plotting(
    df: pd.DataFrame,
    param_name: str,
    method: str,
    condition: Optional[pd.Series] = None,
    convert_to_seconds: bool = True,
    rolling_window: int = 5
) -> pd.DataFrame:
    """
    Prepare data for plotting by filtering, converting and smoothing.

    :param df: Input DataFrame
    :param param_name: Name of parameter column
    :param method: Algorithm method name
    :param condition: Additional filtering condition (default: None)
    :param convert_to_seconds: Whether to convert nanoseconds to seconds
    :param rolling_window: Window size for rolling average

    :return: Prepared DataFrame for plotting
    """
    # Base condition to filter out non-numeric and error values
    base_condition = (
        df[method].notna() &
        (df[method] != "OVERFLOW") &
        (df[method] != "ERROR")
    )

    # Combine with additional condition if provided
    if condition is not None:
        final_condition = base_condition & condition
    else:
        final_condition = base_condition

    # Extract and prepare data
    valid_data = df[final_condition].copy()

    if valid_data.empty:
        return pd.DataFrame()

    # Convert to numeric and process
    valid_data.loc[:, method] = valid_data[method].astype(float)

    if convert_to_seconds:
        valid_data.loc[:, method] = valid_data[method] / 1e9

    # Apply rolling average for smoothing
    valid_data.loc[:, method] = valid_data[method].rolling(
        window=rolling_window,
        center=True,
        min_periods=1
    ).mean()

    # Sort by parameter value
    return valid_data.sort_values(param_name)


def _create_plot(
    df: pd.DataFrame,
    param_name: str,
    methods: List[str],
    condition: Optional[pd.Series] = None,
    convert_to_seconds: bool = True,
    rolling_window: int = 5,
    figsize: tuple = (12, 8),
    log_scale: bool = True,
    title: str = "Algorithm Performance",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    grid_style: Optional[Dict[str, Any]] = None
) -> plt.Figure:
    """
    Create a performance plot with the specified settings.

    :param df: Input DataFrame
    :param param_name: Name of parameter column
    :param methods: List of algorithm methods to plot
    :param condition: Additional filtering condition
    :param convert_to_seconds: Whether to convert nanoseconds to seconds
    :param rolling_window: Window size for rolling average
    :param figsize: Figure size (width, height) in inches
    :param log_scale: Whether to use logarithmic scale for y-axis
    :param title: Plot title
    :param xlabel: X-axis label (default: based on param_name)
    :param ylabel: Y-axis label (default: based on time_unit)
    :param grid_style: Grid style parameters

    :return: Matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each method
    for method in methods:
        if method in df.columns:
            valid_data = _prepare_data_for_plotting(
                df, param_name, method, condition,
                convert_to_seconds, rolling_window
            )

            if not valid_data.empty:
                ax.plot(
                    valid_data[param_name],
                    valid_data[method],
                    label=method,
                    marker=".",
                    markersize=2.5,
                    linestyle='-',
                    linewidth=2,
                    alpha=0.8
                )

    # Set labels and title
    ax.set_xlabel(xlabel or f"Input Size ({param_name})", fontsize=12)
    ax.set_ylabel(ylabel or "Execution Time (seconds)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Set scale
    if log_scale:
        ax.set_yscale("log")

    # Add legend
    ax.legend(
        title="Algorithms",
        fontsize=10,
        title_fontsize=11,
        framealpha=0.9,
        edgecolor='gray'
    )

    # Add grid
    if grid_style:
        ax.grid(True, **grid_style)

    plt.tight_layout()
    return fig


def _save_plot(
    fig: plt.Figure,
    path: str,
    filename: str,
    formats: List[str],
    dpi: int,
    display_plot: bool = False
) -> Tuple[List[str], plt.Figure]:
    """
    Save a plot in multiple formats and optionally display it.

    :param fig: Matplotlib figure to save
    :param path: Directory path to save to
    :param filename: Base filename without extension
    :param formats: List of file formats to save as
    :param dpi: Resolution for saved images
    :param display_plot: Whether to display the plot

    :return: Tuple of (list of saved file paths, figure object)
    """
    saved_files = []
    for fmt in formats:
        full_path = os.path.join(path, f"{filename}.{fmt}")
        fig.savefig(full_path, dpi=dpi, bbox_inches="tight")
        saved_files.append(full_path)

    # Optionally display the plot
    if display_plot:
        plt.show()
    else:
        plt.close(fig)

    return saved_files, fig


def _create_small_input_plot(
    df: pd.DataFrame,
    path: str,
    param_name: str,
    time_unit: str,
    algorithms: List[str],
    title_prefix: str,
    filename_prefix: str,
    threshold: int,
    convert_to_seconds: bool,
    rolling_window: int,
    formats: List[str],
    dpi: int,
    figsize: tuple,
    grid_style: Dict[str, Any],
    display_plot: bool = False
) -> Tuple[List[str], plt.Figure]:
    """Create and save plot for small input values."""
    condition = df[param_name] <= threshold
    title = f"{title_prefix} Performance ({param_name} ≤ {threshold})"
    ylabel = f"Execution Time ({time_unit})"

    fig = _create_plot(
        df, param_name, algorithms, condition, convert_to_seconds,
        rolling_window, figsize, True, title,
        f"Input Size ({param_name})", ylabel, grid_style
    )

    return _save_plot(fig, path, f"{filename_prefix}_all_small", formats, dpi, display_plot)


def _create_large_input_plot(
    df: pd.DataFrame,
    path: str,
    param_name: str,
    time_unit: str,
    algorithms: List[str],
    title_prefix: str,
    filename_prefix: str,
    threshold: int,
    convert_to_seconds: bool,
    rolling_window: int,
    formats: List[str],
    dpi: int,
    figsize: tuple,
    grid_style: Dict[str, Any],
    display_plot: bool = False
) -> Tuple[List[str], plt.Figure]:
    """Create and save plot for large input values with log scale."""
    condition = df[param_name] >= threshold
    title = f"{title_prefix} Performance ({param_name} ≥ {threshold})"
    ylabel = f"Execution Time ({time_unit})"

    fig = _create_plot(
        df, param_name, algorithms, condition, convert_to_seconds,
        rolling_window, figsize, True, title,
        f"Input Size ({param_name})", ylabel, grid_style
    )

    return _save_plot(fig, path, f"{filename_prefix}_fast_log", formats, dpi, display_plot)


def _create_linear_scale_plot(
    df: pd.DataFrame,
    path: str,
    param_name: str,
    time_unit: str,
    algorithms: List[str],
    title_prefix: str,
    filename_prefix: str,
    convert_to_seconds: bool,
    rolling_window: int,
    formats: List[str],
    dpi: int,
    figsize: tuple,
    grid_style: Dict[str, Any],
    display_plot: bool = False
) -> Tuple[List[str], plt.Figure]:
    """Create and save plot with linear scale for all input values."""
    title = f"{title_prefix} Performance (Linear Scale)"
    ylabel = f"Execution Time ({time_unit})"

    fig = _create_plot(
        df, param_name, algorithms, None, convert_to_seconds,
        rolling_window, figsize, False, title,
        f"Input Size ({param_name})", ylabel, grid_style
    )

    return _save_plot(fig, path, f"{filename_prefix}_linear", formats, dpi, display_plot)


# Convenience wrapper function for direct plotting without saving
def display_performance_graphs(
    df: pd.DataFrame,
    param_name: str = "N",
    time_unit: str = "seconds",
    all_algorithms: Optional[List[str]] = None,
    fast_algorithms: Optional[List[str]] = None,
    title_prefix: str = "Algorithms",
    small_threshold: int = 100,
    large_threshold: int = 200,
    convert_to_seconds: bool = True,
    rolling_window: int = 5,
    figsize: tuple = (12, 8),
    grid_style: Optional[Dict[str, Any]] = None
) -> List[plt.Figure]:
    """
    Display performance graphs for algorithm benchmarks without saving files.
    Useful for interactive exploration in notebooks.

    :param df: DataFrame containing benchmark results
    :param param_name: Name of the parameter column (default: "N")
    :param time_unit: Unit for the time measurement (default: "seconds")
    :param all_algorithms: List of all algorithms to include
    :param fast_algorithms: List of faster algorithms for large input plots
    :param title_prefix: Prefix for plot titles (default: "Algorithms")
    :param small_threshold: Threshold for small input values (default: 100)
    :param large_threshold: Threshold for large input values (default: 200)
    :param convert_to_seconds: Whether to convert nanoseconds to seconds (default: True)
    :param rolling_window: Window size for rolling average (default: 5)
    :param figsize: Figure size (width, height) in inches (default: (12, 8))
    :param grid_style: Dictionary with grid style parameters

    :return: List of figure objects
    """

    # Set default grid style
    if grid_style is None:
        grid_style = {"which": "both", "linestyle": "--", "alpha": 0.3}

    # Determine algorithms to plot
    all_algorithms = all_algorithms or [col for col in df.columns if col != param_name]
    fast_algorithms = fast_algorithms or all_algorithms

    figures = []

    # Small input plot
    condition = df[param_name] <= small_threshold
    title = f"{title_prefix} Performance ({param_name} ≤ {small_threshold})"
    ylabel = f"Execution Time ({time_unit})"

    fig_small = _create_plot(
        df, param_name, all_algorithms, condition, convert_to_seconds,
        rolling_window, figsize, True, title,
        f"Input Size ({param_name})", ylabel, grid_style
    )
    figures.append(fig_small)
    plt.show()

    # Large input plot
    condition = df[param_name] >= large_threshold
    title = f"{title_prefix} Performance ({param_name} ≥ {large_threshold})"

    fig_large = _create_plot(
        df, param_name, fast_algorithms, condition, convert_to_seconds,
        rolling_window, figsize, True, title,
        f"Input Size ({param_name})", ylabel, grid_style
    )
    figures.append(fig_large)
    plt.show()

    # Linear scale plot
    title = f"{title_prefix} Performance (Linear Scale)"

    fig_linear = _create_plot(
        df, param_name, fast_algorithms, None, convert_to_seconds,
        rolling_window, figsize, False, title,
        f"Input Size ({param_name})", ylabel, grid_style
    )
    figures.append(fig_linear)
    plt.show()

    return figures