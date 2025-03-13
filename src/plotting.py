import pandas as pd
import matplotlib.pyplot as plt

from src.utils import create_directory

def plot_performance_graphs(
    df,
    path,
    param_name="N",
    time_unit="seconds",
    all_algorithms=None,
    fast_algorithms=None,
    title_prefix="Algorithms",
    filename_prefix="performance",
    small_threshold=100,
    large_threshold=200,
    convert_to_seconds=True,
    rolling_window=5
):
    """
    Plot performance graphs for algorithm benchmarks.

    :param df: DataFrame containing benchmark results
    :type df: pd.DataFrame
    :param path: Directory path where plots will be saved
    :type path: str
    :param param_name: Name of the parameter column (default: "N")
    :type param_name: str
    :param time_unit: Unit for the time measurement (default: "seconds")
    :type time_unit: str
    :param all_algorithms: List of all algorithms to include (default: None, uses all columns except param_name)
    :type all_algorithms: list, optional
    :param fast_algorithms: List of faster algorithms for large input plots (default: None, uses all columns except param_name)
    :type fast_algorithms: list, optional
    :param title_prefix: Prefix for plot titles (default: "Algorithms")
    :type title_prefix: str
    :param filename_prefix: Prefix for saved files (default: "performance")
    :type filename_prefix: str
    :param small_threshold: Threshold for small input values (default: 100)
    :type small_threshold: int
    :param large_threshold: Threshold for large input values (default: 200)
    :type large_threshold: int
    :param convert_to_seconds: Whether to convert nanoseconds to seconds (default: True)
    :type convert_to_seconds: bool
    :param rolling_window: Window size for rolling average (default: 5)
    :type rolling_window: int
    """
    create_directory(path)

    # If algorithm lists are not provided, use all columns except param_name
    if all_algorithms is None:
        all_algorithms = [col for col in df.columns if col != param_name]
    if fast_algorithms is None:
        fast_algorithms = all_algorithms

    # Plot 1: Small inputs with all algorithms (log scale)
    plt.figure(figsize=(12, 8))
    for method in all_algorithms:
        if method in df.columns:
            valid_data = df[(df[param_name] <= small_threshold) &
                           df[method].notna() &
                           (df[method] != "OVERFLOW") &
                           (df[method] != "ERROR")].copy()
            if not valid_data.empty:
                valid_data.loc[:, method] = valid_data[method].astype(float)
                if convert_to_seconds:
                    valid_data.loc[:, method] = valid_data[method] / 1e9
                valid_data.loc[:, method] = valid_data[method].rolling(
                    window=rolling_window, center=True, min_periods=1).mean()
                valid_data = valid_data.sort_values(param_name)

                plt.plot(valid_data[param_name], valid_data[method],
                         label=method, markersize=2.5,
                         marker=".", linestyle='-',
                         linewidth=2, alpha=0.8)

    plt.xlabel(f"Input Size ({param_name})", fontsize=12)
    plt.ylabel(f"Execution Time ({time_unit})", fontsize=12)
    plt.yscale("log")
    plt.title(f"{title_prefix} Performance ({param_name} ≤ {small_threshold})",
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(title="Algorithms", fontsize=10,
               title_fontsize=11, framealpha=0.9,
               edgecolor='gray')
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{path}/{filename_prefix}_all_small.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 2: Large inputs with fast algorithms (log scale)
    plt.figure(figsize=(12, 8))
    for method in fast_algorithms:
        if method in df.columns:
            valid_data = df[(df[param_name] >= large_threshold) &
                           df[method].notna() &
                           (df[method] != "OVERFLOW") &
                           (df[method] != "ERROR")].copy()
            if not valid_data.empty:
                valid_data.loc[:, method] = valid_data[method].astype(float)
                if convert_to_seconds:
                    valid_data.loc[:, method] = valid_data[method] / 1e9
                valid_data.loc[:, method] = valid_data[method].rolling(
                    window=rolling_window, center=True, min_periods=1).mean()
                valid_data = valid_data.sort_values(param_name)

                plt.plot(valid_data[param_name], valid_data[method],
                         label=method, marker=".",
                         markersize=2.5, linestyle='-',
                         linewidth=2, alpha=0.8)

    plt.xlabel(f"Input Size ({param_name})", fontsize=12)
    plt.ylabel(f"Execution Time ({time_unit})", fontsize=12)
    plt.yscale("log")
    plt.title(f"{title_prefix} Performance ({param_name} ≥ {large_threshold})",
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(title="Algorithms", fontsize=10,
               title_fontsize=11, framealpha=0.9,
               edgecolor='gray')
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{path}/{filename_prefix}_fast_log.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 3: All inputs with fast algorithms (linear scale)
    plt.figure(figsize=(12, 8))
    for method in fast_algorithms:
        if method in df.columns:
            valid_data = df[df[method].notna() &
                          (df[method] != "OVERFLOW") &
                          (df[method] != "ERROR")].copy()
            if not valid_data.empty:
                valid_data.loc[:, method] = valid_data[method].astype(float)
                if convert_to_seconds:
                    valid_data.loc[:, method] = valid_data[method] / 1e9
                valid_data.loc[:, method] = valid_data[method].rolling(
                    window=rolling_window, center=True, min_periods=1).mean()
                valid_data = valid_data.sort_values(param_name)

                plt.plot(valid_data[param_name], valid_data[method],
                         label=method, marker=".",
                         markersize=2.5, linestyle='-',
                         linewidth=2, alpha=0.8)

    plt.xlabel(f"Input Size ({param_name})", fontsize=12)
    plt.ylabel(f"Execution Time ({time_unit})", fontsize=12)
    plt.title(f"{title_prefix} Performance (Linear Scale)",
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(title="Algorithms", fontsize=10,
               title_fontsize=11, framealpha=0.9,
               edgecolor='gray')
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{path}/{filename_prefix}_linear.png", dpi=300, bbox_inches="tight")
    plt.close()