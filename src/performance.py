# src/performance.py
import gc
import pandas as pd
from tqdm.notebook import tqdm
from typing import Dict, List, Callable, Any, Union, TypeVar

from src.timing import measure_execution_time

T = TypeVar('T')  # Type for function input
R = TypeVar('R')  # Type for function output

def test_algorithm_performance(
    algorithms: Dict[str, Callable[[T], R]],
    test_values: List[T],
    param_name: str = "N",
    trials: int = 10,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Test the performance of multiple algorithms with various input values.

    This function evaluates the execution time of multiple algorithms against a set
    of test values, collecting results in a DataFrame for analysis. It handles errors
    and overflow conditions gracefully.

    :param algorithms: Dictionary mapping algorithm names to their corresponding functions
    :type algorithms: Dict[str, Callable[[T], R]]
    :param test_values: List of input values to test the algorithms with
    :type test_values: List[T]
    :param param_name: Name of the parameter being tested (used in the results DataFrame)
    :type param_name: str
    :param trials: Number of trials to run for each algorithm and input value
    :type trials: int
    :param show_progress: Whether to display a progress bar during execution
    :type show_progress: bool

    :return: DataFrame containing execution times for each algorithm and input value
    :rtype: pd.DataFrame
    """

    results = []

    # Use progress bar if requested
    iterator = tqdm(test_values, desc=f"Testing {param_name} values", unit=param_name.lower()) if show_progress else test_values

    for val in iterator:
        row = {param_name: val}
        gc.collect()  # Clean up memory before each test

        if show_progress:
            iterator.set_description(f"{param_name}={val}")

        for name, func in algorithms.items():
            try:
                result, exec_time = measure_execution_time(func, val, trials=trials)
                row[name] = "OVERFLOW" if result == "OVERFLOW" else exec_time
            except Exception as e:
                if show_progress:
                    print(f"\nError with {name} for {param_name}={val}: {e}")
                row[name] = "ERROR"

        results.append(row)

    return pd.DataFrame(results)