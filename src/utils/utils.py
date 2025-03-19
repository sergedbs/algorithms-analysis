import os
import random
from typing import Dict, List, Literal, Callable, TypeVar, Any
import numpy as np


def create_directory(path):
    """
    Check if a directory exists, if not, create it.

    :param path: Directory path to be checked or created
    :type path: str
    """
    path = os.path.normpath(path)
    if not os.path.exists(path):
        os.makedirs(path)


def export_results(df, path, filename_prefix="results"):
    """
    Export pandas DataFrame results to a CSV file.

    :param df: DataFrame to be exported
    :type df: pd.DataFrame
    :param path: Directory path where the file will be saved
    :type path: str
    :param filename_prefix: Prefix for the filename (default: "results")
    :type filename_prefix: str

    :return: Absolute path to the saved file
    :rtype: str
    """
    create_directory(path)
    filename = os.path.join(path, f"{filename_prefix}.csv")
    df.to_csv(filename, index=False)
    return os.path.abspath(filename)


def generate_test_inputs(
        sizes: List[int],
        input_types: List[Literal["sorted", "reversed", "random", "duplicates", "nearly_sorted"]],
        random_seed: int = None
) -> Dict[str, Dict[int, List[int]]]:
    """
    Generate test datasets of different sizes and patterns for sorting algorithm analysis.

    :param sizes: List of array sizes to generate
    :type sizes: List[int]
    :param input_types: List of input patterns to generate
    :type input_types: List[Literal["sorted", "reversed", "random", "duplicates", "nearly_sorted"]]
    :param random_seed: Optional seed for random number generation
    :type random_seed: int, optional

    :return: A nested dictionary {input_type: {size: data}}
    :rtype: Dict[str, Dict[int, List[int]]]
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    dataset = {}
    valid_types = ["sorted", "reversed", "random", "duplicates", "nearly_sorted"]

    # Validate input types
    for input_type in input_types:
        if input_type not in valid_types:
            raise ValueError(f"Invalid input type: {input_type}. Must be one of {valid_types}")

    for input_type in input_types:
        dataset[input_type] = {}
        for size in sizes:
            if size <= 0:
                dataset[input_type][size] = []
                continue

            if input_type == "sorted":
                data = list(range(size))
            elif input_type == "reversed":
                data = list(range(size - 1, -1, -1))
            elif input_type == "random":
                # Using numpy for faster generation of large arrays
                if size > 10000:
                    data = np.random.choice(size * 10, size, replace=False).tolist()
                else:
                    data = random.sample(range(size * 10), size)
            elif input_type == "duplicates":
                data = [random.randint(0, 9) for _ in range(size)]
            elif input_type == "nearly_sorted":
                # Create a mostly sorted array with some swaps
                data = list(range(size))
                swaps = min(size // 10, 100)  # Swap about 10% of elements
                for _ in range(swaps):
                    i, j = random.sample(range(size), 2)
                    data[i], data[j] = data[j], data[i]

            dataset[input_type][size] = data

    return dataset


T = TypeVar('T')  # Type for function input


def sort(arr: List[T], func: Callable[[List[T]], Any], copy: bool = True) -> List[T]:
    """
    Apply a sorting function to an array, with option to work on a copy.

    :param arr: Input array to sort
    :type arr: List[T]
    :param func: Sorting function to apply
    :type func: Callable[[List[T]], Any]
    :param copy: Whether to create a copy before sorting (default: True)
    :type copy: bool

    :return: Sorted array
    :rtype: List[T]
    """
    if copy:
        # Create a copy to avoid modifying the original
        array = arr.copy()
        result = func(array)
        # Handle both in-place sorts (returning None) and functional sorts
        return array if result is None else result
    else:
        # Apply directly to original array
        result = func(arr)
        return arr if result is None else result
