import os

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