"""Utilities for loading tabular data files in Streamlit applications.

Provides functions to handle various file formats and convert them into pandas DataFrames.
Supports both Path compatible pathnames and Streamlit UploadedFile objects.
"""

import os
from io import BytesIO
from pathlib import Path

import pandas as pd
from streamlit.runtime.uploaded_file_manager import UploadedFile

TABULAR_FILE_FORMATS_READERS = {
    "csv": pd.read_csv,
    # "xls": pd.read_excel,  # Could be read but need additional import
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
}


def load_tabular_data(file_or_filename: Path | UploadedFile, **kwargs) -> pd.DataFrame:
    """Load tabular data from a file path or Streamlit UploadedFile object.

    Args:
        file_or_filename: Either a Path object pointing to a file or a Streamlit UploadedFile
        **kwargs: Additional arguments to pass to the pandas reader function

    Returns:
        A pandas DataFrame containing the loaded data

    Raises:
        ValueError: If the file doesn't exist or has an unsupported format
    """
    if isinstance(file_or_filename, Path):
        assert file_or_filename.exists()
        loaded_file = BytesIO(file_or_filename.read_bytes())
        loaded_file.name = file_or_filename.name
    elif isinstance(file_or_filename, UploadedFile):
        loaded_file = file_or_filename
    else:
        raise ValueError(f"incorrect file: {file_or_filename}")
    ext = os.path.splitext(loaded_file.name)[1][1:].lower()
    if ext in TABULAR_FILE_FORMATS_READERS:
        return TABULAR_FILE_FORMATS_READERS[ext](loaded_file, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
