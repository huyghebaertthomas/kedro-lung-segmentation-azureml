from pathlib import Path, PurePosixPath
from typing import Any, Dict

import os
import fsspec
import numpy as np

from kedro.io import AbstractDataset
from kedro.io.core import get_filepath_str, get_protocol_and_path

class NumpyDataset(AbstractDataset[np.ndarray, np.ndarray]):
    def __init__(self, filepath: str):
        super().__init__()
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol)
        #self._filepath = PurePosixPath(filepath)

    def _load(self) -> np.ndarray:
        load_path = get_filepath_str(self._filepath, self._protocol)
        with self._fs.open(load_path) as f:
            np_array = np.load(f)
            return np_array
        #return np.load(self._filepath)
        
    def _save(self, data: np.ndarray) -> None:
        save_path = get_filepath_str(self._filepath, self._protocol)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, data)

    def _exists(self) -> bool:
        return Path(self._filepath.as_posix()).exists()

    def _describe(self) -> Dict[str, Any]:
        return dict(filepath=self._filepath, protocol=self._protocol)
