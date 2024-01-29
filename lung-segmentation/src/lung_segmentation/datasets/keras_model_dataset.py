from pathlib import Path, PurePosixPath
from typing import Any, Dict

import os
import fsspec
import keras
import tensorflow as tf

from kedro.io import AbstractDataset
from kedro.io.core import get_filepath_str, get_protocol_and_path

class KerasModelDataset(AbstractDataset[keras.models.Model, keras.models.Model]):
    def __init__(self, filepath: str):
        super().__init__()
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol)

    def _load(self) -> keras.models.Model:
        load_path = get_filepath_str(self._filepath, self._protocol)
        with self._fs.open(load_path) as f:
            return tf.keras.models.load_model(f)

    def _save(self, data: keras.models.Model) -> None:
        save_path = get_filepath_str(self._filepath, self._protocol)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        data.save(save_path)

    def _exists(self) -> bool:
        return Path(self._filepath.as_posix()).exists()

    def _describe(self) -> Dict[str, Any]:
        return dict(filepath=self._filepath, protocol=self._protocol)
    