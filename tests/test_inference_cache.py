from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.inference_cache import (
    load_json_object,
    save_json_object,
    save_npy,
    save_csv,
    validate_cached_ids,
)


def test_atomic_cache_writers_create_parent_and_validate_shape(tmp_path: Path):
    target = tmp_path / "nested" / "ids.npy"
    values = np.arange(6, dtype=np.int32).reshape(2, 3)
    save_npy(target, values)
    validate_cached_ids(target, n_rows=2, max_length=3)

    metadata = target.with_suffix(".json")
    save_json_object(metadata, {"n": 2})
    assert load_json_object(metadata) == {"n": 2}

    with pytest.raises(ValueError, match="shape 错误"):
        validate_cached_ids(target, n_rows=3, max_length=2)

    csv_path = tmp_path / "result" / "rows.csv"
    save_csv(csv_path, pd.DataFrame({"value": [1, 2]}))
    assert pd.read_csv(csv_path)["value"].tolist() == [1, 2]
