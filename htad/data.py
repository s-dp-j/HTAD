from __future__ import annotations

import bisect
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class SeriesBundle:
    name: str
    train: List[np.ndarray]
    test: List[np.ndarray]
    labels: List[np.ndarray]
    feature_dim: int
    segment_names: Optional[List[str]] = None


def _natural_key(path: Path) -> Tuple[object, ...]:
    return tuple(int(x) if x.isdigit() else x for x in re.split(r"(\d+)", path.name))


def _as_float32(array: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(array, dtype=np.float32), copy=False)


def _split_array(values: np.ndarray, lengths: Sequence[int]) -> List[np.ndarray]:
    offsets = np.cumsum([0] + [int(value) for value in lengths], dtype=np.int64)
    if int(offsets[-1]) != len(values):
        raise ValueError(
            "NASA segment lengths sum to {}, but the array contains {} rows".format(
                int(offsets[-1]), len(values)
            )
        )
    return [values[int(start) : int(end)] for start, end in zip(offsets[:-1], offsets[1:])]


def _nasa_segments(root: Path, name: str, train_rows: int, test_rows: int):
    """Recover the original Telemanom entity boundaries when metadata exists.

    The commonly used SMAP/MSL benchmark arrays concatenate independent
    telemetry streams.  Treating the concatenation as one sequence creates
    cross-entity windows and invalid calibration statistics.  The official
    Telemanom CSV supplies the test order/lengths; the Hugging Face dataset
    mirror exposes the corresponding train split lengths.
    """
    labels_path = root / "labeled_anomalies.csv"
    sizes_path = root / "telemanom_size.json"
    if not labels_path.is_file() or not sizes_path.is_file():
        return None

    metadata = pd.read_csv(labels_path)
    rows = metadata[metadata["spacecraft"].astype(str).str.upper() == name].copy()
    # The source CSV contains P-2 twice.  The standard 427,617-row SMAP NPY
    # excludes both duplicated entries, which is verified by exact length.
    duplicated = rows["chan_id"].duplicated(keep=False)
    # The benchmark NPY files were created from lexicographically sorted file
    # names (for example D-11 precedes D-2), not from CSV row order or natural
    # numeric ordering.  This order reproduces every stored label exactly.
    rows = rows.loc[~duplicated].sort_values("chan_id", kind="mergesort")

    with sizes_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    split_rows = payload.get("size", {}).get("splits", [])
    train_lengths = {
        item["config"]: int(item["num_rows"])
        for item in split_rows
        if item.get("split") == "train"
    }
    test_lengths = {
        item["config"]: int(item["num_rows"])
        for item in split_rows
        if item.get("split") == "test"
    }
    names = rows["chan_id"].astype(str).tolist()
    if any(value not in train_lengths or value not in test_lengths for value in names):
        return None
    train_sizes = [train_lengths[value] for value in names]
    test_sizes = [test_lengths[value] for value in names]
    if sum(train_sizes) != int(train_rows) or sum(test_sizes) != int(test_rows):
        return None
    return names, train_sizes, test_sizes


def _load_nasa(root: Path, name: str) -> SeriesBundle:
    folder = root / name
    train = _as_float32(np.load(folder / f"{name}_train.npy"))
    test = _as_float32(np.load(folder / f"{name}_test.npy"))
    labels = np.asarray(np.load(folder / f"{name}_test_label.npy"), dtype=np.int64).reshape(-1)
    segments = _nasa_segments(root, name, len(train), len(test))
    if segments is None:
        return SeriesBundle(name, [train], [test], [labels], int(train.shape[1]), [name])
    names, train_lengths, test_lengths = segments
    return SeriesBundle(
        name,
        _split_array(train, train_lengths),
        _split_array(test, test_lengths),
        _split_array(labels, test_lengths),
        int(train.shape[1]),
        names,
    )


def _load_smd(root: Path) -> SeriesBundle:
    folder = root / "SMD"
    train_paths = sorted((folder / "train").glob("*.txt"), key=_natural_key)
    if not train_paths:
        raise FileNotFoundError(f"No SMD files found below {folder / 'train'}")
    train, test, labels = [], [], []
    for train_path in train_paths:
        test_path = folder / "test" / train_path.name
        label_path = folder / "test_label" / train_path.name
        train.append(_as_float32(np.loadtxt(train_path, delimiter=",")))
        test.append(_as_float32(np.loadtxt(test_path, delimiter=",")))
        labels.append(np.asarray(np.loadtxt(label_path, delimiter=","), dtype=np.int64).reshape(-1))
    return SeriesBundle(
        "SMD", train, test, labels, int(train[0].shape[1]),
        [path.stem for path in train_paths],
    )


def _read_swat_clean(path: Path) -> pd.DataFrame:
    # Prefer the already-cleaned TranAD-style CSVs in this workspace.  Their
    # final column is the binary Normal/Attack label and all other columns are
    # the 51 process variables used by the paper.
    frame = pd.read_csv(path, low_memory=False)
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame


def _load_swat(root: Path) -> SeriesBundle:
    folder = root / "SWaT"
    raw_train = folder / "swat_train.csv"
    raw_test = folder / "swat_test.csv"
    clean_train = folder / "swat_train2.csv"
    clean_test = folder / "swat2.csv"
    if raw_train.exists() and raw_test.exists():
        # These files preserve all 496,800/449,919 points quoted in Table I.
        train_frame = pd.read_csv(raw_train, skiprows=1, low_memory=False)
        test_frame = pd.read_csv(raw_test, skiprows=1, low_memory=False)
        train_frame.columns = [str(c).strip() for c in train_frame.columns]
        test_frame.columns = [str(c).strip() for c in test_frame.columns]
        label_col = next(c for c in test_frame.columns if "attack" in c.lower())
        timestamp_cols = [c for c in train_frame.columns if "time" in c.lower()]
        feature_cols = [c for c in train_frame.columns if c not in timestamp_cols and c != label_col]
        train = _as_float32(train_frame[feature_cols].apply(pd.to_numeric, errors="coerce").values)
        test = _as_float32(test_frame[feature_cols].apply(pd.to_numeric, errors="coerce").values)
        raw_labels = test_frame[label_col]
    elif clean_train.exists() and clean_test.exists():
        train_frame = _read_swat_clean(clean_train)
        test_frame = _read_swat_clean(clean_test)
        train = _as_float32(train_frame.iloc[:, :-1].apply(pd.to_numeric, errors="coerce").values)
        test = _as_float32(test_frame.iloc[:, :-1].apply(pd.to_numeric, errors="coerce").values)
        raw_labels = test_frame.iloc[:, -1]
    else:
        raise FileNotFoundError("No supported SWaT CSV pair was found in {}".format(folder))

    if pd.api.types.is_numeric_dtype(raw_labels):
        labels = (pd.to_numeric(raw_labels, errors="coerce").fillna(0).values > 0).astype(np.int64)
    else:
        labels = raw_labels.astype(str).str.strip().str.lower().str.contains("attack").astype(np.int64).values
    return SeriesBundle(
        "SWaT", [train], [test], [labels.reshape(-1)], int(train.shape[1]), ["SWaT"]
    )


def load_dataset(root: Union[str, Path], name: str) -> SeriesBundle:
    root = Path(root)
    normalized = name.upper()
    if normalized in {"SMAP", "MSL"}:
        return _load_nasa(root, normalized)
    if normalized == "SMD":
        return _load_smd(root)
    if normalized == "SWAT":
        return _load_swat(root)
    raise KeyError(f"Unsupported dataset {name!r}; choose SMAP, MSL, SMD, or SWaT")


def split_and_standardize(
    bundle: SeriesBundle, train_fraction: float = 0.7, per_segment: bool = False
):
    """Return train/validation/test segments scaled using training data only."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    train_segments, val_segments = [], []
    for segment in bundle.train:
        cut = max(1, min(len(segment) - 1, int(len(segment) * train_fraction)))
        train_segments.append(segment[:cut])
        val_segments.append(segment[cut:])

    total = sum(len(x) for x in train_segments)
    if total == 0:
        raise ValueError(f"Dataset {bundle.name} has no training points")
    if per_segment:
        if not (len(train_segments) == len(val_segments) == len(bundle.test)):
            raise ValueError("per-segment scaling requires paired train/validation/test segments")
        scaled_train, scaled_val, scaled_test = [], [], []
        means, scales = [], []
        for train, validation, test in zip(train_segments, val_segments, bundle.test):
            mean = np.asarray(train, dtype=np.float64).mean(axis=0)
            std = np.asarray(train, dtype=np.float64).std(axis=0)
            std[std < 1e-7] = 1.0
            transform = lambda values: np.nan_to_num(
                ((values - mean) / std).astype(np.float32), copy=False
            )
            scaled_train.append(transform(train))
            scaled_val.append(transform(validation))
            scaled_test.append(transform(test))
            means.append(mean)
            scales.append(std)
        return scaled_train, scaled_val, scaled_test, bundle.labels, means, scales

    feature_sum = sum(np.asarray(x, dtype=np.float64).sum(axis=0) for x in train_segments)
    mean = feature_sum / total
    squared_deviation_sum = sum(
        np.square(np.asarray(x, dtype=np.float64) - mean).sum(axis=0) for x in train_segments
    )
    variance = squared_deviation_sum / total
    std = np.sqrt(variance)
    # Match sklearn's StandardScaler convention: constant training channels
    # use a scale of one instead of amplifying tiny numerical differences.
    std[std < 1e-7] = 1.0

    def scale(parts: Sequence[np.ndarray]) -> List[np.ndarray]:
        return [np.nan_to_num(((x - mean) / std).astype(np.float32), copy=False) for x in parts]

    return scale(train_segments), scale(val_segments), scale(bundle.test), bundle.labels, mean, std


class WindowDataset(Dataset):
    """Lazy fixed-window view that never crosses independent series boundaries."""

    def __init__(
        self,
        segments: Sequence[np.ndarray],
        window_size: int,
        stride: int = 1,
        max_windows: int | None = None,
    ) -> None:
        self.segments = list(segments)
        self.window_size = int(window_size)
        self.stride = int(stride)
        if self.window_size <= 0 or self.stride <= 0:
            raise ValueError("window_size and stride must be positive")
        self.counts = [max(0, 1 + (len(x) - self.window_size) // self.stride) for x in self.segments]
        self.cumulative = np.cumsum(self.counts, dtype=np.int64)
        total = int(self.cumulative[-1]) if len(self.cumulative) else 0
        if max_windows is not None and 0 < max_windows < total:
            self.selected = np.linspace(0, total - 1, int(max_windows), dtype=np.int64)
        else:
            self.selected = None

    def __len__(self) -> int:
        if self.selected is not None:
            return int(len(self.selected))
        return int(self.cumulative[-1]) if len(self.cumulative) else 0

    def __getitem__(self, index: int) -> torch.Tensor:
        global_index = int(self.selected[index]) if self.selected is not None else int(index)
        segment_index = bisect.bisect_right(self.cumulative, global_index)
        previous = int(self.cumulative[segment_index - 1]) if segment_index else 0
        local_index = global_index - previous
        start = local_index * self.stride
        window = self.segments[segment_index][start : start + self.window_size]
        return torch.from_numpy(np.asarray(window, dtype=np.float32))


class ContaminatedWindowDataset(Dataset):
    """Deterministically inject synthetic anomalies into an exact fraction of windows.

    The wrapped clean dataset is never modified. A higher ratio with the same seed
    selects a superset of the windows selected by a lower ratio.
    """

    ANOMALY_TYPES = ("spike", "level_shift", "scale_change", "noise_burst")

    def __init__(
        self,
        clean_dataset: Dataset,
        contamination_ratio: float,
        seed: int,
        channel_scales: Optional[np.ndarray] = None,
    ) -> None:
        self.clean_dataset = clean_dataset
        self.contamination_ratio = float(contamination_ratio)
        self.seed = int(seed)
        if not 0.0 <= self.contamination_ratio <= 1.0:
            raise ValueError("contamination_ratio must be between zero and one")
        self.total_windows = int(len(clean_dataset))
        self.contaminated_windows = int(
            np.floor(self.total_windows * self.contamination_ratio + 0.5)
        )

        rng = np.random.RandomState(self.seed)
        permutation = rng.permutation(self.total_windows).astype(np.int64)
        selected = permutation[: self.contaminated_windows]
        type_blocks = []
        for offset in range(0, self.total_windows, len(self.ANOMALY_TYPES)):
            block = np.arange(len(self.ANOMALY_TYPES), dtype=np.int64)
            rng.shuffle(block)
            type_blocks.append(block[: min(len(block), self.total_windows - offset)])
        all_type_codes = (
            np.concatenate(type_blocks)
            if type_blocks
            else np.empty(0, dtype=np.int64)
        )
        all_injection_seeds = rng.randint(
            0, np.iinfo(np.int32).max, size=self.total_windows
        ).astype(np.int64)
        type_codes = all_type_codes[: self.contaminated_windows]
        injection_seeds = all_injection_seeds[: self.contaminated_windows]
        self._spec_by_index = {
            int(index): (int(code), int(injection_seed))
            for index, code, injection_seed in zip(selected, type_codes, injection_seeds)
        }

        if channel_scales is None:
            self.channel_scales = None
        else:
            scales = np.asarray(channel_scales, dtype=np.float32).reshape(-1)
            scales = np.where(scales > 1e-6, scales, 1.0).astype(np.float32)
            self.channel_scales = scales

        digest = hashlib.sha256()
        digest.update(np.asarray(selected, dtype=np.int64).tobytes())
        digest.update(np.asarray(type_codes, dtype=np.int64).tobytes())
        digest.update(np.asarray(injection_seeds, dtype=np.int64).tobytes())
        self.selection_sha256 = digest.hexdigest()
        self.type_counts = {
            name: int(np.sum(type_codes == index))
            for index, name in enumerate(self.ANOMALY_TYPES)
        }

    def __len__(self) -> int:
        return self.total_windows

    def __getitem__(self, index: int) -> torch.Tensor:
        clean = self.clean_dataset[index]
        specification = self._spec_by_index.get(int(index))
        if specification is None:
            return clean

        type_code, injection_seed = specification
        rng = np.random.RandomState(injection_seed)
        values = clean.detach().cpu().numpy().astype(np.float32, copy=True)
        length, channels_total = values.shape
        channel_count = int(rng.randint(1, max(2, channels_total // 5 + 1)))
        channels = np.asarray(
            rng.choice(channels_total, size=min(channel_count, channels_total), replace=False),
            dtype=np.int64,
        )
        gamma = float(rng.choice(np.asarray([2.0, 3.0, 4.0], dtype=np.float32)))
        signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=len(channels))
        scales = (
            np.ones(channels_total, dtype=np.float32)
            if self.channel_scales is None
            else self.channel_scales
        )

        if type_code == 0:
            width = int(rng.randint(1, min(4, length) + 1))
        else:
            lower = max(2, length // 8)
            upper = max(lower + 1, length // 2)
            width = int(rng.randint(lower, upper + 1))
        start = int(rng.randint(0, max(1, length - width + 1)))
        end = min(length, start + width)

        if type_code in {0, 1}:
            values[start:end, channels] += (
                gamma * signs * scales[channels]
            )[None, :]
        elif type_code == 2:
            values[start:end, channels] *= gamma
        else:
            noise = rng.normal(
                0.0,
                gamma * scales[channels],
                size=(end - start, len(channels)),
            ).astype(np.float32)
            values[start:end, channels] += noise
        return torch.from_numpy(values)

    def manifest(self):
        return {
            "ratio": self.contamination_ratio,
            "total_windows": self.total_windows,
            "contaminated_windows": self.contaminated_windows,
            "actual_ratio": (
                float(self.contaminated_windows) / float(self.total_windows)
                if self.total_windows
                else 0.0
            ),
            "seed": self.seed,
            "types": list(self.ANOMALY_TYPES),
            "type_counts": self.type_counts,
            "gamma_choices": [2.0, 3.0, 4.0],
            "selection_sha256": self.selection_sha256,
            "higher_ratios_are_nested_supersets": True,
        }


def segment_channel_std(segments: Sequence[np.ndarray]) -> np.ndarray:
    """Channel standard deviation without concatenating all training segments."""
    total = sum(len(segment) for segment in segments)
    if total <= 0:
        raise ValueError("cannot estimate channel scale from empty segments")
    channel_sum = sum(
        np.asarray(segment, dtype=np.float64).sum(axis=0) for segment in segments
    )
    mean = channel_sum / float(total)
    squared = sum(
        np.square(np.asarray(segment, dtype=np.float64) - mean).sum(axis=0)
        for segment in segments
    )
    std = np.sqrt(squared / float(total))
    std[std < 1e-7] = 1.0
    return std.astype(np.float32)


def evaluation_windows(length: int, window_size: int, stride: int) -> List[int]:
    if length < window_size:
        return []
    starts = list(range(0, length - window_size + 1, stride))
    tail = length - window_size
    if not starts or starts[-1] != tail:
        starts.append(tail)
    return starts
