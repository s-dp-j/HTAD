from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


@dataclass
class DetectionMetrics:
    auc: float
    au_pr: float
    precision: float
    recall: float
    f1: float
    threshold: float
    threshold_mode: str
    point_adjusted: bool
    points: int
    anomalies: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def anomaly_scores(values: np.ndarray, reconstruction: np.ndarray) -> np.ndarray:
    return np.sqrt(np.square(values - reconstruction).sum(axis=-1)).astype(np.float64)


def level_deviation_scores(values: np.ndarray, top_k: int = 0) -> np.ndarray:
    """Score distance from the normal training centre after standardisation."""
    error = np.abs(np.asarray(values, dtype=np.float64))
    return _aggregate_channels(error, top_k)


def calibrated_reconstruction_scores(
    values: np.ndarray,
    reconstruction: np.ndarray,
    calibration_values: np.ndarray,
    calibration_reconstruction: np.ndarray,
    top_k: int = 0,
) -> np.ndarray:
    """Robustly calibrate each channel using held-out normal errors only."""
    calibration_error = np.abs(
        np.asarray(calibration_values, dtype=np.float64)
        - np.asarray(calibration_reconstruction, dtype=np.float64)
    )
    error = np.abs(np.asarray(values, dtype=np.float64) - np.asarray(reconstruction, dtype=np.float64))
    centre = np.median(calibration_error, axis=0)
    scale = 1.4826 * np.median(np.abs(calibration_error - centre), axis=0)
    # A standard-deviation fallback prevents discrete/constant channels from
    # being amplified when their median absolute deviation is exactly zero.
    fallback = np.std(calibration_error, axis=0)
    scale = np.where(scale > 1e-6, scale, fallback)
    scale = np.maximum(scale, 1e-6)
    calibrated = np.maximum((error - centre) / scale, 0.0)
    return _aggregate_channels(calibrated, top_k)


def _aggregate_channels(error: np.ndarray, top_k: int = 0) -> np.ndarray:
    if error.ndim != 2:
        raise ValueError("channel errors must have shape [time, channels]")
    if top_k <= 0 or top_k >= error.shape[1]:
        return error.mean(axis=1).astype(np.float64)
    selected = np.partition(error, error.shape[1] - top_k, axis=1)[:, -top_k:]
    return selected.mean(axis=1).astype(np.float64)


def point_adjust_predictions(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).astype(bool)
    adjusted = np.asarray(predictions).astype(bool).copy()
    start = None
    for index, value in enumerate(np.r_[labels, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if adjusted[start:index].any():
                adjusted[start:index] = True
            start = None
    return adjusted.astype(np.int64)


def _prf(labels: np.ndarray, predictions: np.ndarray) -> Tuple[float, float, float]:
    labels = labels.astype(bool)
    predictions = predictions.astype(bool)
    true_positive = int(np.logical_and(labels, predictions).sum())
    false_positive = int(np.logical_and(~labels, predictions).sum())
    false_negative = int(np.logical_and(labels, ~predictions).sum())
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return precision, recall, f1


def _events(labels: np.ndarray) -> Iterable[Tuple[int, int]]:
    active = np.asarray(labels).astype(bool)
    changes = np.diff(np.r_[False, active, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return zip(starts, ends)


def best_f1_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    point_adjust: bool = False,
    steps: int = 2000,
) -> Tuple[float, float, float, float]:
    """Find an oracle threshold without repeatedly scanning the full series."""
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    if point_adjust:
        # Point-adjusted recall changes only when a threshold crosses an
        # anomaly-event maximum; false positives change only at normal scores.
        # Their union therefore enumerates every distinct prediction state.
        normal_sorted = np.sort(scores[labels == 0])
        event_maxima, event_lengths = [], []
        for start, end in _events(labels):
            event_maxima.append(float(scores[start:end].max()))
            event_lengths.append(end - start)
        event_maxima = np.asarray(event_maxima, dtype=np.float64)
        event_lengths = np.asarray(event_lengths, dtype=np.int64)
        if not len(event_maxima):
            return float(scores.max()), 0.0, 0.0, 0.0

        candidates = np.unique(np.concatenate([normal_sorted, event_maxima]))
        false_positive = len(normal_sorted) - np.searchsorted(
            normal_sorted, candidates, side="left"
        )
        event_order = np.argsort(event_maxima, kind="mergesort")
        sorted_maxima = event_maxima[event_order]
        sorted_lengths = event_lengths[event_order]
        suffix_lengths = np.r_[np.cumsum(sorted_lengths[::-1])[::-1], 0]
        detected_from = np.searchsorted(sorted_maxima, candidates, side="left")
        true_positive = suffix_lengths[detected_from].astype(np.float64)
        positives = int((labels > 0).sum())
        precision = true_positive / np.maximum(1.0, true_positive + false_positive)
        recall = true_positive / max(1, positives)
        f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
        index = int(np.nanargmax(f1))
        return (
            float(np.nextafter(candidates[index], -np.inf)),
            float(precision[index]),
            float(recall[index]),
            float(f1[index]),
        )

    # The unadjusted protocol can be evaluated exactly in O(n log n).  This
    # avoids missing the true best threshold when scores are concentrated in
    # a narrow interval (the previous evenly-spaced 2,000 point grid could).
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2.0 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    index = int(np.nanargmax(f1))
    return (
        # sklearn's PR curve uses ``score >= threshold`` while the prediction
        # path below uses ``score > threshold``.  Store the adjacent lower
        # float so the reported threshold reproduces the exact same set.
        float(np.nextafter(thresholds[index], -np.inf)),
        float(precision[index]),
        float(recall[index]),
        float(f1[index]),
    )


def calculate_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold_mode: str = "oracle",
    calibration_scores: np.ndarray = None,
    quantile: float = 0.995,
    point_adjust: bool = False,
) -> DetectionMetrics:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have equal length")
    if len(np.unique(labels)) < 2:
        auc = float("nan")
        au_pr = float("nan")
    else:
        auc = float(roc_auc_score(labels, scores))
        au_pr = float(average_precision_score(labels, scores))

    if threshold_mode == "oracle":
        threshold, precision, recall, f1 = best_f1_threshold(labels, scores, point_adjust=point_adjust)
    elif threshold_mode == "train-quantile":
        if calibration_scores is None or not len(calibration_scores):
            raise ValueError("train-quantile mode requires calibration_scores")
        threshold = float(np.quantile(calibration_scores, quantile))
        prediction = (scores > threshold).astype(np.int64)
        if point_adjust:
            prediction = point_adjust_predictions(labels, prediction)
        precision, recall, f1 = _prf(labels, prediction)
    else:
        raise KeyError("threshold_mode must be 'oracle' or 'train-quantile'")

    return DetectionMetrics(
        auc=auc,
        au_pr=au_pr,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        threshold=float(threshold),
        threshold_mode=threshold_mode,
        point_adjusted=point_adjust,
        points=int(len(labels)),
        anomalies=int(labels.sum()),
    )
