"""
Unit tests for ActivityRecognizer.predict_activity().

Ensures the 2-tuple / 3-tuple contract is stable, which is what the rest
of the codebase relies on.
"""

import numpy as np
import pytest

from models.activity_recognizer import ActivityRecognizer


def _make_recognizer():
    """Small model so tests run quickly."""
    return ActivityRecognizer(
        sequence_length=30,
        num_keypoints=33,
        num_classes=5,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
    )


# ---------------------------------------------------------------------
#  Short sequence: fewer than sequence_length frames
# ---------------------------------------------------------------------
def test_predict_short_sequence_returns_two_tuple():
    rec = _make_recognizer()
    short = np.zeros((5, 99), dtype=np.float32)  # 5 < sequence_length
    out = rec.predict_activity(short)
    assert isinstance(out, tuple)
    assert len(out) == 2
    activity, confidence = out
    assert activity is None
    assert confidence == 0.0


def test_predict_short_sequence_with_probs_returns_three_tuple():
    rec = _make_recognizer()
    short = np.zeros((5, 99), dtype=np.float32)
    out = rec.predict_activity(short, return_probabilities=True)
    assert isinstance(out, tuple)
    assert len(out) == 3
    activity, confidence, probs = out
    assert activity is None
    assert confidence == 0.0
    assert probs is None


# ---------------------------------------------------------------------
#  Empty / None input
# ---------------------------------------------------------------------
def test_predict_none_sequence_returns_two_tuple():
    rec = _make_recognizer()
    out = rec.predict_activity(None)
    assert out == (None, 0.0)


def test_predict_none_sequence_with_probs_returns_three_tuple():
    rec = _make_recognizer()
    out = rec.predict_activity(None, return_probabilities=True)
    assert out == (None, 0.0, None)


# ---------------------------------------------------------------------
#  Full-length sequence
# ---------------------------------------------------------------------
def test_predict_full_sequence_returns_label():
    rec = _make_recognizer()
    seq = np.random.rand(30, 99).astype(np.float32)
    activity, confidence = rec.predict_activity(seq)
    assert activity in rec.activity_labels
    assert 0.0 <= confidence <= 1.0


def test_predict_full_sequence_with_probs_returns_distribution():
    rec = _make_recognizer()
    seq = np.random.rand(30, 99).astype(np.float32)
    activity, confidence, probs = rec.predict_activity(
        seq, return_probabilities=True
    )
    assert activity in rec.activity_labels
    assert 0.0 <= confidence <= 1.0
    assert probs is not None
    assert probs.shape == (rec.num_classes,)
    assert abs(float(probs.sum()) - 1.0) < 1e-4


# ---------------------------------------------------------------------
#  Longer-than-needed sequence: only the last N frames should be used
# ---------------------------------------------------------------------
def test_predict_long_sequence_uses_tail():
    rec = _make_recognizer()
    seq = np.random.rand(60, 99).astype(np.float32)
    activity, confidence = rec.predict_activity(seq)
    assert activity in rec.activity_labels
    assert 0.0 <= confidence <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
