import numpy as np

from chart_maker.quantize import quantize

RES = 192


def test_straight_sixteenths_stay_straight():
    ideal = np.arange(16) * 48
    noisy = ideal + np.random.default_rng(0).normal(0, 4, ideal.size)
    assert list(quantize(noisy, RES)) == list(ideal)


def test_triplets_snap_to_triplet_grid():
    ideal = np.array([0, 64, 128, 192, 256, 320, 384])
    assert list(quantize(ideal + 3, RES)) == list(ideal)


def test_thirty_seconds_only_when_slow():
    pos = np.array([0, 24, 48, 72, 96, 120, 144, 168], dtype=float)
    slow = quantize(pos, RES, beat_seconds=np.full(pos.size, 0.75))
    fast = quantize(pos, RES, beat_seconds=np.full(pos.size, 0.3))
    assert list(slow) == list(pos.astype(int))
    assert len(set(fast)) < len(pos)  # merged onto a coarser grid
