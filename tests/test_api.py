"""The public API reproduces numbers stated in the paper."""
import numpy as np
import pytest

import gammacert as gc

# Report-shaped tiers of Section 7: published sampling rates, assumed shares and recalls.
PI = [1, 0.25, 0.05, 0.002, 0.0002]
SHARES = [0.40, 0.10, 0.20, 0.21, 0.09]
RECALL = [0.94, 0.94, 0.58, 0.58, 0.44]


def test_zero_detection_certificate_grows_with_concentration():
    assert gc.certificate(PI, RECALL, SHARES, gamma=1) == pytest.approx(2.6, abs=0.06)
    assert gc.certificate(PI, RECALL, SHARES, gamma=2) == pytest.approx(244.1, abs=0.06)


def test_equal_coverage_is_gamma_free():
    pi = gc.equal_coverage_design(RECALL, SHARES, budget=np.dot(SHARES, PI))
    curve = gc.certificate_curve(pi, RECALL, SHARES, gammas=[1, 2, 10, 100])
    assert np.allclose(curve, 8.3, atol=0.06)


def test_campaign_escape_table1():
    assert gc.campaign_escape(24, PI, RECALL, SHARES, gamma=2) == pytest.approx(0.74, abs=0.005)


def test_hidden_floor():
    assert gc.hidden_floor(30, 0.44) == pytest.approx(0.22, abs=0.005)
    assert gc.floor_certificate(gc.hidden_floor(30, 0.44), 0.44) == pytest.approx(30.0)


def test_poisson_constant_is_smaller_than_chernoff():
    garwood = gc.certificate(0.5, 0.8, detections=1)
    chernoff = gc.certificate(0.5, 0.8, detections=1, method="chernoff")
    assert garwood < chernoff and garwood == pytest.approx(4.7439 / 0.4, rel=1e-3)


def test_zero_coverage_stratum_is_unbounded():
    assert np.isinf(gc.certificate([0.25, 0.25], [0.9, 0.0], [0.5, 0.5], gamma=2))


def test_recall_bound_and_monitor_mixture():
    assert gc.recall_lower_bounds(153, 163, simultaneous=False)[0] == pytest.approx(0.890, abs=6e-4)
    value, p = gc.randomized_monitor([[0.9, 0.3], [0.2, 0.8]])
    assert value == pytest.approx(0.55) and p.sum() == pytest.approx(1.0)


def test_single_action_and_selection():
    assert not gc.single_action_certified([1, 0.25], [0.94, 0.94])
    assert gc.selection_coverage([0.1, 0.9, 0.2, 0.8], [3, 0, 2, 1], 0.5) == pytest.approx(0.15)
