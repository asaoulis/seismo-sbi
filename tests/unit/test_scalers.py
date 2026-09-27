import numpy as np
import pytest

from seismo_sbi.sbi.scalers import ZeroOneScaler, SymmetricLogScaler


def test_zero_one_scaler_roundtrip():
    scaler = ZeroOneScaler(bounds=[-5.0, 5.0])
    x = np.array([-3.0, 0.0, 2.5, 4.9])
    assert np.allclose(scaler.inverse_transform(scaler.transform(x)), x)


def test_zero_one_scaler_lower_boundary():
    scaler = ZeroOneScaler(bounds=[-5.0, 5.0])
    assert np.isclose(scaler.transform(np.array([-5.0])), 0.0)


def test_zero_one_scaler_upper_boundary():
    scaler = ZeroOneScaler(bounds=[-5.0, 5.0])
    assert np.isclose(scaler.transform(np.array([5.0])), 1.0)


def test_zero_one_scaler_midpoint():
    scaler = ZeroOneScaler(bounds=[0.0, 10.0])
    assert np.isclose(scaler.transform(np.array([5.0])), 0.5)


def test_zero_one_scaler_array_roundtrip():
    bounds = np.array([-1e13, 1e13])
    scaler = ZeroOneScaler(bounds=bounds)
    x = np.array([-5e12, 0.0, 3e12, 9e12])
    assert np.allclose(scaler.inverse_transform(scaler.transform(x)), x, rtol=1e-10)


def test_symmetric_log_scaler_positive_roundtrip():
    scaler = SymmetricLogScaler(lower_bound=1e-3, upper_bound=1e3)
    x = np.array([1e-2, 0.1, 1.0, 10.0, 100.0])
    recovered = scaler.inverse_transform(scaler.transform(x))
    assert np.allclose(recovered, x, rtol=1e-5)


def test_symmetric_log_scaler_negative_roundtrip():
    scaler = SymmetricLogScaler(lower_bound=1e-3, upper_bound=1e3)
    x = np.array([-100.0, -10.0, -1.0])
    recovered = scaler.inverse_transform(scaler.transform(x))
    assert np.allclose(recovered, x, rtol=1e-5)


def test_symmetric_log_scaler_centred_at_half():
    """SymmetricLogScaler maps ±lower_bound to exactly 0.5 (boundary of log region)."""
    scaler = SymmetricLogScaler(lower_bound=1.0, upper_bound=100.0)
    positive = scaler.transform(np.array([1.0]))
    negative = scaler.transform(np.array([-1.0]))
    # lower_bound maps to exactly 0.5 (log argument is 0 → linear contribution is 0)
    assert np.isclose(positive, 0.5, atol=1e-10)
    assert np.isclose(negative, 0.5, atol=1e-10)
    # Values above lower_bound should be above 0.5
    above = scaler.transform(np.array([10.0]))
    below = scaler.transform(np.array([-10.0]))
    assert above > 0.5
    assert below < 0.5
