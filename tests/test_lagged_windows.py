import numpy as np
import pytest

from pyshred.processor.utils import (
    generate_lagged_sensor_measurements,
    generate_lagged_sensor_measurements_rom,
)

NUM_TIMESTEPS = 12
NUM_SENSORS = 3
LAGS = 4


def sensor_measurements(num_timesteps=NUM_TIMESTEPS, num_sensors=NUM_SENSORS, offset=1):
    """Every timestep holds distinct non-zero values, so sequences can be traced back to
    their source rows and never collide with the zero padding."""
    total = num_timesteps * num_sensors
    return np.arange(offset, offset + total, dtype=float).reshape(num_timesteps, num_sensors)


def test_shape():
    lagged = generate_lagged_sensor_measurements(sensor_measurements(), LAGS)
    assert lagged.shape == (NUM_TIMESTEPS, LAGS, NUM_SENSORS)


def test_sequence_ends_at_target_timestep():
    """The defining property: the sequence reconstructing u(t) must contain s(t)."""
    s = sensor_measurements()
    lagged = generate_lagged_sensor_measurements(s, LAGS)
    for t in range(NUM_TIMESTEPS):
        np.testing.assert_array_equal(lagged[t, -1], s[t])


def test_sequence_spans_lags_timesteps_back_from_target():
    s = sensor_measurements()
    lagged = generate_lagged_sensor_measurements(s, LAGS)
    for t in range(LAGS - 1, NUM_TIMESTEPS):
        np.testing.assert_array_equal(lagged[t], s[t - LAGS + 1:t + 1])
        np.testing.assert_array_equal(lagged[t, 0], s[t - LAGS + 1])


def test_warmup_sequences_are_zero_padded_at_the_front():
    s = sensor_measurements()
    lagged = generate_lagged_sensor_measurements(s, LAGS)
    for t in range(LAGS - 1):
        num_zero_rows = LAGS - 1 - t
        np.testing.assert_array_equal(lagged[t, :num_zero_rows], 0.0)
        assert (lagged[t, num_zero_rows:] != 0.0).all()


def test_single_lag_is_the_current_timestep_only():
    s = sensor_measurements()
    lagged = generate_lagged_sensor_measurements(s, 1)
    assert lagged.shape == (NUM_TIMESTEPS, 1, NUM_SENSORS)
    for t in range(NUM_TIMESTEPS):
        np.testing.assert_array_equal(lagged[t, 0], s[t])


def test_rom_sequences_never_span_two_trajectories():
    first = sensor_measurements(offset=1)
    second = sensor_measurements(offset=1000)
    dataset = np.stack([first, second])

    lagged = generate_lagged_sensor_measurements_rom(dataset, LAGS)
    assert lagged.shape == (2 * NUM_TIMESTEPS, LAGS, NUM_SENSORS)

    expected = np.concatenate([
        generate_lagged_sensor_measurements(first, LAGS),
        generate_lagged_sensor_measurements(second, LAGS),
    ], axis=0)
    np.testing.assert_array_equal(lagged, expected)

    # the second trajectory's warm-up sequences pad with zeros rather than bleeding backwards
    assert not np.isin(lagged[NUM_TIMESTEPS:], first).any()


@pytest.mark.parametrize("bad_lags", [0, -1])
def test_rejects_non_positive_lags(bad_lags):
    with pytest.raises(ValueError, match="positive integer"):
        generate_lagged_sensor_measurements(sensor_measurements(), bad_lags)
