import numpy as np
import pytest
import torch

from pyshred import ParametricDataManager, ParametricSHREDEngine, SHRED

NUM_TRAJECTORIES = 10
NUM_TIMESTEPS = 15
NUM_SPACE = 8
NUM_PARAMS = 2
SENSORS = [(1,), (5,)]
LAGS = 4


def state_data(seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(NUM_TRAJECTORIES, NUM_TIMESTEPS, NUM_SPACE))


def time_varying_params():
    """Every (trajectory, timestep) holds distinct values, so input sequences can be traced
    back to the timesteps they came from."""
    total = NUM_TRAJECTORIES * NUM_TIMESTEPS * NUM_PARAMS
    return np.arange(1, total + 1, dtype=float).reshape(NUM_TRAJECTORIES, NUM_TIMESTEPS, NUM_PARAMS)


def constant_params(seed=1):
    rng = np.random.default_rng(seed)
    return rng.uniform(1.0, 5.0, size=(NUM_TRAJECTORIES, NUM_PARAMS))


def prepared_manager(params=None):
    manager = ParametricDataManager(lags=LAGS, train_size=0.8, val_size=0.1, test_size=0.1)
    manager.add_data(data=state_data(), id="U", stationary=SENSORS, compress=False, params=params)
    return manager, manager.prepare()


def test_params_are_appended_to_the_sensor_inputs():
    _, (train, val, test) = prepared_manager(params=time_varying_params())
    _, (train_no_params, val_no_params, test_no_params) = prepared_manager()

    for with_params, without_params in [(train, train_no_params), (val, val_no_params), (test, test_no_params)]:
        assert with_params.X.shape == without_params.X.shape[:-1] + (len(SENSORS) + NUM_PARAMS,)
        # sensors come first and are unchanged, and params are inputs only, never targets
        torch.testing.assert_close(with_params.X[..., :len(SENSORS)], without_params.X)
        torch.testing.assert_close(with_params.Y, without_params.Y)


def test_param_sequences_end_at_the_target_timestep():
    params = time_varying_params()
    manager, (train, _, _) = prepared_manager(params=params)
    X = train.X.cpu().numpy()
    train_params = params[manager.train_indices]
    scaled_params = manager.params_scaler.transform(train_params.reshape(-1, NUM_PARAMS)).reshape(train_params.shape)

    for trajectory in range(len(manager.train_indices)):
        for t in range(LAGS - 1, NUM_TIMESTEPS):
            sequence = X[trajectory * NUM_TIMESTEPS + t, :, len(SENSORS):]
            np.testing.assert_allclose(sequence, scaled_params[trajectory, t - LAGS + 1:t + 1], rtol=1e-6, atol=1e-7)


def test_params_scaler_is_fit_on_training_trajectories_only():
    params = time_varying_params()
    manager, _ = prepared_manager(params=params)
    train_params = params[manager.train_indices]
    np.testing.assert_array_equal(manager.params_scaler.data_min_, train_params.reshape(-1, NUM_PARAMS).min(axis=0))
    np.testing.assert_array_equal(manager.params_scaler.data_max_, train_params.reshape(-1, NUM_PARAMS).max(axis=0))


def test_params_are_split_like_the_sensor_measurements():
    params = time_varying_params()
    manager, _ = prepared_manager(params=params)
    np.testing.assert_array_equal(manager.train_params, params[manager.train_indices])
    np.testing.assert_array_equal(manager.val_params, params[manager.val_indices])
    np.testing.assert_array_equal(manager.test_params, params[manager.test_indices])


@pytest.mark.parametrize("as_tensor", [False, True])
def test_constant_params_can_leave_out_the_time_axis(as_tensor):
    params = constant_params()
    repeated = np.repeat(params[:, np.newaxis, :], NUM_TIMESTEPS, axis=1)
    _, datasets = prepared_manager(params=torch.tensor(params) if as_tensor else params)
    _, datasets_repeated = prepared_manager(params=repeated)
    for dataset, dataset_repeated in zip(datasets, datasets_repeated):
        torch.testing.assert_close(dataset.X, dataset_repeated.X)


def test_params_can_be_provided_with_any_one_dataset():
    manager = ParametricDataManager(lags=LAGS)
    manager.add_data(data=state_data(0), id="U", stationary=SENSORS, compress=False)
    manager.add_data(data=state_data(1), id="V", compress=False, params=constant_params())
    train, _, _ = manager.prepare()
    assert train.X.shape[-1] == len(SENSORS) + NUM_PARAMS


def test_rejects_params_on_a_second_dataset():
    manager = ParametricDataManager(lags=LAGS)
    manager.add_data(data=state_data(0), id="U", stationary=SENSORS, compress=False, params=constant_params())
    with pytest.raises(ValueError, match="already provided with dataset 'U'"):
        manager.add_data(data=state_data(1), id="V", compress=False, params=constant_params())


@pytest.mark.parametrize("bad_shape", [
    (NUM_TRAJECTORIES + 1, NUM_TIMESTEPS, NUM_PARAMS),
    (NUM_TRAJECTORIES, NUM_TIMESTEPS + 1, NUM_PARAMS),
    (NUM_TRAJECTORIES + 1, NUM_PARAMS),
    (NUM_TRAJECTORIES,),
])
def test_rejects_params_that_do_not_line_up_with_the_data(bad_shape):
    manager = ParametricDataManager(lags=LAGS)
    with pytest.raises(ValueError, match="`params` must have shape"):
        manager.add_data(data=state_data(), id="U", stationary=SENSORS, compress=False, params=np.ones(bad_shape))


@pytest.fixture(scope="module")
def fitted():
    torch.manual_seed(0)
    manager, (train, val, test) = prepared_manager(params=time_varying_params())
    shred = SHRED()
    shred.fit(train, val, num_epochs=1, verbose=False)
    return manager, shred, test


def test_engine_matches_the_prepared_test_inputs(fitted):
    manager, shred, test = fitted
    engine = ParametricSHREDEngine(manager, shred)
    latents = engine.sensor_to_latent(manager.test_sensor_measurements, params=manager.test_params)
    with torch.no_grad():
        expected = shred._seq_model_outputs(test.X.to(next(shred.parameters()).device)).cpu().numpy()
    np.testing.assert_allclose(latents, expected, rtol=1e-5, atol=1e-6)


def test_engine_accepts_a_single_trajectory(fitted):
    manager, shred, _ = fitted
    engine = ParametricSHREDEngine(manager, shred)
    latents = engine.sensor_to_latent(manager.test_sensor_measurements, params=manager.test_params)
    single = engine.sensor_to_latent(manager.test_sensor_measurements[0], params=manager.test_params[0])
    np.testing.assert_allclose(single, latents[:NUM_TIMESTEPS], rtol=1e-5, atol=1e-6)


def test_engine_evaluate_passes_params_through(fitted):
    manager, shred, _ = fitted
    engine = ParametricSHREDEngine(manager, shred)
    test_Y = {"U": state_data()[manager.test_indices].reshape(-1, NUM_SPACE)}
    errors = engine.evaluate(manager.test_sensor_measurements, test_Y, params=manager.test_params)
    assert list(errors.index) == ["U"]


def test_engine_requires_params_when_the_manager_has_them(fitted):
    manager, shred, _ = fitted
    engine = ParametricSHREDEngine(manager, shred)
    with pytest.raises(ValueError, match="`params` must be passed"):
        engine.sensor_to_latent(manager.test_sensor_measurements)


def test_engine_rejects_params_when_the_manager_has_none(fitted):
    _, shred, _ = fitted
    manager, _ = prepared_manager()
    engine = ParametricSHREDEngine(manager, shred)
    with pytest.raises(ValueError, match="not given `params`"):
        engine.sensor_to_latent(manager.test_sensor_measurements, params=time_varying_params()[manager.test_indices])
