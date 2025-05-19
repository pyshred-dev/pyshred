# # normalize latent outputs between [-1,1]
# from sklearn.preprocessing import MinMaxScaler

# # assume `Z` is your (N × k) latent array, either np.ndarray or torch.Tensor converted to np
# scaler = MinMaxScaler(feature_range=(-1, 1))
# Z_norm = scaler.fit_transform(Z)

# Z_val_norm = scaler.transform(Z_val)
# Z_unscaled = scaler.inverse_transform(Z_norm)

# # Torch
# mins, _ = Z.min(dim=0)
# maxs, _ = Z.max(dim=0)
# Z0_1 = (Z - mins) / (maxs - mins)    # now in [0,1]
# Zm1_1 = 2 * Z0_1 - 1                 # now in [-1,1]

#  fit Sindy

#  simulate sindy with an initial seed (latent space at a single timestep)


# SHREDEngine(DataManager, SHRED)
# encode_sensors(sensor_measurements) -> latent_space
# encode_forecast(t, initialization = None) -> latent_space # can get initialization from self.encoder_sensors(...)
# decode(latent_space)

class SHREDEngine():

    def __init__(DataManager, SHRED):
        pass

    # padding always true
    def sensor_to_latent(sensor_measurements):
        pass

    def forecast_latent(init, t):
        # if sindy, use simulate
        # else, roll latent forward
        pass

    def decode_latent(latent):
        pass