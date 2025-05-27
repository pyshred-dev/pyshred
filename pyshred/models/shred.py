import torch
from torch.utils.data import DataLoader
from ..models.sindy import sindy_library_torch, e_sindy_library_torch
from ..models.sequence_models.abstract_sequence import AbstractSequence
from ..models.decoder_models.abstract_decoder import AbstractDecoder
from ..latent_forecaster_models.abstract_latent_forecaster import AbstractLatentForecaster
from ..models.decoder_models.sdn_model import SDN
from ..models.decoder_models.unet_model import UNET
from ..models.sequence_models.gru_model import GRU
from ..models.sequence_models.lstm_model import LSTM
from ..models.sequence_models.transformer_model import TRANSFORMER
import warnings
from ..objects.dataset import TimeSeriesDataset
from ..latent_forecaster_models.sindy import SINDy_Forecaster
from ..latent_forecaster_models.lstm import LSTM_Forecaster

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SINDy(torch.nn.Module):
    def __init__(self, latent_dim, library_dim, poly_order, include_sine):
        super().__init__()
        self.latent_dim = latent_dim
        self.poly_order = poly_order
        self.include_sine = include_sine
        self.library_dim = library_dim
        self.coefficients = torch.ones(library_dim, latent_dim, requires_grad=True)
        torch.nn.init.normal_(self.coefficients, mean=0.0, std=0.001)
        self.coefficient_mask = torch.ones(library_dim, latent_dim, requires_grad=False).to(device)
        self.coefficients = torch.nn.Parameter(self.coefficients)

    def forward(self, h, dt):
        library_Theta = sindy_library_torch(h, self.latent_dim, self.poly_order, self.include_sine)
        h = h + library_Theta @ (self.coefficients * self.coefficient_mask) * dt
        return h
    
    def thresholding(self, threshold):
        self.coefficient_mask = torch.abs(self.coefficients) > threshold
        self.coefficients.data = self.coefficient_mask * self.coefficients.data
        
    def add_noise(self, noise=0.1):
        self.coefficients.data += torch.randn_like(self.coefficients.data) * noise
        self.coefficient_mask = torch.ones(self.library_dim, self.latent_dim, requires_grad=False).to(device)
        
    def recenter(self):
        self.coefficients.data = torch.randn_like(self.coefficients.data) * 0.0
        self.coefficient_mask = torch.ones(self.library_dim, self.latent_dim, requires_grad=False).to(device)   

class E_SINDy(torch.nn.Module):
    def __init__(self, num_replicates, latent_dim, library_dim, poly_order, include_sine):
        super().__init__()
        self.num_replicates = num_replicates
        self.latent_dim = latent_dim
        self.poly_order = poly_order
        self.include_sine = include_sine
        self.library_dim = library_dim
        self.coefficients = torch.ones(num_replicates, library_dim, latent_dim, requires_grad=True)
        torch.nn.init.normal_(self.coefficients, mean=0.0, std=0.001)
        self.coefficient_mask = torch.ones(num_replicates, library_dim, latent_dim, requires_grad=False).to(device)
        self.coefficients = torch.nn.Parameter(self.coefficients)

    def forward(self, h_replicates, dt):
        num_data, num_replicates, latent_dim = h_replicates.shape
        h_replicates = h_replicates.reshape(num_data * num_replicates, latent_dim)
        library_Thetas = e_sindy_library_torch(h_replicates, self.latent_dim, self.poly_order, self.include_sine)
        library_Thetas = library_Thetas.reshape(num_data, num_replicates, self.library_dim)
        h_replicates = h_replicates.reshape(num_data, num_replicates, latent_dim)
        h_replicates = h_replicates + torch.einsum('ijk,jkl->ijl', library_Thetas, (self.coefficients * self.coefficient_mask)) * dt
        return h_replicates
    
    def thresholding(self, threshold, base_threshold=0):
        threshold_tensor = torch.full_like(self.coefficients, threshold)
        for i in range(self.num_replicates):
            threshold_tensor[i] = threshold_tensor[i] * 10**(0.2 * i - 1) + base_threshold
        self.coefficient_mask = torch.abs(self.coefficients) > threshold_tensor
        self.coefficients.data = self.coefficient_mask * self.coefficients.data
        
    def add_noise(self, noise=0.1):
        self.coefficients.data += torch.randn_like(self.coefficients.data) * noise
        self.coefficient_mask = torch.ones(self.num_replicates, self.library_dim, self.latent_dim, requires_grad=False).to(device)   
        
    def recenter(self):
        self.coefficients.data = torch.randn_like(self.coefficients.data) * 0.0
        self.coefficient_mask = torch.ones(self.num_replicates, self.library_dim, self.latent_dim, requires_grad=False).to(device)  


SEQUENCE_MODELS = {
    "LSTM": LSTM,
    "TRANSFORMER": TRANSFORMER,
    "GRU": GRU,
}

DECODER_MODELS = {
    "SDN": SDN,
    "UNET": UNET,
}

LATENT_FORECASTER_MODELS = {
    "SINDY_FORECASTER": SINDy_Forecaster,
    "LSTM_FORECASTER": LSTM_Forecaster
}

NUM_REPLICATES = 10

class SHRED(torch.nn.Module):
    def __init__(self, sequence_model = None, decoder_model = None, latent_forecaster = None, layer_norm = False):
        super().__init__()
        if sequence_model is None:
            if latent_forecaster is not None:
                self.sequence = GRU()
            else:
                self.sequence = LSTM()
        elif isinstance(sequence_model, AbstractSequence):
            self.sequence = sequence_model
        elif isinstance(sequence_model, str):
            sequence_model = sequence_model.upper()
            if sequence_model not in SEQUENCE_MODELS:
                raise ValueError(f"Invalid sequence model: {sequence_model}. Choose from: {list(SEQUENCE_MODELS.keys())}")
            self.sequence = SEQUENCE_MODELS[sequence_model]()
        else:
            raise ValueError("Invalid type for 'sequence_model'. Must be str or an AbstractSequence instance or None.")

        if decoder_model is None:
            self.decoder = SDN()
        elif isinstance(decoder_model, AbstractDecoder):
            if latent_forecaster is not None and not isinstance(decoder_model, SDN):
                warnings.warn(
                    "`latent_forecaster` is not None, but `decoder_model` is not an instance of `SDN`. "
                    "The decoder is being overridden with `SDN()` for compatibility with the `latent_forecaster`.",
                    UserWarning
                )
                self.decoder = SDN()
            else:
                self.decoder = decoder_model
        elif isinstance(decoder_model, str):
            decoder_model = decoder_model.upper()
            if decoder_model not in DECODER_MODELS:
                raise ValueError(f"Invalid decoder model: {decoder_model}. Choose from: {list(DECODER_MODELS.keys())}")
            if latent_forecaster is not None and decoder_model!="SDN":
                warnings.warn(
                    "`latent_forecaster` is not None, but `decoder_model` is not set to \"SDN\". "
                    "The decoder is being overridden with `SDN()` for compatibility with the `latent_forecaster`.",
                    UserWarning
                )
                self.decoder = SDN()
            else:
                self.decoder = DECODER_MODELS[decoder_model]()
        else:
            raise ValueError("Invalid type for 'decoder'. Must be str or an AbstractDecoder instance or None.")
        self.num_replicates = NUM_REPLICATES
        self.use_layer_norm = layer_norm
        self.layer_norm_gru = torch.nn.LayerNorm(self.sequence.hidden_size)
        if latent_forecaster is not None:
            if isinstance(latent_forecaster, AbstractLatentForecaster):
                self.latent_forecaster = latent_forecaster
            elif isinstance(latent_forecaster, str):
                latent_forecaster = latent_forecaster.upper()
                if latent_forecaster not in LATENT_FORECASTER_MODELS:
                    raise ValueError(f"Invalid sequence model: {latent_forecaster}. Choose from: {list(LATENT_FORECASTER_MODELS.keys())}")
                self.latent_forecaster = LATENT_FORECASTER_MODELS[latent_forecaster]()
            else:
                raise TypeError("Invalid type for 'latent_forecaster'. Must be str or an AbstractLatentForecaster instance or None.")
            self.latent_forecaster.initialize(latent_dim = self.sequence.hidden_size)

    def forward(self, x, sindy=False):
        if sindy == True:
            h_out = self.sequence(x)
            if self.use_layer_norm:
                h_out = self.layer_norm_gru(h_out)
            output = self.decoder(h_out)
            with torch.autograd.set_detect_anomaly(True):
                if sindy:
                    h_t = h_out[:-1, :]
                    ht_replicates = h_t.unsqueeze(1).repeat(1, self.num_replicates, 1)
                    for _ in range(10):
                        ht_replicates = self.e_sindy(ht_replicates, dt=self.dt)
                    h_out_replicates = h_out[1:, :].unsqueeze(1).repeat(1, self.num_replicates, 1)
                    output = output, h_out_replicates, ht_replicates
        else:
            h_out = self.sequence(x)
            output = self.decoder(h_out)
        return output

    
    def seq_model_outputs(self, x, sindy=False):
        if sindy == True:
            h_out = self.sequence(x)
            if self.use_layer_norm:
                h_out = self.layer_norm_gru(h_out)
            if sindy:
                h_t = h_out[:-1, :]
                ht_replicates = h_t.unsqueeze(1).repeat(1, self.num_replicates, 1)
                for _ in range(10):
                    ht_replicates = self.e_sindy(ht_replicates, dt=self.dt)
                h_out_replicates = h_out[1:, :].unsqueeze(1).repeat(1, self.num_replicates, 1)
                h_outs = h_out_replicates, ht_replicates
        else:
            h_outs = self.sequence(x)
        return h_outs

    def sindys_threshold(self, threshold):
        self.e_sindy.thresholding(threshold)

    def sindys_add_noise(self, noise):
        self.e_sindy.add_noise(noise)


    # sindy_regularization previously set to 1.0
    def fit(self, train_dataset, val_dataset,  batch_size=64, num_epochs=200, lr=1e-3, sindy_regularization=0,
            optimizer="AdamW", verbose=True, threshold=0.5, base_threshold=0.0, patience=20,
            thres_epoch=100, weight_decay=0.01):
        if not isinstance(self.latent_forecaster, SINDy_Forecaster) and sindy_regularization > 0:
            warnings.warn(
                "`latent_forecaster` is not a SINDy_Forecaster; disabling SINDy regularization.",
                UserWarning
            )
            sindy_regularization = 0

        if sindy_regularization > 0:
            sindy = True
            if not isinstance(self.decoder, SDN):
                warnings.warn("WARNING: SINDy regularization > 0: switching decoder to SDN for compatibility.",
                    UserWarning
                )
                self.decoder = SDN()
        else:
            sindy = False

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(self.latent_forecaster, SINDy_Forecaster):
            self.dt = self.latent_forecaster.dt
            self.poly_order = self.latent_forecaster.poly_order
            self.lib_dim = self.latent_forecaster.lib_dim
            self.include_sine = self.latent_forecaster.include_sine
            self.e_sindy = E_SINDy(self.num_replicates, self.sequence.hidden_size, self.lib_dim, self.poly_order,
                                self.include_sine).to(device)
        input_size = train_dataset.X.shape[2] # nsensors + nparams
        output_size = train_dataset.Y.shape[1]
        lags = train_dataset.X.shape[1] # lags

        self.sequence.initialize(input_size=input_size, lags=lags, decoder=self.decoder)
        self.sequence.to(device)
        self.decoder.initialize(input_size=self.sequence.output_size, output_size=output_size)
        self.decoder.to(device)
        self.to(device)
        # shuffle is False (used to be True)
        train_loader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size)
        criterion = torch.nn.MSELoss()
        if optimizer == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        val_error_list = []
        patience_counter = 0
        best_params = self.state_dict()
        best_val_error = float('inf')  # Initialize with a large value
        if sindy == True:
            print("Fitting SindySHRED...")
        else:
            print("Fitting SHRED...")
        for epoch in range(1, num_epochs + 1):
            self.train()
            running_loss = 0.0
            for data in train_loader:
                if sindy:
                    outputs, h_gru, h_sindy = self(data[0], sindy=True)
                    optimizer.zero_grad()
                    loss = criterion(outputs, data[1]) + criterion(h_gru, h_sindy) * sindy_regularization + torch.abs(torch.mean(h_gru)) * 0.1
                else:
                    outputs = self(data[0], sindy=False)
                    optimizer.zero_grad()
                    loss = criterion(outputs, data[1])
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            if verbose:
                print(f"Epoch {epoch}: Average training loss = {running_loss / len(train_loader):.6f}")
            if sindy:
                if epoch % thres_epoch == 0 and epoch != 0:
                    self.e_sindy.thresholding(threshold=threshold, base_threshold=base_threshold)
            self.eval()
            val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
            val_criterion = torch.nn.MSELoss(reduction="sum")
            total_val_loss = 0.0
            total_val_elems = 0
            with torch.no_grad():
                for X_val, Y_val in val_loader:
                    X_val, Y_val = X_val.to(device), Y_val.to(device)
                    preds = self(X_val)
                    # if you ever forward with sindy=True, unpack the reconstruction
                    if isinstance(preds, tuple):
                        preds = preds[0]
                    batch_loss = val_criterion(preds, Y_val)
                    total_val_loss += batch_loss.item()
                    total_val_elems += Y_val.numel()
            # now compute the true mean‐squared‐error
            val_error = total_val_loss / total_val_elems
            val_error_list.append(val_error)
            if verbose:
                print(f"Validation MSE (epoch {epoch}): {val_error:.6f}")
            if val_error < best_val_error:
                best_val_error = val_error
                best_params = self.state_dict()
                patience_counter = 0  # Reset if improvement
            else:
                patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print("Early stopping triggered: patience threshold reached.")
                break
        self.load_state_dict(best_params)
        device = next(self.parameters()).device
        X_train = train_dataset.X.to(device)    # shape (N_train, lags, n_sensors)
        X_val   = val_dataset.X.to(device)      # shape (N_val,   lags, n_sensors)
        X_all = torch.cat([X_train, X_val], dim=0)   # shape (N_train+N_val, lags, n_sensors)
        # run through encoder to get latents
        self.eval()
        with torch.no_grad():
            latents = self.seq_model_outputs(X_all, sindy=False)   # (N_train+N_val, latent_dim)
        # to numpy and hand off to pysindy
        latents_np = latents.cpu().numpy()
        if isinstance(self.latent_forecaster, SINDy_Forecaster):
            self.latent_forecaster.fit(latents_np)
        elif isinstance(self.latent_forecaster, LSTM_Forecaster):
            self.latent_forecaster.fit(latents=latents_np, num_epochs=num_epochs, batch_size=batch_size, lr=lr)
        return torch.tensor(val_error_list).detach().cpu().numpy()


    def evaluate(self, dataset: TimeSeriesDataset, batch_size: int=64):
        """
        Compute mean squared error on a held‐out test dataset.

        Parameters
        ----------
        test_dataset : torch.utils.data.Dataset
            Should return (X, Y) pairs just like your train/val datasets.
        batch_size : int, optional
            How many samples per batch. Defaults to 64.

        Returns
        -------
        float
            The MSE over all elements in the test set.
        """
        self.eval()
        loader = DataLoader(dataset, shuffle=False, batch_size=batch_size)
        criterion = torch.nn.MSELoss(reduction="sum")
        device = next(self.parameters()).device
        total_loss = 0.0
        total_elements = 0
        with torch.no_grad():
            for X, Y in loader:
                X, Y = X.to(device), Y.to(device)
                preds = self(X) 
                # if sindy=True forward returns a tuple,
                # we only want the reconstruction
                if isinstance(preds, tuple):
                    preds = preds[0]
                loss = criterion(preds, Y)
                total_loss += loss.item()
                total_elements += Y.numel()
        # mean over every scalar element
        mse = total_loss / total_elements
        return mse