import torch
from torch.utils.data import DataLoader
from ..objects.sindy import E_SINDy
from ..models.sequence_models.abstract_sequence import AbstractSequence
from ..models.decoder_models.abstract_decoder import AbstractDecoder
from ..models.latent_forecaster_models.abstract_latent_forecaster import AbstractLatentForecaster
from ..models.decoder_models.mlp_model import MLP
from ..models.decoder_models.unet_model import UNET
from ..models.sequence_models.gru_model import GRU
from ..models.sequence_models.lstm_model import LSTM
from ..models.sequence_models.transformer_model import TRANSFORMER
import warnings
from ..objects.dataset import TimeSeriesDataset
from ..models.latent_forecaster_models.sindy import SINDy_Forecaster
from ..models.latent_forecaster_models.lstm import LSTM_Forecaster
from ..objects.device import get_device




SEQUENCE_MODELS = {
    "LSTM": LSTM,
    "TRANSFORMER": TRANSFORMER,
    "GRU": GRU,
}

DECODER_MODELS = {
    "MLP": MLP,
    "UNET": UNET,
}

LATENT_FORECASTER_MODELS = {
    "SINDY_FORECASTER": SINDy_Forecaster,
    "LSTM_FORECASTER": LSTM_Forecaster
}

NUM_REPLICATES = 10

class SHRED(torch.nn.Module):
    """
    SHallow REcurrent Decoder (SHRED) neural network architecture.

    SHRED leverages a sequence model to learn a latent representation of the temporal dynamics of sensor measurements, a 
    latent forecaster model to forecast the latent space into the future, and a decoder model to learn a mapping between 
    the latent space and the high-dimensional full-state space. The SHRED architecture enables accurate full-state 
    reconstructions and forecasts from limited sensors.

    Parameters
    ----------
    sequence_model : AbstractSequence or str, optional
        Sequence model instance (GRU, LSTM, Transformer) or its name.
        Default None → LSTM if not using SINDy_Forecaster, otherwise GRU.
    decoder_model : AbstractDecoder or str, optional
        Decoder model instance (MLP, UNET) or its name.
        Default None → MLP.
    latent_forecaster : AbstractLatentForecaster or str, optional
        Latent forecaster instance (SINDy_Forecaster, LSTM_Forecaster) or its name.
        Default None → no latent forecaster.

    Attributes
    ----------
    sequence : AbstractSequence
        The sequence model that encodes the temporal dynamics of sensor measurements in the latent space.
    decoder : AbstractDecoder
        The decoder model that maps the latent space back to the full‐state space.
    latent_forecaster : AbstractLatentForecaster or None
        The latent forecaster that forecasts future latent space states.
        If None, no latent forecaster is used.

    Examples
    --------
    >>> # basic SHRED
    >>> model = SHRED(sequence_model='LSTM', decoder_model='MLP', latent_forecaster='LSTM_Forecaster')
    >>> # SINDy SHRED
    >>> model = SHRED(sequence_model='GRU', decoder_model='MLP', latent_forecaster='SINDy_Forecaster')
    """
    def __init__(self, sequence_model: AbstractSequence = None, decoder_model: AbstractDecoder = None,
                 latent_forecaster: AbstractLatentForecaster = None):
        """
        Initialize a SHallow REcurrent Decoder (SHRED) model.

        Parameters
        ----------
        sequence_model : AbstractSequence or str, optional
            Sequence model instance (GRU, LSTM, Transformer) or its name.
            Default None → LSTM if not using SINDy_Forecaster, otherwise GRU.
        decoder_model : AbstractDecoder or str, optional
            Decoder model instance (MLP, UNET) or its name.
            Default None → MLP.
        latent_forecaster : AbstractLatentForecaster or str, optional
            Latent forecaster instance (SINDy_Forecaster, LSTM_Forecaster) or its name.
            Default None → no latent forecaster.

        Raises
        ------
        ValueError
            If a string name is given but not found in the corresponding model mapping.
        TypeError
            If an object of the wrong type is passed in for any of the three arguments.
        """
        super().__init__()
        if sequence_model is None:
            if (isinstance(latent_forecaster, SINDy_Forecaster) or 
                (isinstance(latent_forecaster, str) and latent_forecaster.upper() == "SINDY_FORECASTER")):
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
            raise TypeError("Invalid type for 'sequence_model'. Must be str or an AbstractSequence instance or None.")

        if decoder_model is None:
            self.decoder = MLP()
        elif isinstance(decoder_model, AbstractDecoder):
            if latent_forecaster is not None and not isinstance(decoder_model, MLP):
                warnings.warn(
                    "`latent_forecaster` is not None, but `decoder_model` is not an instance of `MLP`. "
                    "The decoder is being overridden with `MLP()` for compatibility with the `latent_forecaster`.",
                    UserWarning
                )
                self.decoder = MLP()
            else:
                self.decoder = decoder_model
        elif isinstance(decoder_model, str):
            decoder_model = decoder_model.upper()
            if decoder_model not in DECODER_MODELS:
                raise ValueError(f"Invalid decoder model: {decoder_model}. Choose from: {list(DECODER_MODELS.keys())}")
            if latent_forecaster is not None and decoder_model!="MLP":
                warnings.warn(
                    "`latent_forecaster` is not None, but `decoder_model` is not set to \"MLP\". "
                    "The decoder is being overridden with `MLP()` for compatibility with the `latent_forecaster`.",
                    UserWarning
                )
                self.decoder = MLP()
            else:
                self.decoder = DECODER_MODELS[decoder_model]()
        else:
            raise TypeError("Invalid type for 'decoder'. Must be str or an AbstractDecoder instance or None.")
        self.num_replicates = NUM_REPLICATES
        self.latent_forecaster = None
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
        """
        Forward pass through the SHRED model.

        Parameters
        ----------
        x : torch.Tensor
            Input sensor sequences of shape (batch_size, lags, n_sensors).
        sindy : bool, optional
            Whether to compute SINDy regularization terms. Defaults to False.

        Returns
        -------
        torch.Tensor or tuple
            If sindy=False: Reconstructed full-state tensor.
            If sindy=True: Tuple of (reconstruction, target_latents, predicted_latents).
        """
        if sindy == True:
            h_out = self.sequence(x)
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

    
    def _seq_model_outputs(self, x, sindy=False):
        """
        Get sequence model outputs with optional SINDy ensemble computation.

        Parameters
        ----------
        x : torch.Tensor
            Input sensor sequences.
        sindy : bool, optional
            Whether to compute SINDy ensemble outputs. Defaults to False.

        Returns
        -------
        torch.Tensor or tuple
            Sequence model outputs.
        """
        if sindy == True:
            h_out = self.sequence(x)
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

    def _sindys_threshold(self, threshold):
        """
        Apply thresholding to SINDy coefficients.

        Parameters
        ----------
        threshold : float
            Threshold value for coefficient pruning.
        """
        self.e_sindy.thresholding(threshold)

    def _sindys_add_noise(self, noise):
        """
        Add noise to SINDy coefficients.

        Parameters
        ----------
        noise : float
            Noise level to add.
        """
        self.e_sindy.add_noise(noise)


    # sindy_regularization previously set to 1.0
    def fit(self, train_dataset, val_dataset,  batch_size=64, num_epochs=200, lr=1e-3, sindy_regularization=0,
            optimizer="AdamW", verbose=True, threshold=0.5, base_threshold=0.0, patience=20,
            sindy_thres_epoch=20, weight_decay=0.01):
        """
        Train the SHRED model on the provided datasets.

        Parameters
        ----------
        train_dataset : TimeSeriesDataset
            Training dataset containing sensor sequences and target reconstructions.
        val_dataset : TimeSeriesDataset
            Validation dataset for monitoring training progress.
        batch_size : int, optional
            Batch size for training. Defaults to 64.
        num_epochs : int, optional
            Maximum number of training epochs. Defaults to 200.
        lr : float, optional
            Learning rate. Defaults to 1e-3.
        sindy_regularization : float, optional
            Weight for SINDy regularization term. Defaults to 0.
        optimizer : str, optional
            Optimizer type. Defaults to "AdamW".
        verbose : bool, optional
            Whether to print training progress. Defaults to True.
        threshold : float, optional
            SINDy thresholding value. Defaults to 0.5.
        base_threshold : float, optional
            Base threshold for SINDy. Defaults to 0.0.
        patience : int, optional
            Early stopping patience. Defaults to 20.
        sindy_thres_epoch : int, optional
            Frequency of SINDy thresholding. Defaults to 20.
        weight_decay : float, optional
            Weight decay for optimizer. Defaults to 0.01.

        Returns
        -------
        np.ndarray
            Array of validation errors for each epoch.
        """
        if not isinstance(self.latent_forecaster, SINDy_Forecaster) and sindy_regularization > 0:
            warnings.warn(
                "`latent_forecaster` is not a SINDy_Forecaster; disabling SINDy regularization.",
                UserWarning
            )
            sindy_regularization = 0

        if sindy_regularization > 0:
            sindy = True
            if not isinstance(self.decoder, MLP):
                warnings.warn("WARNING: SINDy regularization > 0: switching decoder to MLP for compatibility.",
                    UserWarning
                )
                self.decoder = MLP()
        else:
            sindy = False

        device = get_device()
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

        self.sequence.initialize(input_size=input_size, lags=lags, decoder_type=type(self.decoder).__name__)
        self.sequence.to(device)
        self.decoder.initialize(input_size=self.sequence.output_size, output_size=output_size)
        self.decoder.to(device)
        self.to(device)
        # shuffle is False (used to be True)
        train_loader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size)
        criterion = torch.nn.MSELoss()
        optimizer = SHRED._get_optimizer(params=self.parameters(), optimizer=optimizer, lr=lr, weight_decay=weight_decay)
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
                if epoch % sindy_thres_epoch == 0 and epoch != 0:
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
            latents = self._seq_model_outputs(X_all, sindy=False)   # (N_train+N_val, latent_dim)
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


    @staticmethod
    def _get_optimizer(params, optimizer: str, lr: float, weight_decay: float):
        if optimizer == "AdamW":
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        elif optimizer == "Adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif optimizer == "SGD":
            return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
        elif optimizer == "RMSprop":
            return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
        elif optimizer == "Adagrad":
            return torch.optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(
                f"Unsupported optimizer {optimizer!r}. Choose from: "
                "'Adam', 'AdamW', 'SGD', 'RMSprop', or 'Adagrad'."
            )