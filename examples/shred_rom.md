# ROM-SHRED Tutorial on Kuramoto Sivashinsky

#### Import Libraries


```python
# PYSHRED
%load_ext autoreload
%autoreload 2
from pyshred import ParametricDataManager, SHRED, ParametricSHREDEngine

# Other helper libraries
import matplotlib.pyplot as plt
from scipy.io import loadmat
import torch
import numpy as np
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload
    

#### Load Kuramoto Sivashinsky dataset


```python
import numpy as np
import urllib.request
# URL of the NPZ file
url = 'https://zenodo.org/records/14524524/files/KuramotoSivashinsky_data.npz?download=1'
# Local filename to save the downloaded file
filename = 'KuramotoSivashinsky_data.npz'
# Download the file from the URL
urllib.request.urlretrieve(url, filename)
# Load the data from the NPZ file
dataset = np.load(filename)
```

#### Initialize Data Manager


```python
# Initialize ParametricSHREDDataManager
manager = ParametricDataManager(
    lags = 20,
    train_size = 0.8,
    val_size = 0.1,
    test_size = 0.1,
    )
```

#### Add datasets and sensors


```python
data = dataset['u'] # shape (500, 201, 100)

manager.add_data(
    data=data,
    random=3,
    # stationary=[(15,),(30,),(45,)],
    id = 'KS',
    compress = False
)
```

#### Analyze sensor summary


```python
manager.sensor_measurements_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>KS-0</th>
      <th>KS-1</th>
      <th>KS-2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.154032</td>
      <td>-1.140074</td>
      <td>-0.006130</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.332515</td>
      <td>-1.079287</td>
      <td>-0.023609</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.361393</td>
      <td>-1.070717</td>
      <td>0.002758</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.404951</td>
      <td>-1.049306</td>
      <td>0.015024</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.485893</td>
      <td>-1.031259</td>
      <td>0.017951</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>100495</th>
      <td>0.261712</td>
      <td>-1.793277</td>
      <td>-0.135204</td>
    </tr>
    <tr>
      <th>100496</th>
      <td>-0.622521</td>
      <td>-1.301706</td>
      <td>0.021596</td>
    </tr>
    <tr>
      <th>100497</th>
      <td>-1.591052</td>
      <td>-0.838068</td>
      <td>0.158474</td>
    </tr>
    <tr>
      <th>100498</th>
      <td>-2.129965</td>
      <td>-0.466887</td>
      <td>0.281114</td>
    </tr>
    <tr>
      <th>100499</th>
      <td>-2.005693</td>
      <td>-0.192670</td>
      <td>0.383173</td>
    </tr>
  </tbody>
</table>
<p>100500 rows × 3 columns</p>
</div>




```python
manager.sensor_summary_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>data id</th>
      <th>sensor_number</th>
      <th>type</th>
      <th>loc/traj</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>KS</td>
      <td>0</td>
      <td>stationary</td>
      <td>(15,)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>KS</td>
      <td>1</td>
      <td>stationary</td>
      <td>(30,)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>KS</td>
      <td>2</td>
      <td>stationary</td>
      <td>(45,)</td>
    </tr>
  </tbody>
</table>
</div>



#### Get train, validation, and test set


```python
train_dataset, val_dataset, test_dataset= manager.prepare()
```

#### Initialize SHRED

When using a `ParametricDataManager`, ensure `latent_forecaster` is set to None.


```python
shred = SHRED(sequence_model="LSTM", decoder_model="MLP", latent_forecaster=None)
```

#### Fit SHRED


```python
val_errors = shred.fit(train_dataset=train_dataset, val_dataset=val_dataset, num_epochs=10, sindy_regularization=0)
print('val_errors:', val_errors)
```

    Fitting SHRED...
    Epoch 1: Average training loss = 0.023503
    Validation MSE (epoch 1): 0.015884
    Epoch 2: Average training loss = 0.011217
    Validation MSE (epoch 2): 0.013228
    Epoch 3: Average training loss = 0.009117
    Validation MSE (epoch 3): 0.010834
    Epoch 4: Average training loss = 0.007808
    Validation MSE (epoch 4): 0.009817
    Epoch 5: Average training loss = 0.006982
    Validation MSE (epoch 5): 0.008005
    val_errors: [0.01588413 0.01322768 0.01083446 0.00981664 0.0080053 ]
    

#### Evaluate SHRED


```python
train_mse = shred.evaluate(dataset=train_dataset)
val_mse = shred.evaluate(dataset=val_dataset)
test_mse = shred.evaluate(dataset=test_dataset)
print(f"Train MSE: {train_mse:.3f}")
print(f"Val   MSE: {val_mse:.3f}")
print(f"Test  MSE: {test_mse:.3f}")
```

    Train MSE: 0.006
    Val   MSE: 0.008
    Test  MSE: 0.005
    

#### Initialize Parametric SHRED Engine for Downstream Tasks


```python
engine = ParametricSHREDEngine(manager, shred)
```

#### Sensor Measurements to Latent Space


```python
test_latent_from_sensors = engine.sensor_to_latent(manager.test_sensor_measurements)
```

#### Decode Latent Space to Full-State Space


```python
test_prediction = engine.decode(test_latent_from_sensors) # latent space generated from sensor data
```

Compare prediction against the truth


```python
spatial_shape = data.shape[2:]
truth      = data.reshape(-1, *spatial_shape)
prediction = test_prediction['KS']

compare_data = [truth, prediction]
titles = ["Test Truth Ground Truth", "Test Prediction"]

vmin, vmax = np.min([d.min() for d in compare_data]), np.max([d.max() for d in compare_data])

fig, axes = plt.subplots(1, 2, figsize=(20, 4), constrained_layout=True)

for ax, d, title in zip(axes, compare_data, titles):
    im = ax.imshow(d, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set(title=title)
    ax.axis("off")

fig.colorbar(im, ax=axes, label="Value", shrink=0.8)
```




    <matplotlib.colorbar.Colorbar at 0x2031d9725f0>




    
![png](shred_rom_files/shred_rom_27_1.png)
    


#### Evaluate MSE on Ground Truth Data

Since both number of trajectories (`data.shape[0]`) and number of timesteps (`data.shape[1]`) are both variable, we will leave them combined on the first axis. The remaining axes are all spatial dimensions.


```python
# Train
t_train = len(manager.train_sensor_measurements)
train_Y = {'KS': data[0:t_train].reshape(-1, *spatial_shape)} # unpack the spatial dimensions
train_error = engine.evaluate(manager.train_sensor_measurements, train_Y)

# Val
t_val = len(manager.test_sensor_measurements)
val_Y = {'KS': data[t_train:t_train+t_val].reshape(-1, *spatial_shape)}
val_error = engine.evaluate(manager.val_sensor_measurements, val_Y)

# Test
t_test = len(manager.test_sensor_measurements)
test_Y = {'KS': data[-t_test:].reshape(-1, *spatial_shape)}
test_error = engine.evaluate(manager.test_sensor_measurements, test_Y)

print('---------- TRAIN ----------')
print(train_error)
print('\n---------- VAL   ----------')
print(val_error)
print('\n---------- TEST  ----------')
print(test_error)
```

    ---------- TRAIN ----------
                 MSE      RMSE      MAE        R2
    dataset                                      
    KS       0.18685  0.432262  0.27461  0.857352
    
    ---------- VAL   ----------
                  MSE      RMSE       MAE        R2
    dataset                                        
    KS       0.270706  0.520294  0.336177  0.789836
    
    ---------- TEST  ----------
                  MSE      RMSE       MAE        R2
    dataset                                        
    KS       0.168299  0.410243  0.252061  0.872379
    
