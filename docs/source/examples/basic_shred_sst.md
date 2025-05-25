# Basic SHRED Tutorial on Sea Surface Temperature


```python
%load_ext autoreload
%autoreload 2
```

#### Import Libraries


```python
# PYSHRED
from pyshred import DataManager, SHRED, SHREDEngine, LSTM_Forecaster

# Other helper libraries
import matplotlib.pyplot as plt
from scipy.io import loadmat
import torch
import numpy as np
```

#### Load in SST Data


```python
mat = loadmat("SST_data.mat")
sst_data = mat['Z'].T
sst_data = sst_data.reshape(1400, 180, 360)
sst_data.shape
```




    (1400, 180, 360)




```python
# Plotting a single frame
plt.figure()
plt.imshow(sst_data[0]) 
plt.colorbar()
plt.show()
```


    
![png](basic_shred_sst_files/basic_shred_sst_6_0.png)
    


#### Initialize Data Manager


```python
manager = DataManager(
    lags = 52,
    train_size = 0.8,
    val_size = 0.1,
    test_size = 0.1,
)
```

#### Add datasets and sensors


```python
manager.add_data(
    data = sst_data,
    id = "SST",
    random = 50,
    # mobile=,
    # stationary=,
    # measurements=,
    compress=False,
)
```

#### Analyze sensor summary


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
      <th>number</th>
      <th>type</th>
      <th>loc/traj</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SST</td>
      <td>0</td>
      <td>stationary (random)</td>
      <td>(141, 211)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SST</td>
      <td>1</td>
      <td>stationary (random)</td>
      <td>(113, 95)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SST</td>
      <td>2</td>
      <td>stationary (random)</td>
      <td>(179, 325)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SST</td>
      <td>3</td>
      <td>stationary (random)</td>
      <td>(54, 185)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SST</td>
      <td>4</td>
      <td>stationary (random)</td>
      <td>(171, 72)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>SST</td>
      <td>5</td>
      <td>stationary (random)</td>
      <td>(29, 1)</td>
    </tr>
    <tr>
      <th>6</th>
      <td>SST</td>
      <td>6</td>
      <td>stationary (random)</td>
      <td>(78, 126)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>SST</td>
      <td>7</td>
      <td>stationary (random)</td>
      <td>(118, 237)</td>
    </tr>
    <tr>
      <th>8</th>
      <td>SST</td>
      <td>8</td>
      <td>stationary (random)</td>
      <td>(177, 60)</td>
    </tr>
    <tr>
      <th>9</th>
      <td>SST</td>
      <td>9</td>
      <td>stationary (random)</td>
      <td>(100, 102)</td>
    </tr>
    <tr>
      <th>10</th>
      <td>SST</td>
      <td>10</td>
      <td>stationary (random)</td>
      <td>(56, 26)</td>
    </tr>
    <tr>
      <th>11</th>
      <td>SST</td>
      <td>11</td>
      <td>stationary (random)</td>
      <td>(27, 240)</td>
    </tr>
    <tr>
      <th>12</th>
      <td>SST</td>
      <td>12</td>
      <td>stationary (random)</td>
      <td>(146, 50)</td>
    </tr>
    <tr>
      <th>13</th>
      <td>SST</td>
      <td>13</td>
      <td>stationary (random)</td>
      <td>(71, 26)</td>
    </tr>
    <tr>
      <th>14</th>
      <td>SST</td>
      <td>14</td>
      <td>stationary (random)</td>
      <td>(171, 83)</td>
    </tr>
    <tr>
      <th>15</th>
      <td>SST</td>
      <td>15</td>
      <td>stationary (random)</td>
      <td>(5, 169)</td>
    </tr>
    <tr>
      <th>16</th>
      <td>SST</td>
      <td>16</td>
      <td>stationary (random)</td>
      <td>(121, 204)</td>
    </tr>
    <tr>
      <th>17</th>
      <td>SST</td>
      <td>17</td>
      <td>stationary (random)</td>
      <td>(41, 337)</td>
    </tr>
    <tr>
      <th>18</th>
      <td>SST</td>
      <td>18</td>
      <td>stationary (random)</td>
      <td>(82, 194)</td>
    </tr>
    <tr>
      <th>19</th>
      <td>SST</td>
      <td>19</td>
      <td>stationary (random)</td>
      <td>(145, 42)</td>
    </tr>
    <tr>
      <th>20</th>
      <td>SST</td>
      <td>20</td>
      <td>stationary (random)</td>
      <td>(140, 205)</td>
    </tr>
    <tr>
      <th>21</th>
      <td>SST</td>
      <td>21</td>
      <td>stationary (random)</td>
      <td>(44, 127)</td>
    </tr>
    <tr>
      <th>22</th>
      <td>SST</td>
      <td>22</td>
      <td>stationary (random)</td>
      <td>(140, 51)</td>
    </tr>
    <tr>
      <th>23</th>
      <td>SST</td>
      <td>23</td>
      <td>stationary (random)</td>
      <td>(118, 342)</td>
    </tr>
    <tr>
      <th>24</th>
      <td>SST</td>
      <td>24</td>
      <td>stationary (random)</td>
      <td>(19, 326)</td>
    </tr>
    <tr>
      <th>25</th>
      <td>SST</td>
      <td>25</td>
      <td>stationary (random)</td>
      <td>(105, 28)</td>
    </tr>
    <tr>
      <th>26</th>
      <td>SST</td>
      <td>26</td>
      <td>stationary (random)</td>
      <td>(64, 43)</td>
    </tr>
    <tr>
      <th>27</th>
      <td>SST</td>
      <td>27</td>
      <td>stationary (random)</td>
      <td>(116, 1)</td>
    </tr>
    <tr>
      <th>28</th>
      <td>SST</td>
      <td>28</td>
      <td>stationary (random)</td>
      <td>(125, 73)</td>
    </tr>
    <tr>
      <th>29</th>
      <td>SST</td>
      <td>29</td>
      <td>stationary (random)</td>
      <td>(21, 354)</td>
    </tr>
    <tr>
      <th>30</th>
      <td>SST</td>
      <td>30</td>
      <td>stationary (random)</td>
      <td>(97, 169)</td>
    </tr>
    <tr>
      <th>31</th>
      <td>SST</td>
      <td>31</td>
      <td>stationary (random)</td>
      <td>(25, 287)</td>
    </tr>
    <tr>
      <th>32</th>
      <td>SST</td>
      <td>32</td>
      <td>stationary (random)</td>
      <td>(94, 34)</td>
    </tr>
    <tr>
      <th>33</th>
      <td>SST</td>
      <td>33</td>
      <td>stationary (random)</td>
      <td>(21, 341)</td>
    </tr>
    <tr>
      <th>34</th>
      <td>SST</td>
      <td>34</td>
      <td>stationary (random)</td>
      <td>(130, 198)</td>
    </tr>
    <tr>
      <th>35</th>
      <td>SST</td>
      <td>35</td>
      <td>stationary (random)</td>
      <td>(55, 268)</td>
    </tr>
    <tr>
      <th>36</th>
      <td>SST</td>
      <td>36</td>
      <td>stationary (random)</td>
      <td>(177, 336)</td>
    </tr>
    <tr>
      <th>37</th>
      <td>SST</td>
      <td>37</td>
      <td>stationary (random)</td>
      <td>(32, 218)</td>
    </tr>
    <tr>
      <th>38</th>
      <td>SST</td>
      <td>38</td>
      <td>stationary (random)</td>
      <td>(37, 33)</td>
    </tr>
    <tr>
      <th>39</th>
      <td>SST</td>
      <td>39</td>
      <td>stationary (random)</td>
      <td>(135, 86)</td>
    </tr>
    <tr>
      <th>40</th>
      <td>SST</td>
      <td>40</td>
      <td>stationary (random)</td>
      <td>(41, 336)</td>
    </tr>
    <tr>
      <th>41</th>
      <td>SST</td>
      <td>41</td>
      <td>stationary (random)</td>
      <td>(171, 246)</td>
    </tr>
    <tr>
      <th>42</th>
      <td>SST</td>
      <td>42</td>
      <td>stationary (random)</td>
      <td>(156, 96)</td>
    </tr>
    <tr>
      <th>43</th>
      <td>SST</td>
      <td>43</td>
      <td>stationary (random)</td>
      <td>(156, 11)</td>
    </tr>
    <tr>
      <th>44</th>
      <td>SST</td>
      <td>44</td>
      <td>stationary (random)</td>
      <td>(40, 332)</td>
    </tr>
    <tr>
      <th>45</th>
      <td>SST</td>
      <td>45</td>
      <td>stationary (random)</td>
      <td>(81, 95)</td>
    </tr>
    <tr>
      <th>46</th>
      <td>SST</td>
      <td>46</td>
      <td>stationary (random)</td>
      <td>(37, 86)</td>
    </tr>
    <tr>
      <th>47</th>
      <td>SST</td>
      <td>47</td>
      <td>stationary (random)</td>
      <td>(60, 197)</td>
    </tr>
    <tr>
      <th>48</th>
      <td>SST</td>
      <td>48</td>
      <td>stationary (random)</td>
      <td>(87, 246)</td>
    </tr>
    <tr>
      <th>49</th>
      <td>SST</td>
      <td>49</td>
      <td>stationary (random)</td>
      <td>(38, 212)</td>
    </tr>
  </tbody>
</table>
</div>




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
      <th>data id</th>
      <th>SST-0</th>
      <th>SST-1</th>
      <th>SST-2</th>
      <th>SST-3</th>
      <th>SST-4</th>
      <th>SST-5</th>
      <th>SST-6</th>
      <th>SST-7</th>
      <th>SST-8</th>
      <th>SST-9</th>
      <th>...</th>
      <th>SST-40</th>
      <th>SST-41</th>
      <th>SST-42</th>
      <th>SST-43</th>
      <th>SST-44</th>
      <th>SST-45</th>
      <th>SST-46</th>
      <th>SST-47</th>
      <th>SST-48</th>
      <th>SST-49</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10.99</td>
      <td>24.539999</td>
      <td>-0.0</td>
      <td>16.180000</td>
      <td>-0.0</td>
      <td>8.78</td>
      <td>27.589999</td>
      <td>25.419999</td>
      <td>-0.0</td>
      <td>28.479999</td>
      <td>...</td>
      <td>12.04</td>
      <td>-0.0</td>
      <td>-0.0</td>
      <td>-0.26</td>
      <td>11.40</td>
      <td>28.339999</td>
      <td>0.0</td>
      <td>19.690000</td>
      <td>24.539999</td>
      <td>6.78</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11.04</td>
      <td>23.999999</td>
      <td>-0.0</td>
      <td>16.290000</td>
      <td>-0.0</td>
      <td>8.67</td>
      <td>27.389999</td>
      <td>23.869999</td>
      <td>-0.0</td>
      <td>27.929999</td>
      <td>...</td>
      <td>11.99</td>
      <td>-0.0</td>
      <td>-0.0</td>
      <td>0.13</td>
      <td>11.11</td>
      <td>28.249999</td>
      <td>0.0</td>
      <td>19.660000</td>
      <td>24.559999</td>
      <td>6.43</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.86</td>
      <td>24.399999</td>
      <td>-0.0</td>
      <td>15.800000</td>
      <td>-0.0</td>
      <td>8.33</td>
      <td>27.239999</td>
      <td>24.569999</td>
      <td>-0.0</td>
      <td>28.179999</td>
      <td>...</td>
      <td>12.08</td>
      <td>-0.0</td>
      <td>-0.0</td>
      <td>0.35</td>
      <td>11.21</td>
      <td>28.259999</td>
      <td>0.0</td>
      <td>19.240000</td>
      <td>24.949999</td>
      <td>5.96</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10.33</td>
      <td>24.399999</td>
      <td>-0.0</td>
      <td>15.870000</td>
      <td>-0.0</td>
      <td>8.47</td>
      <td>27.019999</td>
      <td>24.659999</td>
      <td>-0.0</td>
      <td>28.279999</td>
      <td>...</td>
      <td>11.65</td>
      <td>-0.0</td>
      <td>-0.0</td>
      <td>0.57</td>
      <td>10.90</td>
      <td>28.129999</td>
      <td>0.0</td>
      <td>19.040000</td>
      <td>25.329999</td>
      <td>5.91</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10.91</td>
      <td>24.279999</td>
      <td>-0.0</td>
      <td>15.350000</td>
      <td>-0.0</td>
      <td>8.54</td>
      <td>27.229999</td>
      <td>24.159999</td>
      <td>-0.0</td>
      <td>28.359999</td>
      <td>...</td>
      <td>11.58</td>
      <td>-0.0</td>
      <td>-0.0</td>
      <td>0.66</td>
      <td>10.74</td>
      <td>28.359999</td>
      <td>0.0</td>
      <td>19.510000</td>
      <td>25.529999</td>
      <td>5.78</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1395</th>
      <td>6.92</td>
      <td>21.100000</td>
      <td>-0.0</td>
      <td>23.529999</td>
      <td>-0.0</td>
      <td>12.98</td>
      <td>30.249999</td>
      <td>19.870000</td>
      <td>-0.0</td>
      <td>27.909999</td>
      <td>...</td>
      <td>16.10</td>
      <td>-0.0</td>
      <td>-0.0</td>
      <td>-1.80</td>
      <td>14.48</td>
      <td>28.389999</td>
      <td>0.0</td>
      <td>25.729999</td>
      <td>24.939999</td>
      <td>13.85</td>
    </tr>
    <tr>
      <th>1396</th>
      <td>7.32</td>
      <td>21.170000</td>
      <td>-0.0</td>
      <td>23.229999</td>
      <td>-0.0</td>
      <td>11.99</td>
      <td>30.039999</td>
      <td>19.990000</td>
      <td>-0.0</td>
      <td>28.089999</td>
      <td>...</td>
      <td>15.78</td>
      <td>-0.0</td>
      <td>-0.0</td>
      <td>-1.80</td>
      <td>14.11</td>
      <td>28.509999</td>
      <td>0.0</td>
      <td>25.789999</td>
      <td>23.649999</td>
      <td>13.28</td>
    </tr>
    <tr>
      <th>1397</th>
      <td>7.82</td>
      <td>21.220000</td>
      <td>-0.0</td>
      <td>22.579999</td>
      <td>-0.0</td>
      <td>11.58</td>
      <td>29.949999</td>
      <td>20.350000</td>
      <td>-0.0</td>
      <td>28.029999</td>
      <td>...</td>
      <td>14.98</td>
      <td>-0.0</td>
      <td>-0.0</td>
      <td>-1.77</td>
      <td>13.61</td>
      <td>28.579999</td>
      <td>0.0</td>
      <td>25.529999</td>
      <td>25.569999</td>
      <td>12.40</td>
    </tr>
    <tr>
      <th>1398</th>
      <td>7.70</td>
      <td>21.510000</td>
      <td>-0.0</td>
      <td>22.160000</td>
      <td>-0.0</td>
      <td>11.08</td>
      <td>29.929999</td>
      <td>21.360000</td>
      <td>-0.0</td>
      <td>28.129999</td>
      <td>...</td>
      <td>14.72</td>
      <td>-0.0</td>
      <td>-0.0</td>
      <td>-1.65</td>
      <td>13.75</td>
      <td>28.699999</td>
      <td>0.0</td>
      <td>25.429999</td>
      <td>24.839999</td>
      <td>12.03</td>
    </tr>
    <tr>
      <th>1399</th>
      <td>7.49</td>
      <td>21.490000</td>
      <td>-0.0</td>
      <td>22.389999</td>
      <td>-0.0</td>
      <td>11.29</td>
      <td>29.929999</td>
      <td>20.840000</td>
      <td>-0.0</td>
      <td>28.129999</td>
      <td>...</td>
      <td>14.33</td>
      <td>-0.0</td>
      <td>-0.0</td>
      <td>-1.72</td>
      <td>13.27</td>
      <td>28.749999</td>
      <td>0.0</td>
      <td>25.219999</td>
      <td>24.499999</td>
      <td>10.96</td>
    </tr>
  </tbody>
</table>
<p>1400 rows × 50 columns</p>
</div>



#### Get train, validation, and test set


```python
train_dataset, val_dataset, test_dataset= manager.prepare()
```

#### Initialize a latent forecaster


```python
latent_lags = 10 # number of timesteps to look back in latent space to build latent sequences
latent_forecaster = LSTM_Forecaster(lags=latent_lags)
```

#### Initialize SHRED


```python
shred = SHRED(sequence_model="LSTM", decoder_model="SDN", latent_forecaster=latent_forecaster)
```

#### Fit SHRED


```python
val_errors = shred.fit(train_dataset=train_dataset, val_dataset=val_dataset, num_epochs=10, thres_epoch=20, sindy_regularization=0)
print('val_errors:', val_errors)
```

    Fitting SHRED...
    Epoch 1: Average training loss = 0.075075
    Validation MSE (epoch 1): 0.038405
    Epoch 2: Average training loss = 0.035112
    Validation MSE (epoch 2): 0.033361
    Epoch 3: Average training loss = 0.030324
    Validation MSE (epoch 3): 0.025519
    Epoch 4: Average training loss = 0.016844
    Validation MSE (epoch 4): 0.011481
    Epoch 5: Average training loss = 0.011579
    Validation MSE (epoch 5): 0.011419
    Epoch 6: Average training loss = 0.011205
    Validation MSE (epoch 6): 0.010817
    Epoch 7: Average training loss = 0.010458
    Validation MSE (epoch 7): 0.010221
    Epoch 8: Average training loss = 0.010263
    Validation MSE (epoch 8): 0.009860
    Epoch 9: Average training loss = 0.009952
    Validation MSE (epoch 9): 0.009608
    Epoch 10: Average training loss = 0.009570
    Validation MSE (epoch 10): 0.009208
    val_errors: [0.03840464 0.03336134 0.02551896 0.01148106 0.01141871 0.01081694
     0.01022083 0.00985987 0.00960797 0.00920787]
    

#### Evaluate SHRED


```python
train_mse = shred.evaluate(dataset=train_dataset)
val_mse = shred.evaluate(dataset=val_dataset)
test_mse = shred.evaluate(dataset=test_dataset)
print(f"Train MSE: {train_mse:.3f}")
print(f"Val   MSE: {val_mse:.3f}")
print(f"Test  MSE: {test_mse:.3f}")
```

    Train MSE: 0.008
    Val   MSE: 0.009
    Test  MSE: 0.011
    

#### Initialize SHRED Engine for Downstream Tasks


```python
engine = SHREDEngine(manager, shred)
```

#### Sensor Measurements to Latent Space


```python
test_latent_from_sensors = engine.sensor_to_latent(manager.test_sensor_measurements)
```

#### Forecast Latent Space (No Sensor Measurements)


```python
val_latents = engine.sensor_to_latent(manager.val_sensor_measurements)
init_latents = val_latents[-latent_lags:] # seed forecaster with final lag timesteps of latent space from val
t = len(manager.test_sensor_measurements)
test_latent_from_forecaster = engine.forecast_latent(t=t, init_latents=init_latents)
```

#### Decode Latent Space to Full-State Space


```python
test_prediction = engine.decode(test_latent_from_sensors) # latent space generated from sensor data
test_forecast = engine.decode(test_latent_from_forecaster) # latent space generated from latent forecasted (no sensor data)
```

Compare final frame in prediction and forecast to ground truth:


```python
truth      = sst_data[-1]
prediction = test_prediction['SST'][t-1]
forecast   = test_forecast['SST'][t-1]

data   = [truth, prediction, forecast]
titles = ["Test Truth Ground Truth", "Test Prediction", "Test Forecast"]

vmin, vmax = np.min([d.min() for d in data]), np.max([d.max() for d in data])

fig, axes = plt.subplots(1, 3, figsize=(20, 4), constrained_layout=True)

for ax, d, title in zip(axes, data, titles):
    im = ax.imshow(d, vmin=vmin, vmax=vmax)
    ax.set(title=title)
    ax.axis("off")

fig.colorbar(im, ax=axes, label="Value", shrink=0.8)
```




    <matplotlib.colorbar.Colorbar at 0x23234091930>




    
![png](basic_shred_sst_files/basic_shred_sst_33_1.png)
    


#### Evaluate MSE on Ground Truth Data


```python
# Train
t_train = len(manager.train_sensor_measurements)
train_Y = {'SST': sst_data[0:t_train]}
train_error = engine.evaluate(manager.train_sensor_measurements, train_Y)

# Val
t_val = len(manager.test_sensor_measurements)
val_Y = {'SST': sst_data[t_train:t_train+t_val]}
val_error = engine.evaluate(manager.val_sensor_measurements, val_Y)

# Test
t_test = len(manager.test_sensor_measurements)
test_Y = {'SST': sst_data[-t_test:]}
test_error = engine.evaluate(manager.test_sensor_measurements, test_Y)

print('---------- TRAIN ----------')
print(train_error)
print('\n---------- VAL   ----------')
print(val_error)
print('\n---------- TEST  ----------')
print(test_error)
```

    ---------- TRAIN ----------
                  MSE      RMSE       MAE        R2
    dataset                                        
    SST      0.393446  0.627253  0.364334  0.448413
    
    ---------- VAL   ----------
                  MSE      RMSE       MAE        R2
    dataset                                        
    SST      0.466176  0.682771  0.385686 -0.232119
    
    ---------- TEST  ----------
                  MSE      RMSE       MAE        R2
    dataset                                        
    SST      0.565816  0.752207  0.438306 -0.531334
    
