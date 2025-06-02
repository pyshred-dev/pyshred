# SINDy-SHRED Tutorial on Sea Surface Temperature

#### Import Libraries


```python
# PYSHRED
from pyshred import DataManager, SHRED, SHREDEngine, SINDy_Forecaster

# Other helper libraries
import matplotlib.pyplot as plt
from scipy.io import loadmat
import torch
import numpy as np
```

#### Load in SST Data


```python
sst_data = np.load("sst_data.npy")
```


```python
# Plotting a single frame
plt.figure()
plt.imshow(sst_data[0]) 
plt.colorbar()
plt.show()
```


    
![png](sindy_shred_sst_files/sindy_shred_sst_5_0.png)
    


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
      <th>sensor_number</th>
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
      <td>(40, 343)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SST</td>
      <td>1</td>
      <td>stationary (random)</td>
      <td>(136, 255)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SST</td>
      <td>2</td>
      <td>stationary (random)</td>
      <td>(66, 219)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SST</td>
      <td>3</td>
      <td>stationary (random)</td>
      <td>(75, 5)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SST</td>
      <td>4</td>
      <td>stationary (random)</td>
      <td>(2, 206)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>SST</td>
      <td>5</td>
      <td>stationary (random)</td>
      <td>(43, 229)</td>
    </tr>
    <tr>
      <th>6</th>
      <td>SST</td>
      <td>6</td>
      <td>stationary (random)</td>
      <td>(121, 31)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>SST</td>
      <td>7</td>
      <td>stationary (random)</td>
      <td>(46, 136)</td>
    </tr>
    <tr>
      <th>8</th>
      <td>SST</td>
      <td>8</td>
      <td>stationary (random)</td>
      <td>(43, 88)</td>
    </tr>
    <tr>
      <th>9</th>
      <td>SST</td>
      <td>9</td>
      <td>stationary (random)</td>
      <td>(147, 153)</td>
    </tr>
    <tr>
      <th>10</th>
      <td>SST</td>
      <td>10</td>
      <td>stationary (random)</td>
      <td>(105, 14)</td>
    </tr>
    <tr>
      <th>11</th>
      <td>SST</td>
      <td>11</td>
      <td>stationary (random)</td>
      <td>(154, 16)</td>
    </tr>
    <tr>
      <th>12</th>
      <td>SST</td>
      <td>12</td>
      <td>stationary (random)</td>
      <td>(142, 151)</td>
    </tr>
    <tr>
      <th>13</th>
      <td>SST</td>
      <td>13</td>
      <td>stationary (random)</td>
      <td>(144, 320)</td>
    </tr>
    <tr>
      <th>14</th>
      <td>SST</td>
      <td>14</td>
      <td>stationary (random)</td>
      <td>(171, 86)</td>
    </tr>
    <tr>
      <th>15</th>
      <td>SST</td>
      <td>15</td>
      <td>stationary (random)</td>
      <td>(55, 318)</td>
    </tr>
    <tr>
      <th>16</th>
      <td>SST</td>
      <td>16</td>
      <td>stationary (random)</td>
      <td>(154, 84)</td>
    </tr>
    <tr>
      <th>17</th>
      <td>SST</td>
      <td>17</td>
      <td>stationary (random)</td>
      <td>(45, 16)</td>
    </tr>
    <tr>
      <th>18</th>
      <td>SST</td>
      <td>18</td>
      <td>stationary (random)</td>
      <td>(142, 319)</td>
    </tr>
    <tr>
      <th>19</th>
      <td>SST</td>
      <td>19</td>
      <td>stationary (random)</td>
      <td>(145, 241)</td>
    </tr>
    <tr>
      <th>20</th>
      <td>SST</td>
      <td>20</td>
      <td>stationary (random)</td>
      <td>(82, 164)</td>
    </tr>
    <tr>
      <th>21</th>
      <td>SST</td>
      <td>21</td>
      <td>stationary (random)</td>
      <td>(54, 195)</td>
    </tr>
    <tr>
      <th>22</th>
      <td>SST</td>
      <td>22</td>
      <td>stationary (random)</td>
      <td>(105, 310)</td>
    </tr>
    <tr>
      <th>23</th>
      <td>SST</td>
      <td>23</td>
      <td>stationary (random)</td>
      <td>(178, 123)</td>
    </tr>
    <tr>
      <th>24</th>
      <td>SST</td>
      <td>24</td>
      <td>stationary (random)</td>
      <td>(112, 242)</td>
    </tr>
    <tr>
      <th>25</th>
      <td>SST</td>
      <td>25</td>
      <td>stationary (random)</td>
      <td>(139, 358)</td>
    </tr>
    <tr>
      <th>26</th>
      <td>SST</td>
      <td>26</td>
      <td>stationary (random)</td>
      <td>(149, 227)</td>
    </tr>
    <tr>
      <th>27</th>
      <td>SST</td>
      <td>27</td>
      <td>stationary (random)</td>
      <td>(10, 98)</td>
    </tr>
    <tr>
      <th>28</th>
      <td>SST</td>
      <td>28</td>
      <td>stationary (random)</td>
      <td>(109, 97)</td>
    </tr>
    <tr>
      <th>29</th>
      <td>SST</td>
      <td>29</td>
      <td>stationary (random)</td>
      <td>(38, 245)</td>
    </tr>
    <tr>
      <th>30</th>
      <td>SST</td>
      <td>30</td>
      <td>stationary (random)</td>
      <td>(97, 217)</td>
    </tr>
    <tr>
      <th>31</th>
      <td>SST</td>
      <td>31</td>
      <td>stationary (random)</td>
      <td>(104, 23)</td>
    </tr>
    <tr>
      <th>32</th>
      <td>SST</td>
      <td>32</td>
      <td>stationary (random)</td>
      <td>(155, 71)</td>
    </tr>
    <tr>
      <th>33</th>
      <td>SST</td>
      <td>33</td>
      <td>stationary (random)</td>
      <td>(134, 41)</td>
    </tr>
    <tr>
      <th>34</th>
      <td>SST</td>
      <td>34</td>
      <td>stationary (random)</td>
      <td>(66, 215)</td>
    </tr>
    <tr>
      <th>35</th>
      <td>SST</td>
      <td>35</td>
      <td>stationary (random)</td>
      <td>(173, 194)</td>
    </tr>
    <tr>
      <th>36</th>
      <td>SST</td>
      <td>36</td>
      <td>stationary (random)</td>
      <td>(172, 139)</td>
    </tr>
    <tr>
      <th>37</th>
      <td>SST</td>
      <td>37</td>
      <td>stationary (random)</td>
      <td>(57, 337)</td>
    </tr>
    <tr>
      <th>38</th>
      <td>SST</td>
      <td>38</td>
      <td>stationary (random)</td>
      <td>(56, 298)</td>
    </tr>
    <tr>
      <th>39</th>
      <td>SST</td>
      <td>39</td>
      <td>stationary (random)</td>
      <td>(125, 245)</td>
    </tr>
    <tr>
      <th>40</th>
      <td>SST</td>
      <td>40</td>
      <td>stationary (random)</td>
      <td>(160, 347)</td>
    </tr>
    <tr>
      <th>41</th>
      <td>SST</td>
      <td>41</td>
      <td>stationary (random)</td>
      <td>(166, 283)</td>
    </tr>
    <tr>
      <th>42</th>
      <td>SST</td>
      <td>42</td>
      <td>stationary (random)</td>
      <td>(85, 26)</td>
    </tr>
    <tr>
      <th>43</th>
      <td>SST</td>
      <td>43</td>
      <td>stationary (random)</td>
      <td>(62, 346)</td>
    </tr>
    <tr>
      <th>44</th>
      <td>SST</td>
      <td>44</td>
      <td>stationary (random)</td>
      <td>(96, 65)</td>
    </tr>
    <tr>
      <th>45</th>
      <td>SST</td>
      <td>45</td>
      <td>stationary (random)</td>
      <td>(121, 335)</td>
    </tr>
    <tr>
      <th>46</th>
      <td>SST</td>
      <td>46</td>
      <td>stationary (random)</td>
      <td>(128, 251)</td>
    </tr>
    <tr>
      <th>47</th>
      <td>SST</td>
      <td>47</td>
      <td>stationary (random)</td>
      <td>(122, 62)</td>
    </tr>
    <tr>
      <th>48</th>
      <td>SST</td>
      <td>48</td>
      <td>stationary (random)</td>
      <td>(96, 69)</td>
    </tr>
    <tr>
      <th>49</th>
      <td>SST</td>
      <td>49</td>
      <td>stationary (random)</td>
      <td>(73, 237)</td>
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
      <th></th>
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
      <td>11.74</td>
      <td>10.08</td>
      <td>22.190000</td>
      <td>0.0</td>
      <td>-1.80</td>
      <td>11.12</td>
      <td>25.069999</td>
      <td>2.78</td>
      <td>0.0</td>
      <td>3.28</td>
      <td>...</td>
      <td>0.15</td>
      <td>-0.0</td>
      <td>0.0</td>
      <td>20.200000</td>
      <td>29.159999</td>
      <td>22.689999</td>
      <td>15.22</td>
      <td>21.740000</td>
      <td>28.719999</td>
      <td>24.539999</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11.67</td>
      <td>10.21</td>
      <td>22.100000</td>
      <td>0.0</td>
      <td>-1.80</td>
      <td>10.70</td>
      <td>24.049999</td>
      <td>2.12</td>
      <td>0.0</td>
      <td>3.61</td>
      <td>...</td>
      <td>0.06</td>
      <td>-0.0</td>
      <td>0.0</td>
      <td>19.910000</td>
      <td>28.129999</td>
      <td>23.609999</td>
      <td>14.96</td>
      <td>22.589999</td>
      <td>28.189999</td>
      <td>24.279999</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11.73</td>
      <td>10.61</td>
      <td>21.890000</td>
      <td>0.0</td>
      <td>-1.80</td>
      <td>10.29</td>
      <td>24.849999</td>
      <td>1.53</td>
      <td>0.0</td>
      <td>3.63</td>
      <td>...</td>
      <td>0.31</td>
      <td>-0.0</td>
      <td>0.0</td>
      <td>19.170000</td>
      <td>28.789999</td>
      <td>23.079999</td>
      <td>15.90</td>
      <td>21.410000</td>
      <td>28.639999</td>
      <td>24.899999</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.33</td>
      <td>10.91</td>
      <td>21.600000</td>
      <td>0.0</td>
      <td>-1.80</td>
      <td>9.87</td>
      <td>24.879999</td>
      <td>1.68</td>
      <td>0.0</td>
      <td>3.72</td>
      <td>...</td>
      <td>0.53</td>
      <td>-0.0</td>
      <td>0.0</td>
      <td>19.180000</td>
      <td>28.539999</td>
      <td>23.349999</td>
      <td>17.87</td>
      <td>22.350000</td>
      <td>28.559999</td>
      <td>24.389999</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11.17</td>
      <td>11.19</td>
      <td>21.590000</td>
      <td>0.0</td>
      <td>-1.80</td>
      <td>9.45</td>
      <td>24.889999</td>
      <td>1.35</td>
      <td>0.0</td>
      <td>3.76</td>
      <td>...</td>
      <td>0.52</td>
      <td>-0.0</td>
      <td>0.0</td>
      <td>18.770000</td>
      <td>28.059999</td>
      <td>23.999999</td>
      <td>17.69</td>
      <td>22.100000</td>
      <td>28.449999</td>
      <td>24.219999</td>
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
      <td>15.89</td>
      <td>8.09</td>
      <td>24.459999</td>
      <td>0.0</td>
      <td>-1.70</td>
      <td>17.31</td>
      <td>21.770000</td>
      <td>17.67</td>
      <td>0.0</td>
      <td>1.56</td>
      <td>...</td>
      <td>-1.75</td>
      <td>-0.0</td>
      <td>0.0</td>
      <td>21.540000</td>
      <td>27.819999</td>
      <td>18.070000</td>
      <td>11.61</td>
      <td>16.390000</td>
      <td>27.649999</td>
      <td>27.269999</td>
    </tr>
    <tr>
      <th>1396</th>
      <td>15.61</td>
      <td>8.22</td>
      <td>24.159999</td>
      <td>0.0</td>
      <td>-1.78</td>
      <td>16.82</td>
      <td>20.870000</td>
      <td>16.34</td>
      <td>0.0</td>
      <td>1.83</td>
      <td>...</td>
      <td>-1.70</td>
      <td>-0.0</td>
      <td>0.0</td>
      <td>21.690000</td>
      <td>28.039999</td>
      <td>17.930000</td>
      <td>11.92</td>
      <td>16.970000</td>
      <td>27.649999</td>
      <td>26.809999</td>
    </tr>
    <tr>
      <th>1397</th>
      <td>15.17</td>
      <td>8.34</td>
      <td>24.369999</td>
      <td>0.0</td>
      <td>-1.80</td>
      <td>16.12</td>
      <td>20.980000</td>
      <td>14.77</td>
      <td>0.0</td>
      <td>1.41</td>
      <td>...</td>
      <td>-1.65</td>
      <td>-0.0</td>
      <td>0.0</td>
      <td>21.740000</td>
      <td>28.209999</td>
      <td>17.940000</td>
      <td>12.23</td>
      <td>17.360000</td>
      <td>27.799999</td>
      <td>26.839999</td>
    </tr>
    <tr>
      <th>1398</th>
      <td>14.79</td>
      <td>8.75</td>
      <td>24.199999</td>
      <td>0.0</td>
      <td>-1.80</td>
      <td>15.60</td>
      <td>21.330000</td>
      <td>13.03</td>
      <td>0.0</td>
      <td>1.87</td>
      <td>...</td>
      <td>-1.47</td>
      <td>-0.0</td>
      <td>0.0</td>
      <td>21.840000</td>
      <td>28.019999</td>
      <td>18.140000</td>
      <td>12.83</td>
      <td>17.410000</td>
      <td>27.619999</td>
      <td>26.679999</td>
    </tr>
    <tr>
      <th>1399</th>
      <td>14.71</td>
      <td>8.75</td>
      <td>24.239999</td>
      <td>0.0</td>
      <td>-1.80</td>
      <td>14.83</td>
      <td>21.560000</td>
      <td>11.23</td>
      <td>0.0</td>
      <td>2.12</td>
      <td>...</td>
      <td>-1.32</td>
      <td>-0.0</td>
      <td>0.0</td>
      <td>22.449999</td>
      <td>28.499999</td>
      <td>18.500000</td>
      <td>12.71</td>
      <td>17.740000</td>
      <td>28.489999</td>
      <td>26.969999</td>
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
latent_forecaster = SINDy_Forecaster(poly_order=1, include_sine=True, dt=1/5)
```

#### Initialize SHRED


```python
shred = SHRED(sequence_model="GRU", decoder_model="MLP", latent_forecaster=latent_forecaster)
```

#### Fit SHRED


```python
val_errors = shred.fit(train_dataset=train_dataset, val_dataset=val_dataset, num_epochs=10, sindy_thres_epoch=20, sindy_regularization=1)
print('val_errors:', val_errors)
```

    Fitting SindySHRED...
    Epoch 1: Average training loss = 0.091680
    Validation MSE (epoch 1): 0.032684
    Epoch 2: Average training loss = 0.032257
    Validation MSE (epoch 2): 0.015280
    Epoch 3: Average training loss = 0.020602
    Validation MSE (epoch 3): 0.013595
    Epoch 4: Average training loss = 0.018892
    Validation MSE (epoch 4): 0.013358
    Epoch 5: Average training loss = 0.017929
    Validation MSE (epoch 5): 0.013441
    Epoch 6: Average training loss = 0.017431
    Validation MSE (epoch 6): 0.013291
    Epoch 7: Average training loss = 0.017039
    Validation MSE (epoch 7): 0.014045
    Epoch 8: Average training loss = 0.016548
    Validation MSE (epoch 8): 0.013414
    Epoch 9: Average training loss = 0.015898
    Validation MSE (epoch 9): 0.013693
    Epoch 10: Average training loss = 0.015332
    Validation MSE (epoch 10): 0.013768
    val_errors: [0.03268439 0.01527982 0.01359512 0.01335787 0.01344112 0.01329135
     0.01404459 0.01341446 0.0136935  0.01376815]
    

#### Evaluate SHRED


```python
train_mse = shred.evaluate(dataset=train_dataset)
val_mse = shred.evaluate(dataset=val_dataset)
test_mse = shred.evaluate(dataset=test_dataset)
print(f"Train MSE: {train_mse:.3f}")
print(f"Val   MSE: {val_mse:.3f}")
print(f"Test  MSE: {test_mse:.3f}")
```

    Train MSE: 0.010
    Val   MSE: 0.014
    Test  MSE: 0.018
    

#### SINDy Discovered Latent Dynamics


```python
print(shred.latent_forecaster)
```

    (x0)' = 0.269 1 + 0.326 x0 + -0.079 x1 + 0.486 x2
    (x1)' = 0.449 1 + 0.333 x0 + -0.308 x1 + 0.189 x2
    (x2)' = -0.492 1 + -0.529 x0 + 0.191 x1 + -0.292 x2
    

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
init_latents = val_latents[-1] # seed forecaster with final latent space from val
h = len(manager.test_sensor_measurements)
test_latent_from_forecaster = engine.forecast_latent(h=h, init_latents=init_latents)
```

#### Decode Latent Space to Full-State Space


```python
test_prediction = engine.decode(test_latent_from_sensors) # latent space generated from sensor data
test_forecast = engine.decode(test_latent_from_forecaster) # latent space generated from latent forecasted (no sensor data)
```

Compare final frame in prediction and forecast to ground truth:


```python
truth      = sst_data[-1]
prediction = test_prediction['SST'][-1]
forecast   = test_forecast['SST'][-1]

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




    <matplotlib.colorbar.Colorbar at 0x250fe875750>




    
![png](sindy_shred_sst_files/sindy_shred_sst_34_1.png)
    


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
    SST      0.580466  0.761883  0.425608  0.407465
    
    ---------- VAL   ----------
                  MSE     RMSE       MAE        R2
    dataset                                       
    SST      0.936849  0.96791  0.498381 -0.579918
    
    ---------- TEST  ----------
                  MSE      RMSE       MAE        R2
    dataset                                        
    SST      1.248095  1.117182  0.587646 -0.453009
    
