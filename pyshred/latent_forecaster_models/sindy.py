import pysindy as ps

class SINDy_Forecaster():
    
    def __init__(self, latents, dt, poly_order=1, optimizer = ps.STLSQ(threshold=0.0, alpha=0.05), diff_method = ps.differentiation.FiniteDifference()):
        self.model = ps.SINDy(
            optimizer = optimizer,
            differentiation_method = diff_method,
            feature_library = ps.PolynomialLibrary(degree=poly_order)
        )
        self.model.fit(latents, t=dt)

def evaluate(self, init, test_dataset, inverse_transform=True):
    pass

# thoughts:
# model only performs prediction error
# forecasting error requires a forecasting model which we fit later