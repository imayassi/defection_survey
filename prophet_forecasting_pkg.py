import pandas as pd
import numpy as np
# from fbprophet import Prophet
import pystan

model_code = 'parameters {real y;} model {y ~ normal(0,1);}'
model = enum.StanModel(model_code=model_code)
y = model.sampling(n_jobs=1).extract()['y']
y.mean()  # with luck the result will be near 0