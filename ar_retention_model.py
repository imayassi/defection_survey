import pandas as pd
from pandas import Series
from matplotlib import pyplot
from pandas.tools.plotting import lag_plot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
import numpy as np

df=pd.read_csv('test_fcst.csv', index_col='TAX_DAY', sep=',')
# X = np.log(df.values.astype(float))
# size = int(len(X))
# print size
# train, test = X[0:size], X[size:len(X)]
# print train, test
# autocorrelation_plot(df)
# pyplot.show()
# ar_model = AR(np.log(df.values))
# ar_model_fit = ar_model.fit()
# x=ar_model_fit.k_ar
#
# print('Lag: %s' % ar_model_fit.k_ar)
#
# model = ARIMA(np.log(df.values.astype(float)), order=(4,1,0))
# model_fit = model.fit(disp=0)
# print(model_fit.summary())
# # plot residual errors
# # residuals = DataFrame(model_fit.resid)
# # residuals.plot()
# # pyplot.show()
# # residuals.plot(kind='kde')
# # pyplot.show()
# # print(residuals.describe())
#
#
# history = [x for x in train]
# predictions = list()
# for t in range(173):
# 	model = ARIMA(history, order=(4,1,0))
# 	model_fit = model.fit(disp=0)
# 	output = model_fit.forecast()
# 	yhat = output[0]
# 	predictions.append(yhat)
# 	# obs = test[t]
# 	# history.append(obs)
# 	print('predicted=%f' % (yhat))
# # error = mean_squared_error(test, predictions)
# # print('Test MSE: %.3f' % error)
# # plot
# # pyplot.plot(test)
# pyplot.plot(predictions, color='red')
# pyplot.show()
#
#
#
#


plot_acf(df, lags=172)
pyplot.show()
# split dataset
X = df.values
train, test = X[1:len(X)], X[len(X)-172:]
print len(train)
# train autoregression
model = AR(train)
model_fit = model.fit()
print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+172, dynamic=False)
for i in range(len(predictions)):
	print('predicted=%f, expected=%f' % (predictions[i], test[i]))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot results
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
df.plot()
pyplot.show()
lag_plot(df)
pyplot.show()


