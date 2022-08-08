
from statsmodels.tsa.ar_model import AutoReg

x = data_train["co2"].values
p = 3

ar_model = AutoReg(x, seasonal=True, lags=np.arange(1, p+1), trend='t', period=12).fit()
x_forecast = ar_model.predict(start=len(x), end=2*len(x))

t = np.arange(len(x))
t_forecast = len(x) + np.arange(len(x_forecast))
plt.plot(t, x)
plt.plot(t_forecast, x_forecast)
plt.title("forecast (statsmodels)")
