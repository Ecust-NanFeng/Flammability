import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

books_pure = pd.read_excel(r"Pure substance dataset.xlsx", sheet_name="LFL")
books_mixture = pd.read_excel(r"Refrigerant mixture dataset.xlsx", sheet_name="LFL")

x_pure = books_pure.iloc[:,5:]
x_mixture = books_mixture.iloc[:,9:]
y_pure = books_pure['LFL(vol%)']
y_mixture = books_mixture['LFL(vol%)']
x = pd.concat([x_pure, x_mixture], axis=0)
x.reset_index(drop=True, inplace=True)
y = pd.concat([y_pure, y_mixture], axis=0)
y.reset_index(drop=True, inplace=True)
scaler_x = StandardScaler()
x_std = scaler_x.fit_transform(x)
y_log = np.log10(y)


regressor = SVR(kernel='rbf', C=6, epsilon=0.008)
model = regressor.fit(x_std, y_log)
train_pre = model.predict(x_std)
train_pre = pd.DataFrame(train_pre)
print("result: ", r2_score(y_log, train_pre), np.sqrt(mean_squared_error(y_log, train_pre)), mean_absolute_error(y_log, train_pre))
