
from IPython import get_ipython

get_ipython().system('pip install sklearn --upgrade')
get_ipython().system('pip install joblib')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

keplar_data = pd.read_csv("Resources/exoplanet_data.csv")
keplar_data = keplar_data.dropna(axis='columns', how='all')
keplar_data = keplar_data.dropna()

# ------------------------------------------------------------
# SELECT & SPLIT DATA
selected_features = keplar_data[['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co',
       'koi_fpflag_ec', 'koi_period', 'koi_period_err1', 'koi_period_err2',
       'koi_time0bk', 'koi_time0bk_err1', 'koi_time0bk_err2', 'koi_impact',
       'koi_impact_err1', 'koi_impact_err2', 'koi_duration',
       'koi_duration_err1', 'koi_duration_err2', 'koi_depth', 'koi_depth_err1',
       'koi_depth_err2', 'koi_prad', 'koi_prad_err1', 'koi_prad_err2',
       'koi_teq', 'koi_insol', 'koi_insol_err1', 'koi_insol_err2',
       'koi_model_snr', 'koi_tce_plnt_num', 'koi_steff', 'koi_steff_err1',
       'koi_steff_err2', 'koi_slogg', 'koi_slogg_err1', 'koi_slogg_err2',
       'koi_srad', 'koi_srad_err1', 'koi_srad_err2', 'ra', 'dec',
       'koi_kepmag']]

x = selected_features
y = keplar_data[['koi_disposition']]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

x_scaler = StandardScaler().fit(x_train)
y_encoded = LabelEncoder().fit(y_train)

x_scaler_train = x_scaler.transform(x_train)
x_scaler_test = x_scaler.transform(x_test)
y_encoded_train = y_encoded.transform(y_train)
y_encoded_test = y_encoded.transform(y_test)

model3 = SVC(kernel='linear')
model3.fit(x_scaler_train, y_encoded_train)
predictions = model3.predict(x_scaler_test)

print(f"Training Data Score: {model3.score(x_scaler_train, y_encoded_train)}")
print(f"Testing Data Score: {model3.score(x_scaler_test, y_encoded_test)}")

param_grid = {"C": [1, 5, 10,50],
            "gamma": [0.0001, 0.0005, 0.001, 0.005]}
grid = GridSearchCV(model3, param_grid, verbose=3)

# Training the model with GridSearch
grid.fit(x_scaler_train, y_encoded_train)

print("-------------------------------------------------------------------------")
print("-------------------------------------------------------------------------")
print(grid.best_params_)
print(grid.best_score_)

import joblib
filename = 'best_model.sav'
joblib.dump(best_model, filename)


