import pandas as pd
import numpy as np
from inference_controller import InferenceEngine
from sklearn.metrics import mean_squared_error, r2_score

# load data
data = pd.read_csv("test_summer.csv")
# filter rows with null values
data = data.dropna()

# build inference engine object
"""
Keyword arguments:
path_model -- write path of ML model
path_db_categories -- write path of json database of categorical features
Return: inference object
"""

inference = InferenceEngine(path_model = "xgb_scc_perform_v10.pkl", 
                            path_db_categories = "db_features.json")

# check metadata
print(inference.model_keys)

# preprocessing
"""
Keyword arguments:
data -- put entire dataset, filtering missing values
Return: features (X) and outputs (Y)
"""
x, y = inference.prepare_df(data)

# to get prediction
"""
Keyword arguments:
data -- put data frame corresponded by dataset after filter rows with null values
Return: vector of predictions
"""
# predict
y_pred = inference.predict(x)
print(y_pred)
y_pred.to_csv("xgb_holger_v1.csv")

# evaluate
rmse_val = np.sqrt(mean_squared_error(y, y_pred))
print(f"R-MSE = {rmse_val}")
r2_val = r2_score(y, y_pred)
print(f"R2 = {r2_val}")
