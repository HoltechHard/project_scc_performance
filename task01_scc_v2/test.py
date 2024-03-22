import pandas as pd 
from inference_controller import InferenceEngine

# load data
data = pd.read_csv("datasets/test_summer.csv")
# filter rows with null values
data = data.dropna()

# build inference engine object
"""
Keyword arguments:
path_model -- write path of ML model
path_db_categories -- write path of json database of categorical features
Return: inference object
"""

inference = InferenceEngine(path_model = "models/lgbm_scc_mod_v1.pkl", 
                            path_metadata = "metadata/scc_metadata.json")

# check metadata
print(inference.model_keys)

# preprocessing
"""
Keyword arguments:
data -- put entire dataset, filtering missing values
Return: features (X) and outputs [in log10 scale] (Y)
"""
x, y = inference.prepare_df(data)

# to get prediction
"""
Keyword arguments:
data -- put data frame corresponded by dataset after filter rows with null values
Return: vector of predictions [in seconds]
"""
y_pred = inference.predict(x)
print(y_pred)
y_pred.to_csv("results/lgbm_holger_v2.csv")

# evaluate model
rmse, r2 = inference.evaluate(y, y_pred)
print(f"rmse = {rmse}")
print(f"r2 = {r2}")
