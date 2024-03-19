import pandas as pd 
from inference_controller import InferenceEngine

# load data
data = pd.read_csv("datasets/test_data.csv")
# filter rows with null values
data = data.dropna().reset_index(drop = True)

# build inference engine object
inference = InferenceEngine("models/xgb_scc_perform_v10.pkl", 
                            "datasets/db_features.json")
# to preprocess
x, y = inference.preprocessing(data)
print(x)
print(y)

# to get prediction
y_pred = inference.predict(data)
print(y_pred)
