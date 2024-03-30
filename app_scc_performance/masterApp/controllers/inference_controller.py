import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from typing import List
from sklearn.metrics import mean_squared_error, r2_score
from .preprocess_controller import Preprocessor

#       --- class for inference engine ---
class InferenceEngine:
    # initialize params
    # provide path of model and db_categories file
    def __init__(self, path_model, path_db_categories):        
        self.model = self.load_model(path_model)
        self.preprocessor = Preprocessor(path_db_categories)
        self.feature_names = self.preprocessor.db_categories["metadata"]

    # function to preprocess input data
    def prepare_df(self, data: pd.DataFrame) -> pd.DataFrame:
        # make preprocessing
        print("start preprocessing...")
        input, output = self.preprocessor.preprocessing_pipeline(data)

        return input, output

    # function to load model
    def load_model(self, filename):
        print("loading model...")
        with open(filename, "rb") as file:
            return pickle.load(file)

    # function to get feature names
    @property
    def model_keys(self) -> List[str]:
        return self.feature_names

    # function to predict output
    def predict(self, x) -> pd.Series:
        print("generation of predictions...")
        y_pred = self.model.predict(x)
        return pd.Series(y_pred)
    
    # function to evaluate metrics
    def evaluate_metrics(self, y, y_pred):
        print("evaluate metrics")
        rmse_value = np.sqrt(mean_squared_error(y, y_pred))
        r2_value = r2_score(y, y_pred)
        return rmse_value, r2_value
    
    # function to generate explanation
    def explain(self, x, path):
        print("generation of explanations...")
        explainer = shap.TreeExplainer(self.model, x)
        shap_values = explainer(x)
        shap.summary_plot(shap_values, x, plot_type = "bar", max_display = 10)
        plt.savefig(path)
    