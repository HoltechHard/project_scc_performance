# Project SCC Polytech  

### Task 01  
AI-system based on benchmark of regression models to predict time use of computational resources in SCC Polytech
 
* Projects:
  - app_scc_performance: AI-system for SCC time prediction in python + django + sklean + html + jquery
  - task01_scc: MLOps for regression using time elapsed (in seconds)  
  - task01_scc_v2: MLOps for regression using logarithm10 scale for times of limit, elapsed and wait (in seconds)  
  
* Models:  
  The models can download in the next link:  
  https://drive.google.com/drive/folders/1pPq0k-Gg4WglyxkNj6nKeDNHQv_PQjex?usp=sharing
  
  Models for project task01_scc:  
  - XGBoost: task01_scc/xgb_scc_perform_v10.pkl (25 MB)  
  - LightGBM: task01_scc/lgbm_scc_perform_v10.pkl (16.1 MB)
  
  Models for project task01_scc_v2:  
  - XGBoost: task01_scc_v2/xgb_scc_mod_v1.pkl (318 MB)  
  - LigthGBM: task01_scc_v2/lgbm_scc_mod_v1.pkl (16.5 MB)
   
* Resources:
  - Metadata: datasets/scc_metadata.txt  
  - Database of categorical features: datasets/db_features.json  
  - Class for preprocessing operations: preprocess_controller.py  
  - Class for inference operations: inference_controller.py  
  - Jupyter notebook with MLOps: task01_scc_perform_v6.ipynb  
  - File example to use inference engine: test.py

* System interface:
    
  ![image](https://github.com/HoltechHard/project_scc_performance/assets/35493202/218f3b75-12c4-4e7a-bf28-a1f38f87af12)
   
  Instructions:  
  To run system, run the next commands   
  $ cd app_scc_performance  
  $ python manage.py runserver  
     
  Observations:  
  Currently, system is integrated just for models and metadata of task01_scc  
    
* Results:
  - task01_scc: prediction of time elapsed    
     
  ![image](https://github.com/HoltechHard/project_scc_performance/assets/35493202/8f5c0f41-542a-4c42-bfe1-4f9499cb7454)  

  - task01_scc_v2: prediction of logarithm of time elapsed  
   
  ![image](https://github.com/HoltechHard/project_scc_performance/assets/35493202/7bccb2ff-940c-4685-8904-f32343bbd06d)  
  
  - SHAP vector of 10 most important features  

  ![image](https://github.com/HoltechHard/project_scc_performance/assets/35493202/ebb7f4c7-bfae-4c47-b45b-8f5d4abdf283)


