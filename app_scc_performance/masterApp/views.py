from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_protect, csrf_exempt
from django.conf import settings

# Create your views here.
import pandas as pd
from datetime import datetime
from .controllers.inference_controller import InferenceEngine

# load index.html
@csrf_protect
@csrf_exempt
def index(request):
    if request.method == 'POST' and request.FILES:
        # take requests        
        file_model = request.FILES['fileModel']
        file_metadata = request.FILES['fileMetadata']
        file_data = request.FILES['fileData']
        # path for media files /storage/
        path_files = settings.MEDIA_ROOT
        # upload files
        upload_file(file_model, path_files)
        upload_file(file_metadata, path_files)
        upload_file(file_data, path_files)
        
        # load data
        data = pd.read_csv(f"{path_files}/{file_data.name}")
        # filter rows with null values
        data = data.dropna()
        # build inference engine 
        inference = InferenceEngine(path_model = f"{path_files}/{file_model.name}",
                                    path_db_categories = f"{path_files}/{file_metadata.name}")
        # make preprocessing
        x, y = inference.prepare_df(data)
        # make prediction
        y_pred = inference.predict(x)
        # save predictions in csv file
        timestamp = datetime.timestamp(datetime.now())
        download_path = f"{path_files}/result_{timestamp}.csv" 
        y_pred.to_csv(download_path)
        # evaluate model
        rmse_val, r2_val = inference.evaluate_metrics(y, y_pred)        

        # generate explainer
        #shap_path = f"{path_files}/shap_{timestamp}.png"
        #inference.explain(x, shap_path)

        return JsonResponse({
            'success': True,
            'dataset': file_data.name,            
            'rmse': rmse_val,
            'r2': r2_val,
            'download_path': download_path,
            #'shap_path': shap_path
        })
    
    return render(request, 'index.html')

# function to upload 1 file
def upload_file(file, path):
    # collect information about file
    filename = file.name
    file_dir = f"{path}/{filename}"

    # save the file
    with open(file_dir, "wb") as destination:
        for chunk in file.chunks():
            destination.write(chunk)
