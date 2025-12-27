"# heart-mlops-azure" 

1. Data aquisition:

src/

  data_prep.py 
  sklearn.datasets : fetch_openml(name="heart-disease", version=1, as_frame=True) is used to get dataset
  placed under/data/raw/heart_cleaned.csv
  

4. For Azure ML Ops Pipeline:

    1. Create/update datasource version
    (azureml_py38) azureuser@ag-aiml-compute:~/cloudfiles/code/Users/v_2agho/heart-mlops-azure$ az ml data create \
    --name heart-csv \
    --version 2 \
    --path data/processed/heart_cleaned.csv \
    --type uri_file \
    --workspace-name ag-aiml \
    --resource-group rg-ml-aimllearn
    2. Trigger job
    (azureml_py38) azureuser@ag-aiml-compute:~/cloudfiles/code/Users/v_2agho/heart-mlops-azure$ az ml job         create --file azureml/train_job.yaml

8. Creating and registering Azure ML Container Registry

src/

    score.py :azureml endpoint 
    score.yaml
    best_model.pkl
    environment.yml



Workspace name: ag-aiml
Resource group: rg-ml-aimllearn

register endpoint and deploy the fastapi application:

az ml online-endpoint create \\\
  --name heart-endpoint \\\
  --resource-group rg-ml-aimllearn \\\
  --workspace-name ag-aiml	

az ml online-deployment create \\\
  --name fastapi-deploy \\\
  --endpoint-name heart-endpoint \\\
  --file score.yaml \\\
  --all-traffic

Testing:

Tests folder has test.json
from tests directory

az ml online-endpoint invoke \\\
  --name heart-endpoint \\\
  --request-file test.json

(azureml_py38) azureuser@ag-aiml-compute:~/cloudfiles/code/Users/v_2agho/heart-mlops-azure/tests$ az ml online-endpoint invoke \
  --name heart-endpoint \
  --deployment fastapi-deploy \
  --request-file test.json
"{\"prediction\": 1, \"confidence\": 0.7709388770020859}"