"# heart-mlops-azure" 

1. Data aquisition:

src/

  data_prep.py : fetches model performs cleanup 
  sklearn.datasets : fetch_openml(name="heart-disease", version=1, as_frame=True) is used to get dataset
  placed under/data/raw/heart_cleaned.csv
  
  train.py : performs training of 2 models, k-fold validations also logs graphs. in outputs folder saves the "best_model.pkl" based on higher F1 score
  python train.py --data_path '<path>\data\processed\heart_cleaned.csv' --experiment-name "heart-ml-exp"

  Dataset loaded successfully. Shape: (303, 14)
Trained log_reg
Trained rf

log_reg metrics:
  accuracy: 0.8852459016393442
  precision: 0.8787878787878788
  recall: 0.90625
  f1: 0.8923076923076924
  roc_auc: 0.9267241379310345

rf metrics:
  accuracy: 0.8360655737704918
  precision: 0.84375
  recall: 0.84375
  f1: 0.84375
  roc_auc: 0.9326508620689654

Running 5-fold CV for log_reg...
log_reg accuracy: 0.8416
log_reg precision: 0.8195
log_reg recall: 0.9152
log_reg f1: 0.8638
log_reg roc_auc: 0.8914

Running 5-fold CV for rf...
rf accuracy: 0.8185
rf precision: 0.8185
rf recall: 0.8667
rf f1: 0.8397
rf roc_auc: 0.9076
Best model 'log_reg' saved to outputs\best_model.pkl

4. For Azure ML Ops Pipeline:

    1. Create/update datasource version
    $ az ml data create \
    --name heart-csv \
    --version 2 \
    --path data/processed/heart_cleaned.csv \
    --type uri_file \
    --workspace-name ag-aiml \
    --resource-group rg-ml-aimllearn
    2. Trigger job
    $ az ml job         create --file azureml/train_job.yaml

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

az ml online-endpoint invoke \
  --name heart-endpoint \
  --deployment fastapi-deploy \
  --request-file test.json
"{\"prediction\": 1, \"confidence\": 0.7709388770020859}"