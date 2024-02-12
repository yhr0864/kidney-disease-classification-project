# kidney-disease-classification-project

## Workflows for each stage

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update the main.py
9. Update the dvc.yaml  
10. Update app.py

# How to run?

### STEPS:

clone the repo

```bash
https://github.com/yhr0864/kidney-disease-classification-project.git
```

### STEP 01- create a conda env after opening the repo

```bash
conda create -n myenv python=3.9 -y
```

```bash
conda activate myenv
```

### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```

```bash
##### cmd
mlflow ui
```

### dagshub
[dagshub](https://dagshub.com/)

MLFLOW_TRACKING_URI=https://dagshub.com/yhr0864/kidney-disease-classification-project.mlflow \
MLFLOW_TRACKING_USERNAME=yhr0864 \
MLFLOW_TRACKING_PASSWORD=d9d7c85432b94cdcba7ac783932b5090561e756d \
python script.py

Run this to export as env variables:
```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/yhr0864/kidney-disease-classification-project.mlflow

export MLFLOW_TRACKING_USERNAME=yhr0864

export MLFLOW_TRACKING_PASSWORD=d9d7c85432b94cdcba7ac783932b5090561e756d
```