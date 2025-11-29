# End_To_End Machine Learning Project

setup.sh
bash setup.sh


conda create -n ml python=3.12
conda config --set auto_activate_base false

conda activate ml
conda deactivate
deactivate  # Linux/Mac


sudo apt install python3.12
python3.12 -m venv ml
source ml/bin/activate 
deactivate
rm -rf ml



pip install -r requirements.txt 
pip list


# Workflow 
project/
├── app.py
├── requirements.txt
├── notex.txt
├── setup.py
├── cloud/
│   └── __init__.py
├── components/
│   └── __init__.py
├── constants/
│   └── __init__.py
├── entity/
│   └── __init__.py
├── exception/
│   └── __init__.py
├── logger/
│   └── __init__.py
├── pipeline/
│   └── __init__.py
└── utils/
    └── __init__.py




# Data_Ingestion
data_ingestion/
│
├── feature/
│   └── feature_data.csv
│
└── ingested/
    ├── train.csv
    └── test.csv



