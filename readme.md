# Medical Knowledge Question Answer System


## Introduction
This module is a medical knowledge question answering system based on knowledge graph built on Neo4j. It converts query in natural language to query in cypher, then get structured answer from the knowledge graph and reply in natural language.  


## requirements
- JDK11 (dependency of Neo4j)
- Neo4j
- py2neo==2021.2.3
- scikit_learn==1.0.2
- torch==1.11.0
- tqdm==4.62.2
- transformers==4.10.0

## How to use

#### 0. Download data/checkpoint
Due to size limit, data and checkpoint cannot be uploaded to github. Thus, we will provide the google drive link for you to download. Download the data and ckeckpoint folder, place it in the right place. 
data: https://drive.google.com/file/d/1pfOR1sC_eqCaoq6XlpJ0w8OUbVBMlGkV/view?usp=sharing
checkpoint: https://drive.google.com/file/d/10Fo0x7F_Xe3SHJjZJgKojJs5w3jwplAF/view?usp=sharing
```
.
├── __pycache__
│   ├── config.cpython-38.pyc
│   └── train_model.cpython-38.pyc
├── build_graph.py
├── ckpt
│   └── chinese-roberta-wwm-ext.ckpt
├── config.py
├── data
│   ├── labels.txt
│   ├── medical.json
│   ├── test_data.json
│   ├── train_data.json
│   └── zhdd_lines.txt
├── dict
│   ├── check.txt
│   ├── deny.txt
│   ├── department.txt
│   ├── disease.txt
│   ├── drug.txt
│   ├── food.txt
│   ├── producer.txt
│   └── symptom.txt
├── main.py
├── readme.md
├── requirements.txt
├── train_data_generate.py
└── train_model.py 
```
#### 1. Build graph 
Installing **Neo4j** and **Py2neo**. Adding the username and password of the neo4j database to config.py. Run build_graph.py to build knowledge graph in Neo4j. This script also extracts all the entities and save them in the *dict* folder.   

#### 2. Train data generation
Run *train_data_generate.py* to generate training data for the deep learning model. The generated data will be saved to *data* folder.<br>This package already provides generated data. So this step can be skipped. 

#### 3. Train model 
Run train_model.py to fine-tune a Chinese-roberta model for both intent classification and slot filling. The checkpoint will be saved to *ckpt* folder.<br>
This package already provides a fine-tuned ckpt *chinese-roberta-wwm-ext.ckpt*. So this step can be skipped. 

#### 4. Run the service
Run main.py to start the server. Type in the question to get in answer. Type q to quit. 


