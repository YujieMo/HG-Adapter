# SelHG-Adapter: Improving Pre-Trained Heterogeneous Graph Neural Networks with Dual Adapters 

This repository contains the reference code for the manuscript ``HG-Adapter: Improving Pre-Trained Heterogeneous Graph Neural Networks with Dual Adapters" 

## Contents

0. [Installation](#installation)
0. [Preparation](#Preparation)
0. [Training](#train)


## Installation
* pip install -r requirements.txt 
* Unzip the datasets (can be found in ./dataset/)
* Unzip the pre-trained embedding (can be found in ./pre_trained_embedding)

## Preparation
Important args:
* `--upload_pa` Test checkpoints to reproduce the results 
* `--dataset` Heterogeneous graph: ACM, Yelp, DBLP, Aminer 
* `--custom_key` Node: node classification

## Training
python main.py


