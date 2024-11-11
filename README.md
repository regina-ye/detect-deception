# Deceptive Text Classification with Transformer, Gradient Boosting Model, and Ensemble

This repository contains 3 machine learning models to detect deception in the [DIFrauD dataset](https://huggingface.co/datasets/redasers/difraud). The code processes the data, trains the models, tests them, and outputs the results.

## Repository Structure
- `process_data.py`: prepares dataset for training
- `all_models.py`: defines Transformers and Ensemble models
- `train_and_evaluate.py`: training, evaluation, and result generation
- `requirements.txt`: required Python packages

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/regina-ye/detect-deception.git
```

### 2. Install Dependencies

Install the required packages:

```bash
conda create -n <your environment name>
conda activate <your environment name>
pip install -r requirements.txt
```

### 3. Process the Data
Run the process_data.py script to preprocess and prepare the data for model training:

```bash
python process_data.py
```

### 4. Train and Evaluate Models

Due to the limited time I had to work on this project, I was not able to overcome the use of two different versions of the 'datasets' package. Therefore, you need to upgrade it to the latest version before training. Run this:

```bash
pip install --upgrade datasets
```

Run the train_and_evaluate.py file to train the models, evaluate their performance, and print results:

```bash
python train_and_evaluate.py > results.txt
```

### Author
Regina Ye
# detect-deception
