from tqdm import tqdm
import json
from pathlib import Path
import os

import pandas as pd
import numpy as np
import torch
from torch.utils.data import random_split

from vocab import create_vocab
from util import tokens_to_ids, greedy_tokenize
from datetime import datetime

# Control Parameters
GENERATE_TOKENIZED_DATA = False
GENERATE_VOCAB = False
TARGET_VOCAB_SIZE = 5000
DATASET_PATH = "medical_meadow_medical_flashcards/medical_meadow_wikidoc_medical_flashcards.json"
DATASET_SPLIT_SEED = None

# Constants
VOCAB_PATH = "vocab.json"
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"
DATA_PATH = "data.csv"

# Load Dataset
print(Path(RAW_DATA_PATH) / DATASET_PATH)
dataset = pd.read_json(Path(RAW_DATA_PATH) / DATASET_PATH)

# Clean Dataset
dataset = dataset.replace("", np.nan, inplace=True)
dataset.dropna(inplace=True)
print(dataset.shape)
#33955

# Split dataset
if DATASET_SPLIT_SEED:
    generator = torch.Generator().manual_seed()
else:
    generator = torch.Generator()
train, val, test = random_split(dataset=dataset, lengths=[0.8, 0.1, 0.1], generator=generator)

# Load Vocabulary
if GENERATE_VOCAB:
    create_vocab(train, target_vocab_size=TARGET_VOCAB_SIZE, vocab_path=VOCAB_PATH)

try:
    with open(VOCAB_PATH, 'r') as f:
        vocab = json.load(f)
except Exception as e:
    raise Exception(f"Failed loading vocab: {e}")


if GENERATE_TOKENIZED_DATA:
    # enable tqdm for pandas
    tqdm.pandas()

    start_time = datetime.now()

    # Tokenize data using vocabulary
    print("Tokenizing Inputs")
    dataset["tokenized_input"] = dataset["input"].progress_apply(lambda text: greedy_tokenize(text=text, vocab=vocab))
    print("Tokenizing Outputs")
    dataset["tokenized_output"] = dataset["output"].progress_apply(lambda text: greedy_tokenize(text=text, vocab=vocab))

    # convert tokenized-input to token ids
    print("Converting tokenized input to ids")
    dataset["input_ids"] = dataset["tokenized_input"].progress_apply(lambda tokens: tokens_to_ids(tokens=tokens, vocab=vocab))
    print("Converting tokenized output to ids")
    dataset["output_ids"] = dataset["tokenized_output"].progress_apply(lambda tokens: tokens_to_ids(tokens=tokens, vocab=vocab))

    end_time = datetime.now()
    print("Total processing time: ", end_time-start_time)

    # Save data for future runs
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    dataset.to_csv(Path(PROCESSED_DATA_PATH) / DATA_PATH)
else:
    try: 
        # Read dataset from csv, data should have been previously generated
        dataset = pd.read_csv(Path(PROCESSED_DATA_PATH) / DATA_PATH)
    
    except Exception as e:
        raise Exception(f"Error loading dataset: ", e)
    
print(dataset[["input", "input_ids"]])
print(dataset[["output", "output_ids"]])



# class MyModel(torch.nn.Module):
#     def __init__(self, *args, **kwargs):
#         super(MyModel, self).__init__(*args, **kwargs)
        
#         self.linear1 = torch.nn.Linear(100,200)
#         self.activation = torch.nn.ReLU()
#         self.linear2 = torch.nn.Linear(200,10)
#         self.softmax = torch.nn.Softmax()

#     def forward(self, x):
#         x = self.linear1(x)
#         x = self.activation(x)
#         x = self.linear2(x)
#         x = self.softmax(x)
#         return x

# mymodel = MyModel()

# print("The model:")
# print(mymodel)

# print("\n\nOne Layer")
# print(mymodel.linear2)

# print("\n\nLayer Params")
# for param in mymodel.linear2.parameters():
#     print(param)
