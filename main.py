from tqdm import tqdm
import json
from pathlib import Path
import os

import pandas as pd
import numpy as np
import torch
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split

from vocab import create_vocab
from util import tokens_to_ids, greedy_tokenize
from datetime import datetime

# Control Parameters
GENERATE_TOKENIZED_DATA = True
TARGET_VOCAB_SIZE = 5000
DATASET_PATH = "medical_meadow_medical_flashcards/medical_meadow_wikidoc_medical_flashcards.json"
RANDOM_SEED = None

# Constants
VOCAB_PATH = "vocab.json"
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"
DATA_PATH = "data.csv"
INSTRUCTION = "Answer this question truthfully"

if GENERATE_TOKENIZED_DATA:
    # Load Dataset
    dataset = pd.read_json(Path(RAW_DATA_PATH) / DATASET_PATH)

    ## Clean Dataset
    # Remove empty lines
    dataset.replace("", np.nan, inplace=True)
    dataset.dropna(inplace=True)
    # Drop useless columns
    dataset.drop(["instruction"], axis=1)

    # Split dataset
    # Use train_test_split twice to split into train, val, and test
    # end proportions: 80% train, 10% test, 10% val
    train_df, temp_df = train_test_split(dataset, train_size=0.8, random_state=RANDOM_SEED, shuffle=True)
    test_df, val_df = train_test_split(temp_df, train_size=0.5, random_state=RANDOM_SEED, shuffle=True)

    # Generate Vocabulary
    create_vocab(train_df, target_vocab_size=TARGET_VOCAB_SIZE, vocab_output_path=Path(PROCESSED_DATA_PATH) / VOCAB_PATH)

    try:
        with open(Path(PROCESSED_DATA_PATH) / VOCAB_PATH, 'r') as f:
            vocab = json.load(f)
    except Exception as e:
        raise Exception(f"Failed loading vocab: {e}")


    # enable tqdm for pandas
    tqdm.pandas()

    start_time = datetime.now()

    for split_name, split in zip(['train', 'val', 'test'], [train_df, val_df, test_df]):

        # Tokenize data using vocabulary
        print(f"Tokenizing inputs for {split_name} split")
        split["tokenized_input"] = split["input"].progress_apply(lambda text: greedy_tokenize(text=text, vocab=vocab))
        print(f"Tokenizing outputs for {split_name} split")
        split["tokenized_output"] = split["output"].progress_apply(lambda text: greedy_tokenize(text=text, vocab=vocab))

        # convert tokenized-input to token ids
        print(f"Converting tokenized input to ids for {split_name} split")
        split["input_ids"] = split["tokenized_input"].progress_apply(lambda tokens: tokens_to_ids(tokens=tokens, vocab=vocab))
        print(f"Converting tokenized output to ids for {split_name} split")
        split["output_ids"] = split["tokenized_output"].progress_apply(lambda tokens: tokens_to_ids(tokens=tokens, vocab=vocab))

        # Save processed split data
        os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
        split.to_csv(Path(PROCESSED_DATA_PATH) / DATA_PATH)

    end_time = datetime.now()
    print("Total processing time: ", end_time-start_time)

    # # Save data for future runs
    # os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    # dataset.to_csv(Path(PROCESSED_DATA_PATH) / DATA_PATH)
else:
    # Load Vocab
    try:
        with open(Path(PROCESSED_DATA_PATH) / VOCAB_PATH, 'r') as f:
            vocab = json.load(f)
    except Exception as e:
        raise Exception(f"Failed loading vocab: {e}")
    
    # Load pre processed train, val, and test data splits
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
