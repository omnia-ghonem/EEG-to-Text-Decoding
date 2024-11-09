import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import json
import matplotlib.pyplot as plt
from glob import glob

from transformers import BartTokenizer, BartForConditionalGeneration
from data_raw import ZuCo_dataset
from model_decoding_raw import BrainTranslator
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
from config import get_config

from torch.nn.utils.rnn import pad_sequence


from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
secret_value = os.environ.get("env_var")


if __name__ == '__main__': 

    print(secret_value)
