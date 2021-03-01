import os
import re
import json
import sys
import argparse

import numpy as np
from colorama import Fore
from tokenizers import BertWordPieceTokenizer
from tqdm import tqdm
from transformers import ElectraTokenizer, ElectraForQuestionAnswering
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from Reader.Sample import Sample

model = ElectraForQuestionAnswering.from_pretrained("Reader/electra_QA").to(device=torch.device('cpu'))
model.load_state_dict(torch.load('Reader/weight_electra/weights_3.pth',map_location=torch.device('cpu')))
model.eval()
tokenizer = BertWordPieceTokenizer("Reader/electra_base_uncased/vocab.txt", lowercase=True)


def inference(question, paragraph):
    squad_eg = Sample(tokenizer, question, paragraph)
    squad_eg.preprocess()
    dataset_dict = {
        "input_word_ids": [],
        "input_type_ids": [],
        "input_mask": [],
        "start_token_idx": [],
        "end_token_idx": [],
    }

    if squad_eg.skip is False:
        for key in dataset_dict:
            dataset_dict[key].append(getattr(squad_eg, key))
    
    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])

    x = [dataset_dict["input_word_ids"], dataset_dict["input_mask"], dataset_dict["input_type_ids"]]

    input_word_ids = torch.tensor(x[0], dtype=torch.int64)
    input_mask = torch.tensor(x[1], dtype=torch.int64)
    input_type_ids = torch.tensor(x[2], dtype=torch.int64)

    with torch.no_grad():
        output_ = model(input_ids=input_word_ids,attention_mask=input_mask,token_type_ids=input_type_ids)
        start_logits, end_logits = output_.start_logits, output_.end_logits
        pred_start, pred_end = start_logits.detach().cpu().numpy(), end_logits.detach().cpu().numpy()
    
    offsets = squad_eg.context_token_to_char
    start = np.argmax(pred_start)
    end = np.argmax(pred_end)
    pred_ans = None
    if start >= len(offsets):
        pass
    pred_char_start = offsets[start][0]
    if end < len(offsets):
        pred_char_end = offsets[end][1]
        pred_ans = squad_eg.context[pred_char_start:pred_char_end]
    else:
        pred_ans = squad_eg.context[pred_char_start:]
    return squad_eg.question, pred_ans

# paragraph = "Under the Twenty-second Amendment, ratified in 1951, no person who has been elected to two presidential terms may be elected to a third. In addition, nine vice presidents have become president by virtue of a president's intra-term death or resignation. In all, 45 individuals have served 46 presidencies spanning 58 full four-year terms.Joe Biden is the 46th and current president of the United States, having assumed office on January 20, 2021.In July 1776, during the American Revolutionary War, the Thirteen Colonies, acting jointly through the Second Continental Congress, declared themselves to be 13 independent sovereign states, no longer under British rule. Recognizing the necessity of closely coordinating their efforts against the British, the Continental Congress simultaneously began the process of drafting a constitution that would bind the states together"
# question = "Who is president of USA?"

# _, answer = inference(question, paragraph)
# print(paragraph)
# print(len(paragraph))
# print("="*50)
# print(question)
# print(len(question))
# print("="*50)
# print(answer)

        
    