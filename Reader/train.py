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

from Sample import Sample


def create_squad_examples(raw_data, desc, dir_tokenizer):
    tokenizer = BertWordPieceTokenizer(os.path.join(dir_tokenizer,"vocab.txt"), lowercase=True)
    p_bar = tqdm(total=len(raw_data["data"]), desc=desc,
                 position=0, leave=True,
                 file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET))
    squad_examples = []
    for item in raw_data["data"]:
        for para in item["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                question = qa["question"]
                if "answers" in qa:
                    answer_text = qa["answers"][0]["text"]
                    start_char_idx = qa["answers"][0]["answer_start"]
                    all_answers = [_["text"] for _ in qa["answers"]]
                    squad_eg = Sample(tokenizer, question, context, start_char_idx, answer_text, all_answers)
                else:
                    squad_eg = Sample(tokenizer, question, context)
                squad_eg.preprocess()
                squad_examples.append(squad_eg)
        p_bar.update(1)
    p_bar.close()
    return squad_examples

def create_inputs_targets(squad_examples):
    dataset_dict = {
        "input_word_ids": [],
        "input_type_ids": [],
        "input_mask": [],
        "start_token_idx": [],
        "end_token_idx": [],
    }
    for item in squad_examples:
        if item.skip is False:
            for key in dataset_dict:
                dataset_dict[key].append(getattr(item, key))
    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])
    x = [dataset_dict["input_word_ids"], dataset_dict["input_mask"], dataset_dict["input_type_ids"]]
    y = [dataset_dict["start_token_idx"], dataset_dict["end_token_idx"]]
    return x, y


def normalize_text(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    text = re.sub(regex, " ", text)
    text = " ".join(text.split())
    return text

def load_data(dir_data):
    with open(os.path.join(dir_data, 'train.json')) as f:
        raw_train_data = json.load(f)
    with open(os.path.join(dir_data, 'eval.json')) as f:
        raw_eval_data = json.load(f)
    return raw_train_data,raw_eval_data

def train_model(dir_tokenizer: str = None, dir_model: str = None, dir_data: str = None):
    batch_size = 16
    epochs = 10

    raw_train_data, raw_eval_data = load_data(dir_data)
    train_squad_examples = create_squad_examples(raw_train_data, "Creating training points", dir_tokenizer)
    x_train, y_train = create_inputs_targets(train_squad_examples)

    eval_squad_examples = create_squad_examples(raw_eval_data, "Creating evaluation points",dir_tokenizer)
    x_eval, y_eval = create_inputs_targets(eval_squad_examples)

    train_data = TensorDataset(torch.tensor(x_train[0], dtype=torch.int64),
                            torch.tensor(x_train[1], dtype=torch.float),
                            torch.tensor(x_train[2], dtype=torch.int64),
                            torch.tensor(y_train[0], dtype=torch.int64),
                            torch.tensor(y_train[1], dtype=torch.int64))
    print(f"{len(train_data)} training points created.")
    train_sampler = RandomSampler(train_data)
    train_data_loader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    eval_data = TensorDataset(torch.tensor(x_eval[0], dtype=torch.int64),
                            torch.tensor(x_eval[1], dtype=torch.float),
                            torch.tensor(x_eval[2], dtype=torch.int64),
                            torch.tensor(y_eval[0], dtype=torch.int64),
                            torch.tensor(y_eval[1], dtype=torch.int64))
    print(f"{len(eval_data)} evaluation points created.")
    eval_sampler = SequentialSampler(eval_data)
    validation_data_loader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

    model = ElectraForQuestionAnswering.from_pretrained(dir_model)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}
    ]
    optimizer = torch.optim.Adam(lr=1e-5, betas=(0.9, 0.98), eps=1e-9, params=optimizer_grouped_parameters)
    for epoch in range(1, epochs + 1):
        # ============================================ TRAINING ============================================================
        print("Training epoch ", str(epoch))
        training_pbar = tqdm(total=len(train_data),
                            position=0, leave=True,
                            file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET))
        model.train()
        tr_loss = 0
        nb_tr_steps = 0
        for step, batch in enumerate(train_data_loader):
            batch = tuple(t for t in batch)
            input_word_ids, input_mask, input_type_ids, start_token_idx, end_token_idx = batch
            optimizer.zero_grad()
            output = model(input_ids=input_word_ids,
                            attention_mask=input_mask,
                            token_type_ids=input_type_ids,
                            start_positions=start_token_idx,
                            end_positions=end_token_idx)
            # print(loss)
            loss = output[0]
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            nb_tr_steps += 1
            training_pbar.update(input_word_ids.size(0))
        training_pbar.close()
        print(f"\nTraining loss={tr_loss / nb_tr_steps:.4f}")
        torch.save(model.state_dict(), "./weights_" + str(epoch) + ".pth")
        # ============================================ VALIDATION ==========================================================
        validation_pbar = tqdm(total=len(eval_data),
                            position=0, leave=True,
                            file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET))
        model.eval()
        eval_examples_no_skip = [_ for _ in eval_squad_examples if _.skip is False]
        currentIdx = 0
        count = 0
        for batch in validation_data_loader:
            batch = tuple(t for t in batch)
            input_word_ids, input_mask, input_type_ids, start_token_idx, end_token_idx = batch
            with torch.no_grad():
                output_ = model(input_ids=input_word_ids,
                                                attention_mask=input_mask,
                                                token_type_ids=input_type_ids)
                # print(output_.start_logits)
                start_logits, end_logits = output_.start_logits, output_.end_logits
                pred_start, pred_end = start_logits.detach().cpu().numpy(), end_logits.detach().cpu().numpy()

            for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
                squad_eg = eval_examples_no_skip[currentIdx]
                currentIdx += 1
                offsets = squad_eg.context_token_to_char
                start = np.argmax(start)
                end = np.argmax(end)
                if start >= len(offsets):
                    continue
                pred_char_start = offsets[start][0]
                if end < len(offsets):
                    pred_char_end = offsets[end][1]
                    pred_ans = squad_eg.context[pred_char_start:pred_char_end]
                else:
                    pred_ans = squad_eg.context[pred_char_start:]
                normalized_pred_ans = normalize_text(pred_ans)
                normalized_true_ans = [normalize_text(_) for _ in squad_eg.all_answers]
                if normalized_pred_ans in normalized_true_ans:
                    count += 1
            validation_pbar.update(input_word_ids.size(0))
        acc = count / len(y_eval[0])
        validation_pbar.close()
        print(f"\nEpoch={epoch}, exact match score={acc:.2f}")
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dir_tokenizer",
        default="electra_base_uncased",
        type=str,
        help="The directory tokenizer pretrained",
    )
    parser.add_argument(
        "--dir_model",
        default="electra_QA",
        type=str,
        help="The directory model pretrain",
    )
    parser.add_argument(
        "--dir_data",
        default="../SquadV2_data",
        type=str,
        help="The directory file data",
    )
    args = parser.parse_args()
    print(args)
    train_model(args.dir_tokenizer, args.dir_model, args.dir_data)