from Reader.Sample import Sample
import torch
from reranking import get_passage
import numpy as np

def inference(question, paragraph, model, tokenizer):
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
        pred_char_end = offsets[end][1]+1
        pred_ans = squad_eg.context[pred_char_start:pred_char_end]
    else:
        pred_ans = squad_eg.context[pred_char_start:]
    return squad_eg.context[:pred_char_start], pred_ans, squad_eg.context[pred_char_end:]

def get_answer(question, model, tokenizer, q_tokenizer, q_encoder, ctx_tokenizer, ctx_encoder):
    passages = get_passage(question, q_tokenizer, q_encoder, ctx_tokenizer, ctx_encoder)
    # print(len(passages))
    answers = []
    for i in passages:
        answers.append({"url": i["url"], "answer": inference(question, i["para"], model, tokenizer)})
    return answers