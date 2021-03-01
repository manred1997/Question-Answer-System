from flask import Flask, request, render_template
import torch
from transformers import ElectraTokenizer, ElectraForQuestionAnswering
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from tokenizers import BertWordPieceTokenizer
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from reader import get_answer

model = ElectraForQuestionAnswering.from_pretrained("Reader/electra_QA").to(device=torch.device('cpu'))
model.load_state_dict(torch.load('Reader/weight_electra/weights_3.pth',map_location=torch.device('cpu')))
model.eval()
tokenizer = BertWordPieceTokenizer("Reader/electra_base_uncased/vocab.txt", lowercase=True)

torch.set_grad_enabled(False)
q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
q_encoder = DPRQuestionEncoder.from_pretrained("Retrieval/question_encoder").to(device=torch.device('cpu'))
q_encoder.eval()

# ctx_tokenizer = BertWordPieceTokenizer("ctx_tokenizer/vocab.txt", lowercase=True)
ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
ctx_encoder = DPRContextEncoder.from_pretrained("Retrieval/ctx_encoder").to(device=torch.device('cpu'))
ctx_encoder.eval()

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/', methods=['POST'])
def Answering():
    question = request.form['question']   
    answers = get_answer(question, model, tokenizer, q_tokenizer, q_encoder, ctx_tokenizer, ctx_encoder)
    q_a_s_s = [i['answer']+(i['url'],) for i in answers]
    return render_template('home.html', q= question, q_a_s_s= q_a_s_s)

if __name__ == "__main__":
    app.run(debug=True)