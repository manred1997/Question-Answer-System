from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from tokenizers import BertWordPieceTokenizer
import torch
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

torch.set_grad_enabled(False)
q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
q_encoder = DPRQuestionEncoder.from_pretrained("Retrieval/question_encoder").to(device=torch.device('cpu'))
q_encoder.eval()

# ctx_tokenizer = BertWordPieceTokenizer("ctx_tokenizer/vocab.txt", lowercase=True)
ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
ctx_encoder = DPRContextEncoder.from_pretrained("Retrieval/ctx_encoder").to(device=torch.device('cpu'))
ctx_encoder.eval()

def reranking(query, paragraphs):
    query_tokenizer = q_tokenizer(query, return_tensors="pt")
    query_embedding = q_encoder(**query_tokenizer)[0][0].numpy()

    paragraphs_embedding = []
    for i, para in enumerate(paragraphs):
        # print(i)
        para_tokenizer = ctx_tokenizer(para, return_tensors="pt")
        para_embedding = ctx_encoder(**para_tokenizer)[0][0].numpy()
        paragraphs_embedding.append(para_embedding)
    
    paragraphs_embedding = np.array(paragraphs_embedding)
    # print("========================")

    # print(paragraphs_embedding.shape)
    # print(query_embedding.shape)
    score = np.dot(paragraphs_embedding, query_embedding)
    # print(score)

    score_para = []
    for i in range(len(paragraphs)):
        score_para.append({"score": score[i],
                            "para": paragraphs[i]})
    stored_para = sorted(score_para, key= lambda x: x["score"],reverse=True)
    return stored_para[:5]

# query = "What is president of Germany?"
# paragraphs = ["The president of the United States (POTUS)[A] is the head of state and head of government of the United States of America. The president directs the executive branch of the federal government and is the commander-in-chief of the United States Armed Forces.",
# "Joe Biden is the 46th and current president of the United States, having assumed office on January 20, 2021.",
# "The Apollo program, also known as Project Apollo, was the third United States human spaceflight program carried out by the National Aeronautics and Space Administration (NASA), which accomplished landing the first humans on the Moon from 1969 to 1972. First conceived during Dwight D. Eisenhower's administration as a three-man spacecraft to follow the one-man Project Mercury which put the first Americans in space, Apollo was later dedicated to President John F. Kennedy's national goal of landing a man on the Moon and returning him safely to the Earth by the end of the 1960s, which he proposed in a May 25, 1961, address to Congress.",
# "Apollo used Saturn family rockets as launch vehicles. Apollo/Saturn vehicles were also used for an Apollo Applications Program, which consisted of Skylab, a space station that supported three manned missions in 1973-74, and the Apollo-Soyuz Test Project, a joint Earth orbit mission with the Soviet Union in 1975.",
# "Manchester United Football Club is a professional football club based in Old Trafford, Greater Manchester, England, that competes in the Premier League, the top flight of English football. Nicknamed, the club was founded as Newton Heath LYR Football Club in 1878, changed its name to Manchester United in 1902 and moved to its current stadium, Old Trafford, in 1910.",
# "The first manned flight of Apollo was in 1968. Apollo ran from 1961 to 1972, and was supported by the two man Gemini program which ran concurrently with it from 1962 to 1966. Gemini missions developed some of the space travel techniques that were necessary for the success of the Apollo missions",
# "The president of Germany, officially the Federal President of the Federal Republic of Germany (German: Bundespräsident der Bundesrepublik Deutschland),[2] is the head of state of Germany.",
# "The current officeholder is Frank-Walter Steinmeier who was elected on 12 February 2017 and started his first five-year term on 19 March 2017.",
# "Frank-Walter Steinmeier ( born 5 January 1956)[1] is a German politician serving as President of Germany since 19 March 2017.[2] He was Minister for Foreign Affairs from 2005 to 2009 and from 2013 to 2017, and Vice-Chancellor of Germany from 2007 to 2009. He was chairman-in-office of the Organization for Security and Co-operation in Europe (OSCE) in 2016.",
# "Following the 2005 federal election, Steinmeier became Foreign Minister in the first grand coalition government of Angela Merkel, and since 2007 he additionally held the office of vice chancellor. In 2008, he briefly served as acting chairman of his party. He was the SPD's candidate for chancellor in the 2009 federal election, but his party lost the election and he left the federal cabinet to become leader of the opposition. Following the 2013 federal election, he again became Minister for Foreign Affairs in Merkel's second grand coalition."]

# #relevant paragraphs is [0], [1]

# score, score_para = reranking(query, paragraphs)
# print(score)
# # print(score_para)