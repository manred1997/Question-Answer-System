from search_ggapi import GoogleSearch
import numpy as np

def search(query):
    ggsearch = GoogleSearch()
    paragraphs, urls = ggsearch.search(query)
    # print(len(paragraphs))
    paragraphs_urls=[]

    for i in range(len(paragraphs)):
        paragraphs_urls.append({"p":paragraphs[i],'url':urls[0]})
    return paragraphs_urls

def reranking(query, paragraphs_urls, q_tokenizer, q_encoder, ctx_tokenizer, ctx_encoder):
    # print(query)
    query_tokenizer = q_tokenizer(query, return_tensors="pt")
    # print(query_tokenizer)
    # print(q_encoder(**query_tokenizer)[0])
    query_embedding = q_encoder(**query_tokenizer)[0][0].detach().numpy()

    paragraphs_embedding = []
    for i, para in enumerate(paragraphs_urls):
        # print(i)
        para_tokenizer = ctx_tokenizer(para['p'], return_tensors="pt")
        para_embedding = ctx_encoder(**para_tokenizer)[0][0].detach().numpy()
        paragraphs_embedding.append(para_embedding)
    
    paragraphs_embedding = np.array(paragraphs_embedding)
    # print("========================")

    # print(paragraphs_embedding.shape)
    # print(query_embedding.shape)
    score = np.dot(paragraphs_embedding, query_embedding)
    # print(score)

    score_para = []
    for i in range(len(paragraphs_urls)):
        score_para.append({"score": score[i],
                            "para": paragraphs_urls[i]['p'],
                            'url': paragraphs_urls[i]['url']})
    stored_para = sorted(score_para, key= lambda x: x["score"],reverse=True)
    return stored_para[:5]

def get_passage(query, q_tokenizer, q_encoder, ctx_tokenizer, ctx_encoder):
    paragraphs_urls = search(query)
    stored_para = reranking(query, paragraphs_urls, q_tokenizer, q_encoder, ctx_tokenizer, ctx_encoder)
    return stored_para