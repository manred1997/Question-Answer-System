from search_ggapi import GoogleSearch
from Reader.inference import inference
from Retrieval.reranking import reranking

ggsearch = GoogleSearch()
query = "when was alan turing born?"
paragraphs, urls = ggsearch.search(query)
# print(len(paragraphs))
print(urls)

score_paragraphs = reranking(query, paragraphs)
for i in score_paragraphs:
    _, answer = inference(query, i["para"])
    print(answer)