from googleapiclient.discovery import build
import requests
from bs4 import BeautifulSoup
import re

API_KEY = ['AIzaSyAQCArO_mFnYXSX5KyuSJ9AZohGXc1--vQ']
Custom_Search_Engine_ID = "f941f0bc1110e722b"

regular_word = "[^a-z0-9A-Z_ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹ\s\'\",.?!()\[\]=$%€£-]"

def preprocess(paragraphs):
    remove_element = [p for p in paragraphs if len(p) < 10]
    for i in remove_element:
        paragraphs.remove(i)
    context = ".".join(paragraphs)
    context = re.sub("(\\n)", "", context)
    context = re.sub(regular_word, " ", context)
    # context = re.sub("\[\d+\]", "", context)
    # context = re.sub("\[\D+\]", "", context)
    # context = re.sub("\(\d+\)", "", context)
    # context = re.sub("\(\D+\)", "", context)
    context = re.sub("\s\s+", "", context)
    context = re.sub("\(\w\)", "", context)
    context = re.sub("\(\d\)", "", context)
    context = re.sub(r"\\\W", "\'", context)
    context = context.replace("..", ".")
    # print(context)
    return context


def googlesearch(search_term, api_key, cse_id, **kwargs):
        service = build("customsearch", "v1", developerKey=api_key)
        res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
        return res['items']

def getcontent(url):
    html = requests.get(url, timeout = 10)
    # print(html)
    tree = BeautifulSoup(html.text, 'lxml')

    paragraphs = [p.get_text() for p in tree.find_all("p")]
    content = preprocess(paragraphs)
    return content


class GoogleSearch():

    def search(self, question):
        pages_content  = googlesearch(question, API_KEY[0], Custom_Search_Engine_ID, num=1)
        document_urls = set([])
        for page in pages_content:
            document_urls.add(page['link'])
        document_urls = list(document_urls)
        paragraphs = []
        for url in document_urls:
            contents = getcontent(url).split(".")
            for i, _ in enumerate(contents, 1):
                if (i%8) == 0:
                    paragraphs.append(".".join(contents[i-8:i]))
        paragraphs = [p for p in paragraphs if len(p) > 20]
        # paragraphs = [p for p in paragraphs if len(p) < 2000]
        return paragraphs, document_urls