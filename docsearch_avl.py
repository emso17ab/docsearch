import numpy as np
import re
import json
import pandas as pd

white_space = re.compile('\s')
non_alpha = re.compile('([^A-Za-z\s\'])')

list_of_documents = []

new_document = ""

def import_articles():
    data = pd.read_csv('articles_cleaned.csv')
    return data.to_dict(orient = 'records')

def load_example_documents():
    with open('quotes.json', encoding="utf-8") as f:
            data = json.load(f)
    data = [{'from':d['from'], 'text':d['text']}for d in data]
    return data

def cleanify(string):
    return re.sub(non_alpha, repl="", string=string)

def setify(string):
    listified = re.split(white_space, string)
    setified = set(listified)
    return setified

def map_dict(dict_e, func):
    dict_e['text'] = func(dict_e['text'])
    return dict_e

def parse_dict(dict_e):
    lowerfied = map_dict(dict_e, str.lower)
    cleanified = map_dict(lowerfied, cleanify)
    setified = map_dict(cleanified, setify)
    return setified

def text_corpus_lib_parser(list_of_documents):
    parsed = map(lambda x: parse_dict(x), list_of_documents)
    return list(parsed)
    
def sort_corpus_lib(corpus_lib):
    corpus_lib.sort(reverse=True, key=lambda x: len(x['text']))

def vectorize(sorted_corpus_lib, parsed_search_doc):
    parsed_search_doc['text_vectorized'] = [1 for i in parsed_search_doc['text']]

    for i,e in enumerate(corpus_lib):
        e['text_vectorized'] = []
        for value in parsed_search_doc['text']:
            e['text_vectorized'].append(int(bool({value} & e['text'])))
    return corpus_lib, parsed_search_doc

def dot_product(vector1, vector2):
    if len(vector1) == len(vector2):
        return sum(map(lambda x,y: x*y, vector1, vector2))
    else:
        raise Exception("Vectors does not have same length.")

corpus_lib = text_corpus_lib_parser(import_articles())
sorted_corpus_lib = sort_corpus_lib(corpus_lib)

search_doc = {"from": "search", "text": "An entrepreneur tends to bite off a little more than he can chew hoping he’ll quickly learn how to chew it"}
parsed_search_doc = parse_dict(search_doc)

vectorized_corpus_lib, parsed_search_doc = vectorize(sorted_corpus_lib, parsed_search_doc)

def similarity_dot_product(vectorized_corpus_lib, parsed_search_doc):
    for e in vectorized_corpus_lib: 
        e['similarity_score'] = dot_product(e['text_vectorized'], parsed_search_doc['text_vectorized'])
    return vectorized_corpus_lib

df_dot_product = pd.DataFrame(similarity_dot_product(vectorized_corpus_lib, parsed_search_doc)) 
print(df_dot_product.sort_values('similarity_score', ascending=False))