import numpy as np
import re
import json
import pandas as pd

white_space = re.compile('\s')
non_alpha = re.compile('([^A-Za-z\s\'])')

# Import articles from .csv. Return corpus as list of dictionaries.
def import_articles():
    data = pd.read_csv('articles_cleaned.csv')
    return data.to_dict(orient = 'records')

# Import quotes from .csv. Return corpus as list of dictionaries.
def import_quotes():
    with open('quotes.json', encoding="utf-8") as f:
            data = json.load(f)
    data = [{'from':d['from'], 'text':d['text']}for d in data]
    return data

# Remove non-alphabetical characters. Keep single quotes. Return cleaned string.
def cleanify(string):
    return re.sub(non_alpha, repl="", string=string)

# Input a given string with white spaces as separaptor. 
# Return a set of unique words.
def setify(string):
    listified = re.split(white_space, string)
    setified = set(listified)
    return setified

# Higher order function for our corpus-dictionary that manipulates the text key.
def map_dict(dict_e, func):
    dict_e['text'] = func(dict_e['text'])
    return dict_e

# Parses the corpus dictionary: using str.lower(), cleanify(), setify() in our higher-order function map_dict()
def parse_dict(dict_e):
    lowerfied = map_dict(dict_e, str.lower)
    cleanified = map_dict(lowerfied, cleanify)
    setified = map_dict(cleanified, setify)
    return setified

# Applies parse_dict() to all elements in a list of dictionaries.
def text_corpus_lib_parser(corpus_list):
    parsed = map(lambda x: parse_dict(x), corpus_list)
    return list(parsed)
    
# Sort the corpus list by length of text.
def sort_corpus_lib(corpus_lib):
    corpus_lib.sort(reverse=True, key=lambda x: len(x['text']))

# Vectorizes a list dictionaries set of text by comparing it to the text of the search document.
def vectorize(corpus_lib, parsed_search_doc):
    parsed_search_doc['text_vectorized'] = [1 for i in parsed_search_doc['text']]
    for e in corpus_lib:
        e['text_vectorized'] = []
        for value in parsed_search_doc['text']:
            e['text_vectorized'].append(int(bool({value} & e['text'])))
    return corpus_lib, parsed_search_doc

# Calculates the dot product of two vectors in a list form. 
def dot_product(vector1, vector2):
    if len(vector1) == len(vector2):
        return sum(map(lambda x,y: x*y, vector1, vector2))
    else:
        raise Exception("Vectors does not have same length.")

# Calculates the euclidean_distance of two vectors in a list form. 
def euclidean_distance(vector1, vector2):
    if len(vector1) == len(vector2):
        return sum(map(lambda x,y: (x-y)**2, vector1, vector2))**0.5

# Calculates the dot products in all elements in the list and stores the value as in a new key for each dictionary. 
def similarity_dot_product(vectorized_corpus_lib, parsed_search_doc):
    for e in vectorized_corpus_lib: 
        e['similarity_score_dot_product'] = dot_product(e['text_vectorized'], parsed_search_doc['text_vectorized'])
    return vectorized_corpus_lib

# Calculates the euclidean distance in all elements in the list and stores the value as in a new key for each dictionary. 
def similarity_euclidean_distance(vectorized_corpus_lib, parsed_search_doc):
    for e in vectorized_corpus_lib: 
        e['similarity_score_euclidean_distance'] = euclidean_distance(e['text_vectorized'], parsed_search_doc['text_vectorized'])
    return vectorized_corpus_lib

def run(data = import_articles(), search_text = "An entrepreneur tends to bite off a little more than he can chew hoping he’ll quickly learn how to chew it yolo a to Amir"):
    corpus_lib = text_corpus_lib_parser(data)
    search_doc = {"from": "search", "text": search_text}
    parsed_search_doc = parse_dict(search_doc)

    vectorized_corpus_lib, parsed_search_doc = vectorize(corpus_lib, parsed_search_doc)

    vectorized_corpus_lib = similarity_dot_product(vectorized_corpus_lib, parsed_search_doc)
    vectorized_corpus_lib = similarity_euclidean_distance(vectorized_corpus_lib, parsed_search_doc)

    df_dot_product = pd.DataFrame(vectorized_corpus_lib) 
    print(df_dot_product.sort_values('similarity_score_euclidean_distance', ascending=True))

    df_dot_product.to_csv('doc_search_data_avl.csv', index=False)

run()