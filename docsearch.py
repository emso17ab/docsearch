import pandas as pd
import numpy as np
import json
import re

class Library:

    def __init__(self, docs=[]):
        self._collection = []
        self._index = []
        self._word_vectors = []

        with open('stop_words_english.json', encoding="utf8") as f:
            self.stopwords = json.load(f)

        self._preprocess(docs)
        self._updateBOW()
        self._vectorize()


    @property
    def collection(self):
        collection_columns = ["ID", "DOCUMENT"]
        return pd.DataFrame(self._collection, columns=collection_columns)


    @property
    def index(self):
        collection_columns = ["ID", "WORD_TOKEN"]
        return pd.DataFrame(self._index, columns=collection_columns)

    @property
    def word_vectors(self):
        collection_columns = ["ID", "VECTOR"]
        return pd.DataFrame(self._word_vectors, columns=collection_columns)

    def _updateBOW(self):
        '''
        Updates the bag of words from all documents present in the library.
        '''
        self.bow = []
        for entry in self._index:
            words = [word for word in entry[1] if word not in self.bow and word not in self.stopwords]
            self.bow += words


    def _preprocess(self, docs):
        collection = []
        index = []
        for i, doc in enumerate(docs):
            entry = self._tokenize(doc)
            index.append([i, entry])
            collection.append([i, doc])

        self._collection = collection
        self._index = index

    
    def _tokenize(self, doc):
        #Tokenizing using simple regex
        return [word.lower() for word in re.findall(r"[\w'-]+", doc)]


    def _vectorize(self):
        '''
        Assigns a vector for each doc in the library from the "Bag of words".
        '''
        update = []
        for entry in self._index:
            vector = self._index2vec(entry[1])
            update.append([entry[0], vector])
        self._word_vectors = update

        

    def _index2vec(self, index):
        vector = []
        for word in self.bow:
            if word in index:
                vector.append(index.count(word))
            else:
                vector.append(0)
        return vector


    def addDocuments(self, new_docs):
        if isinstance(new_docs, str):
            new_docs = [new_docs]
        docs = [doc[1] for doc in self._collection] + new_docs
        self._preprocess(docs)
        self._updateBOW()
        self._vectorize()


    def loadExampleCollection(self):
        data = []
        with open('quotes.json', encoding="utf-8") as f:
            data = json.load(f)
        data = [d['text'] for d in data]
        self.addDocuments(data)


    def search(self, doc, method=1, top=10):
        '''
        Returns a list of docs from the library that are most similar to the search
        using vector dot product (method=1) or Euclidean distance (method=2).
        '''
        if method not in [1,2]:
            raise Exception("You must specify either 1 or 2 as method")

        vector_A = self._index2vec(self._tokenize(doc))
        score_index = []
        for vector_B in self._word_vectors:
            if method == 1:
                score = np.dot(vector_A, vector_B[1])
            elif method == 2:
                score = -1 * np.linalg.norm(np.array(vector_A) - np.array(vector_B[1]))
            score_index.append([vector_B[0], score])

        scores_columns = ["ID", "SCORE"]
        scores = pd.DataFrame(score_index, columns=scores_columns)

        collection_columns = ["ID", "DOCUMENT"]
        collection = pd.DataFrame(self._collection, columns=collection_columns)

        result = collection.merge(scores, on="ID")
        result.sort_values(by=['SCORE'], inplace=True, ascending=False)

        return result.head(top)
