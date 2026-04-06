import math
from collections import Counter

class BM25():
    
    def __init__(self, documents, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = [doc.lower().split() for doc in documents]
        self.doc_count = len(self.docs)
        self.avg_dl = sum(len(d) for d in self.docs) / self.doc_count
        # Length normalization: Longer docs get penalized slightly so they don't dominate just by having more words

        self.df = {}
        for doc in self.docs:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1
        # Document frequency - how many documents contain each term

    def score(self, query, doc_index):
        query_terms = query.lower().split()
        doc = self.docs[doc_index]
        doc_len = len(doc)
        tf = Counter(doc)

        score = 0
        for term in query_terms:
            if term not in self.df:
                continue
            idf = math.log((self.doc_count - self.df[term] + 0.5) / (self.df[term] + 0.5) + 1)
            # IDF (Inverse Document Frequency) - rare words get higher socres.
            term_freq = tf.get(term, 0)
            numerator = term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_dl)
            score += idf * numerator / denominator

        return score 