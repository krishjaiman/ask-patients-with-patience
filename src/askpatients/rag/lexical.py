"""Small-corpus BM25 retrieval without downloads; lexical, not semantic RAG."""
from collections import Counter
import math
import re

from .chunking import chunk_text

_STOPWORDS = set('a an and are as at be been but by do for from had has have i in is it its me my of on or that the their them these they this to was were with you your ve lately'.split())

def tokens(text):
    return [t for t in re.findall(r'[a-z0-9]+|[\u4e00-\u9fff]', text.lower()) if t not in _STOPWORDS]


class LexicalRetriever:
    def __init__(self, kb, chunk_size=400, overlap=60):
        self.chunks = [c for e in kb.entries for c in chunk_text(
            e.disease, e.as_grounding_text(), chunk_size, overlap)]
        self.counts = [Counter(tokens(c.text)) for c in self.chunks]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.average = sum(self.lengths) / max(1, len(self.lengths)) or 1
        self.df = Counter(t for c in self.counts for t in c)

    def query(self, text, top_k=5):
        scores = []
        for chunk, counts, length in zip(self.chunks, self.counts, self.lengths):
            score = 0.0
            for token in set(tokens(text)):
                freq = counts[token]
                if freq:
                    idf = math.log(1 + (len(self.chunks) - self.df[token] + .5) / (self.df[token] + .5))
                    score += idf * freq * 2.5 / (freq + 1.5 * (.25 + .75 * length / self.average))
            scores.append((chunk, score))
        return sorted(scores, key=lambda item: -item[1])[:top_k]

    def query_disease_scoped(self, text, disease, top_k=3):
        return [pair for pair in self.query(text, len(self.chunks)) if pair[0].disease == disease][:top_k]
