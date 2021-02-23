import numpy as np
import tensorflow as tf

class MMR(object):
    
    def get_query(self, embeddings):
        query = np.mean(embeddings, axis=0)
        return query
    
    def similarity(self, s1, s2):
        cossim = np.dot(s1, s2) / (np.linalg.norm(s1) * np.linalg.norm(s2))
        sim = 1 - np.arccos(cossim) / np.pi
        return sim
    
    def similarity_with_matrix(self, s, m):
        cossim = np.dot(m, s) / (np.linalg.norm(s) * np.linalg.norm(m, axis=1))
        sim = 1 - np.arccos(cossim) / np.pi
        return sim
    
    def mmr_score(self, s, q, selected, _lambda):
        relevance = _lambda * self.similarity(s, q)
        if selected.shape[0] > 0:
            diversity = (1 - _lambda) * np.max(self.similarity_with_matrix(s, selected))
        else:
            diversity = 0
        return relevance - diversity
            
        
    def summarize(self, embeddings, max_len=3, _lambda=0.5):
        query = self.get_query(embeddings)
        
        selected = [False] * embeddings.shape[0]
        while sum(selected) < max_len:
            remains = [idx for idx, i in enumerate(selected) if not i]
            mmr = [self.mmr_score(embeddings[i], query, embeddings[selected], _lambda) for i in remains]
            best = np.argsort(mmr)[-1]
            selected[best] = True
        
        return selected