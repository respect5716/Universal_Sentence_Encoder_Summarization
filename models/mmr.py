import numpy as np
import tensorflow as tf
from typing import List

class MMR(object):
    def __init__(self, sentence_encoder):
        self.sentence_encoder = sentence_encoder
    
    def get_similarity(self, s1, s2):
        """ Get cosine similarity between vectors

        Params:
        s1 (np.array): 1d sentence embedding (512,)
        s2 (np.array): 1d sentence embedding (512,)
        
        Returns:
        sim (float): cosine similarity 
        """

        cossim = np.dot(s1, s2) / (np.linalg.norm(s1) * np.linalg.norm(s2))
        sim = 1 - np.arccos(cossim) / np.pi
        return sim
    
    def get_similarity_with_matrix(self, s, m):
        """Get cosine similarity between vector and matrix

        Params:
        s (np.array): 1d sentence embedding (512,)
        m (np.array): 2d sentences' embedding (n, 512)

        Returns:
        sim (np.array): similarity (n,)
        """

        cossim = np.dot(m, s) / (np.linalg.norm(s) * np.linalg.norm(m, axis=1))
        sim = 1 - np.arccos(cossim) / np.pi
        return sim
    
    def get_mmr_score(self, s, q, selected):
        """Get MMR (Maximal Marginal Relevance) score of a sentence

        Params:
        s (np.array): sentence embedding (512,)
        q (np.array): query embedding (512,)
        selected (np.array): embedding of selected sentences (m, 512)

        Returns:
        mmr_score (float)
        """

        relevance = self._lambda * self.get_similarity(s, q)
        if selected.shape[0] > 0:
            negative_diversity = (1 - self._lambda) * np.max(self.get_similarity_with_matrix(s, selected))
        else:
            negative_diversity = 0
        return relevance - negative_diversity

    def summarize(self, document:List[str], title:str=None, max_length:int=3, _lambda:float=0.5):
        """Summarize document
        
        Params:
        document (List[str]): list of sentences
        title (str, nullable): title if needed
        max_length (int): maximum number of summary sentences
        _lambda (float): weights in mmr score

        Returns:
        summary (List[str]): summary sentences
        selected (List[bool]): Whether extracted
        """
        
        self.max_length = max_length
        self._lambda = _lambda
        selected = [False] * len(document)

        embedding = self.sentence_encoder(document).numpy()
        if title:
            query = self.sentence_encoder([title])[0].numpy()
        else:
            query = np.mean(embedding, axis=0) # (512,)

        while np.mean(selected) < 1 and np.sum(selected) < max_length:
            selected_embedding = embedding[selected]
            remaining_idx = [idx for idx, i in enumerate(selected) if not i]
            mmr_score = [self.get_mmr_score(embedding[i], query, selected_embedding) for i in remaining_idx]
            best_idx = remaining_idx[np.argsort(mmr_score)[-1]]
            selected[best_idx] = True

        summary = [document[idx] for idx, i in enumerate(selected) if i]
        return summary, selected