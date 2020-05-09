'''
word embeddings related functions
'''

__author__ = 'Oguzhan Gencoglu'

import numpy as np
from sentence_transformers import SentenceTransformer

from configs import config as cf


# _____________ sentence BERT _____________
def get_bert_embeddings(data):
    '''
    extract sentence BERT embeddings
    [data] : pandas series/numpy array/list of strings
    '''

    bert_model = SentenceTransformer(cf.model_identifier)
    embeddings = np.array(bert_model.encode(np.array(data)))

    return embeddings
