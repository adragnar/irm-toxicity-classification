'Utility Functions for Generating Datasets, Loading Word Embeddings, Generating Model Objects, and Preprocessing Text'

from gensim.models import KeyedVectors

import re  

import warnings
warnings.filterwarnings("ignore")

import pickle

from torch.utils.data import Dataset

import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOPWORDS = set(stopwords.words('english'))

# DATASETS
class ToxicityDataset(Dataset):

    def __init__(self, data, rel_cols={'data':'comment_text', 'labels':'toxicity'}, add_sa=False, transform=None):
        """
        Args:
            data (pd dataframe): The pd Dataframe dataset
            rel_cols: Dictinonary mapping names of columns in relevant dataset to ['data', 'labels', 'sens_att'] 
            add_sa: A function applied to data which returns np.series of SA attributes
            transform: Function genrating comment embeddings from raw text
        """

        self.data = transform(data[rel_cols['data']])
        if add_sa:
            sa_col = np.expand_dims(add_sa(data).squeeze(), axis=1)
            assert len(sa_col.shape) == 2
            self.data = np.concatenate((self.data, sa_col), axis=1)
        self.labels = data[rel_cols['labels']].values

        if 'sens_att' in rel_cols:
            self.sens_att = data[rel_cols['sens_att']].values
        else:
            self.sens_att = None

        self.dim = self.data.shape[1]
        self.origdata = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        '''Return dicitionary of the data, label and sensitive attribute for given index'''
        if self.sens_att is not None:
            sa = self.sens_att[idx]
        else:
            sa = None
        return {'x':self.data[idx, :], 'y':self.labels[idx], 'sens_att':sa}


#EMBEDDINGS
class GetSBERT(object):
    """Given a sentence of text, generate sentence embedding"""

    def __init__(self):
        self.model = SentenceTransformer('distilbert-base-nli-mean-tokens')

    def __call__(self, sample):
        ''':param sample: pd.Series'''

        #First convert input sentences into list of strings
        if type(sample) == str:
            sample = [sample]
        elif type(sample) == pd.Series:
            sample = sample.tolist()

        #Now make the embeddings
        sent_embedding = self.model.encode(sample)
        return sent_embedding.squeeze()



class GetEmbedding(object):
    """Given a sentence of text, generate sentence embedding

    Args:
        model: dictionary of words -> embeddings
    """

    def __init__(self, model, stopwords=[], se_type='embed_mean'):
        self.model = model
        self.stopwords = stopwords
        self.unknown_embed = np.zeros(300)  
        self.se_type = se_type

    def embed_sentence(self, sample):
        '''param sample: The sentence to be embedded'''
        words = sample.split(' ')
        if len(words) == 0:
            words = ['flughahanooneknowsthiswort'] #makes embedding zeros vec

        embed_list = [self.model[w] if w in self.model else self.unknown_embed for w in words]
        if self.se_type == 'embed_mean_exunk':
            true_len = max(1, len([e for e in embed_list if (e != self.unknown_embed).any()]))
            sent_embedding = np.sum(embed_list, axis = 0)/true_len
        elif self.se_type == 'embed_mean':
            sent_embedding = np.mean(embed_list, axis = 0)
        elif self.se_type == 'embed_sum':
            sent_embedding = np.sum(embed_list, axis = 0)

        return sent_embedding

    def __call__(self, sample):
        ''':param sample: pd.Series'''
        if type(sample) == str:
            sent_embedding = self.embed_sentence(sample)

        elif type(sample) == pd.Series:
            sent_embedding = np.zeros((len(sample), 300))
            for i, txt in enumerate(sample):
                sent_embedding[i, :] = self.embed_sentence(txt)

        return sent_embedding


def get_word_transform(e_type, fpath, testing=0):
    if e_type == 'sbert':
        assert fpath == 'NA'
        t = GetSBERT()
    elif 'embed' in e_type:
        word2vec, _, _ = load_word_vectors(fpath, testing)
        t = GetEmbedding(word2vec, stopwords=STOPWORDS, se_type=e_type)

    return t

def load_word_vectors(fname, testing=0):
    if testing == 1: 
        n_vecs = 1000
    else:
        n_vecs = None

    model = KeyedVectors.load_word2vec_format(fname, limit=n_vecs, binary=False)
    vecs = model.vectors
    words = list(model.vocab.keys())
    return model, vecs, words


#MODEL LOADING
def load_saved_model(mpath):
    '''Given path to a model data structure manually saved, reconstruct the
    appropiate model objext (ie - such that .predict() can be called)'''
    model_info = pickle.load(open(mpath, 'rb'))
    assert set(['model_base', 'model']) ==  set(model_info['model'].keys())
    
    base = model_info['model']['model_base']
    base.model = model_info['model']['model']
    return base

#DATA PROCESSING
def preprocess_data(data, rel_cols, tox_thresh=None, c_len=15, \
                                 text_clean='reg', stopwords=[]):
    '''Raw preprocessing to be done before data is used on both data All non nan
       values stuff controlled by kwargs
    :param data: pd dataframe of dataset 
    :param rel_cols dictionary mapping keys (data, labels) to names in given dset
    :param tox_thresh: threshold for binarizing tox label (if dataset has labels)
    :param c_len: min length of comment
    :param text_clean: kind of text preprocessing to do
    :param stopwords: words to remove'''

    #remove Nans in comment text column
    data['test_nan']= data[rel_cols['data']].apply(lambda x: 1 if type(x) == str else 0)
    data = data[data['test_nan'] == 1]; data.reset_index(inplace=True, drop=True)
    data.drop(['test_nan'], axis=1, inplace=True)

    #Do social media preprocess
    if text_clean == 'reg':  
        def proc(txt, stopwords=[]):
            words = txt.split(" ")
            words = [re.sub(r'\W+', '', w).lower() for w in words \
                     if re.sub(r'\W+', '', w).lower() not in stopwords]
            return words

        data[rel_cols['data']] = data[rel_cols['data']].apply(\
                        lambda s: " ".join(proc(s, stopwords=STOPWORDS)))

    elif text_clean == 'na':
        pass
    else:
        raise Exception('Unimplemented cleaning')

    #Remove too small comments
    if c_len is not None:
        data['test_len'] = data[rel_cols['data']].apply(lambda x: 1 if (len(str(x)) > c_len) else 0)
        data = data[data['test_len'] == 1]; data.reset_index(inplace=True, drop=True)
        data.drop(['test_len'], axis=1, inplace=True)

    #Binarize labels 
    if tox_thresh is not None:
        data[rel_cols['labels']] = data[rel_cols['labels']].apply((lambda x: 1 if x > tox_thresh else 0))

    return data
