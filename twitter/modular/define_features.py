# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 19:53:49 2020

@author: tharshi
"""
#%% Import Frameworks and Set-up

# import useful frameworks
import numpy as np
import os

import gensim.models as gm
import gensim.downloader as api

import matplotlib.pyplot as plt

import itertools
fr_it = itertools.chain.from_iterable

# make pretty
plt.style.use('seaborn')

# reproducible results
np.random.seed(27)

# load tokens
tokenized = np.load('tokenized_tweets.npy', allow_pickle=True)

#%% Build features from token data

# we take tweets to vec by measuring average word value
def average_word_vectors(words, model, vocab, num_features):
    
    feature_vec = np.zeros((num_features,), dtype="float64")
    n_words = 0.
    
    for word in words:
        if word in vocab: 
            n_words = n_words + 1.
            feature_vec = np.add(feature_vec, model[word])
    
    if n_words:
        feature_vec = np.divide(feature_vec, n_words)
        
    return feature_vec
    
def tweets2vec(corpus, model):
    num_features = model.vector_size
    vocab = set(model.wv.index2word)
    features = [average_word_vectors(tokenized_sentence, 
                                     model, 
                                     vocab, 
                                     num_features) for 
                tokenized_sentence in corpus]
    return np.array(features)
    num_features = model.vector_size
    vocab = set(model.wv.index2word)
    features = [average_word_vectors(tokenized_sentence, 
                                     model, 
                                     vocab, 
                                     num_features) 
                for tokenized_sentence in corpus]
    return np.array(features)

if os.path.isfile('w2v.model'):
    print("Word2Vec model already exists, loading...", end='')
    w2v = gm.keyedvectors.KeyedVectors.load('w2v.model')
    print('Done.\n')
    
else:
    # Load pre-trained model
    print('Downloading glove-twitter dataset...')
    w2v = api.load('glove-twitter-200')
    print('Done.')
    
    print('Saving Word2Vec model...', end='')
    w2v.save('w2v.model')
    print('Done.\n')

# get document level embeddings
print('Creating feature matrix...\n')
X = tweets2vec(corpus=tokenized, model=w2v)
print('\nDone.\n')

print('Saving feature matrix...', end='')
np.save('features.npy', X)
print('Done.\n')

#%%

print("\n\nProgram exiting.")


# from nltk.tokenize import WordPunctTokenizer
# tok = WordPunctTokenizer()
# pat1 = r'@[A-Za-z0-9]+'
# pat2 = r'https?://[A-Za-z0-9./]+'
# combined_pat = r'|'.join((pat1, pat2))
# def tweet_cleaner(text):
#     soup = BeautifulSoup(text, 'lxml')
#     souped = soup.get_text()
#     stripped = re.sub(combined_pat, '', souped)
#     try:
#         clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
#     except:
#         clean = stripped
#     letters_only = re.sub("[^a-zA-Z]", " ", clean)
#     lower_case = letters_only.lower()
#     # During the letters_only process two lines above, it has created unnecessay white spaces,
#     # I will tokenize and join together to remove unneccessary white spaces
#     words = tok.tokenize(lower_case)
#     return (" ".join(words)).strip()
# testing = df.text[:100]
# test_result = []
# for t in testing:
#     test_result.append(tweet_cleaner(t))
# test_result