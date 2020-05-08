# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:52:20 2020

@author: tharshi
"""

#%% Import Frameworks and Set-up

# import useful frameworks
import pandas as pd
import numpy as np
import os

import re
import nltk
from nltk.stem import PorterStemmer
#%% Data and Preprocessing

if os.path.isfile('tokenized_tweets.npy'):
    
    print('Tweets have already been generated, loading...', end='')
    tokenized = np.load('tokenized_tweets.npy', allow_pickle=True).tolist()
    print('Done.')

else:

    # import data
    print("Loading raw data...\n")
    df = pd.read_csv("../../data/Twitter/pre_2020-03-12_working_updated.csv")
    df = df.sample(frac=0.1)
    print("\nDone.")

    print("Cleaning data...", end='')
    # drop uneeded columns
    df = df.drop(["favourites_count", 'is_retweet', 'is_quote',
                  "retweet_count", "place_full_name",
                  "friends_count", "verified",
                  "followers_count",
                  "place_type", "created_at"], axis=1)
    df = df.dropna(axis=0)
    df = df.reset_index(drop=True)
    print("Done.\n")

    print("Tokenizing data...")

    # Emoticons
    emoticons_str = r"""
        (?:
            [:=;] # Eyes
            [oO\-]? # Nose
            [D\)\]\(\]/\\OpP] # Mouth
        )"""

    # Define the regex strings.
    regex_str = [
        #emoticons_str,
        #r'<[^>]+>', # HTML tags
        r'(?:@[\w_]+)', # @-mentions
        r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
#r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',
        r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
        r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
        r'(?:[\w_]+)', # other words
        #r'(?:\S)' # anything else
    ]

    # Assign strings
    tokens_re = re.compile(r'('+'|'.join(regex_str)+')',
                           re.VERBOSE | re.IGNORECASE)
    emoji_re = re.compile(r'^'+emoticons_str+'$',
                             re.VERBOSE | re.IGNORECASE)


    # define stop words
    s_words = nltk.corpus.stopwords.words('english')
    newStopWords = ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'k', 'l',
                    'm', 'n', 'o', 'p', 'r', 's',
                    't', 'u', 'v', 'w', 'x', 'y', 'j', "'"]
    s_words.extend(newStopWords)

    def tokenise(s):
        return tokens_re.findall(s)

    def preprocess(s, lowercase=True):
        # remove urls
        s = re.sub((r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|'
                    '[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+'),
                   '', s, flags=re.MULTILINE)

        # remove hashtags
        s = re.sub(r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", '', s, flags=re.MULTILINE)

        # remove handles
        s = re.sub(r'(?:@[\w_]+)', '', s, flags=re.MULTILINE)

        # remove numbers
        s = re.sub(r'(?:(?:\d+,?)+(?:\.?\d+)?)', '', s, flags=re.MULTILINE)

        tokens = tokenise(s)
        if lowercase:
            tokens = [t if emoji_re.search(t) else t.lower() for t in tokens]

        # stem the tokens
        tokens = [ps.stem(token) for token in tokens]
        
        # remove stop words
        tokens = [token for token in tokens if token not in s_words]

        return tokens

    # create stemmer
    ps = PorterStemmer()
    
    # Get the tokenized value for each word
    tweets = df['text'].to_list()
    tokenized = []
    for tweet in tweets:
        tokens = preprocess(tweet)
        # add to list
        tokenized.append(tokens)
    print("Done.")

    print("Saving tokens...", end='')
    np.save('tokenized_tweets', tokenized)
    print("Done.\n")

#%%

print("\n\nProgram exiting.")