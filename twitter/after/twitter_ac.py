# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 22:29:51 2020

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
from sklearn.feature_extraction.text import TfidfVectorizer

import scipy as sp

import matplotlib as mpl
import matplotlib.pyplot as plt

from keras.layers import Input, Dense
from keras.models import Model#, save_model, load_model
from keras.callbacks import EarlyStopping
#from keras.callbacks import Callback

from sklearn.cluster import KMeans
from kneed import KneeLocator

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# make pretty
plt.style.use('seaborn')

# reproducible results
np.random.seed(27)

#%% DATA

if os.path.isfile('tweets_tokens.pkl'):
    print("Tokens already exist, loading...", end='')
    df = pd.read_pickle('tweets_tokens.pkl')
    print('Done.\n')
    
elif os.path.isfile('tweets_tokens_clusters.pkl'):
    print("Tokens already exist, loading...", end='')
    df = pd.read_pickle('tweets_tokens_clusters.pkl')
    print('Done.\n')

else:

    # import data
    print("Loading raw data...\n")
    df = pd.read_csv("../../data/Twitter/2020-04-01.csv")
    df = df.sample(frac=1.0)
    print("\nDone.")
    
    print("Cleaning data...", end='')
    # drop uneeded columns
    df = df[['text', 'lang']]
    
    # only english tweets
    df = df[df['lang'] == 'en'].drop('lang', axis=1)
    
    # drop NA
    df = df.dropna(axis=0)
    
    # reset index
    df = df.reset_index(drop=True)
    
    # make new token column
    df['tokens'] = ''
    print("Done.\n")
    
    print("Tokenizing data...", end='')

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
        #r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
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

        return tokens

    # create stemmer
    ps = PorterStemmer()

    # Get the tokenized value for each word
    tweets = df['text'].to_list()
    for i in range(len(tweets)):
        tweet = tweets[i]
        tokens = preprocess(tweet)
        # add to data frame
        df.at[i, 'tokens'] = tokens
    print("Done.")

    print("Saving new dataframe with tokens...", end='')
    df.to_pickle("tweets_tokens.pkl")
    print("Done.\n")

#%% Build features from token data

if os.path.isfile('feature_mat.npz'):
    print("TFIDF feature matrix already exists, loading...", end='')
    X = sp.sparse.load_npz('feature_mat.npz')
    print('Done. Features have shape {}.\n'.format(X.shape))
else:
    s_words = nltk.corpus.stopwords.words('english')
    newStopWords = ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'k', 'l',
                    'm', 'n', 'o', 'p', 'r', 's',
                    't', 'u', 'v', 'w', 'x', 'y', 'j', "'"]
    s_words.extend(newStopWords)

    # we pass the list of tokens to the vectorizer
    def dummy_fun(doc):
        return doc

    tfidf = TfidfVectorizer(
        analyzer='word',
        tokenizer=dummy_fun,
        preprocessor=dummy_fun,
        token_pattern=None,
        stop_words=s_words)

    print("Vectorizing tokens using TFIDF transformation...", end='')
    X = tfidf.fit_transform(df['tokens'].to_list())
    print("Done.")

    print("Saving feature matrix...", end='')
    sp.sparse.save_npz("feature_mat.npz", X)
    print('Done. Features have shape {}.\n'.format(X.shape))

#%% Let's define an autoencoder to do the heavy dimension reduction

if os.path.isfile('encoded_mat.npy'):
    print("Encoded features already exist, loading...", end='')
    X_en = np.load('encoded_mat.npy')
    print("Done.\n")

else:
    print("Initializing model...", end='')

    n_epochs = 5
    batch_size = 64
    split = 0.30
    pat = 5

    N1 = 30
    N2 = 20
    N3 = 10

    # define autoencoder
    input_img = Input(shape=(X.shape[-1],))
    encoded = Dense(units=N1, activation='relu')(input_img)
    encoded = Dense(units=N2, activation='relu')(encoded)
    encoded = Dense(units=N3, activation='relu')(encoded)
    decoded = Dense(units=N2, activation='relu')(encoded)
    decoded = Dense(units=N1, activation='relu')(decoded)
    decoded = Dense(units=X.shape[-1], activation='sigmoid')(decoded)
    autoencoder=Model(input_img, decoded)
    encoder = Model(input_img, encoded)

    # print summary
    autoencoder.summary()

    # compile AE
    autoencoder.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy'])

    print("Done.\n")

    earlystop = EarlyStopping(verbose=True,
                              patience=pat,
                              monitor='val_loss')

    fit_params = {"batch_size": batch_size,
                  "epochs": n_epochs,
                  "validation_split": split,
                  "shuffle": True,
                  "callbacks": [earlystop]}

    print("Training model...", end='')
    history = autoencoder.fit(X, X, **fit_params)
    print("Done.")

    # save weights
    print("Saving AE weights...", end='')
    autoencoder.save_weights('ae_weights.h5')
    print("Done.")

    print('Plotting training history...', end='')

    # plot acc and loss
    fig4, ax4 = plt.subplots()
    ax4.plot(history.history['accuracy'], label='train')
    ax4.plot(history.history['val_accuracy'], label='validation')
    ax4.set_title("AE Accuracy")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig("acc_history.png", dpi=300)

    fig5, ax5 = plt.subplots()
    ax5.plot(history.history['loss'], label='train')
    ax5.plot(history.history['val_loss'], label='validation')
    ax5.set_title("AE Loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig("loss_history.png", dpi=300)

    print('Done.\n')

    # encode data to reduce dimension
    print("Encoding data to {} dimensions...".format(N3))
    X_en = encoder.predict(X, verbose=1)
    print("Done.")

    print('Saving encoded data...', end='')
    np.save('encoded_mat.npy', X_en)
    print('Done.\n')

#%% K-Means clustering - choosing K
if os.path.isfile('k_val.npy'):
    print('Optimum K already exists, loading...', end='')
    K = np.load('k_val.npy')
    print('Done.\n')
else:
    print("Running K - Means clustering...", end='')

    # figure out what K is good for the data set
    km_inertias = []

    n_k = 100
    print("There are {} values of K to test:".format(n_k))
    k_range = range(1, n_k + 1)

    for k in k_range:
        # find clusters for given k
        km = KMeans(n_clusters=k)
        km = km.fit(X_en)

        # calculate measure of distance
        inertia = km.inertia_
        km_inertias.append(inertia)
        print("k = {} ... inertia = {}".format(k, inertia))

    #km_inertias = np.array(km_inertias)
    print("Done.")

    # use the kneedle algorithm to find "elbow point"
    kneedle = KneeLocator(k_range, km_inertias, S=1.0,
                          curve='convex', direction='decreasing')
    K = round(kneedle.elbow)
    print("Estimated optimal K = {}".format(K))

    print("Saving K...", end='')
    np.save('k_val.npy', K)
    print("Done.")

    # plot results
    kneedle.plot_knee()
    plt.ylabel("K-Means Inertia")
    plt.xlabel("K")
    plt.title("Inertia Plot for K-Means Algorithm")
    plt.savefig("inertia.png", dpi=300)
#%% Final clustering with chosen K

if os.path.isfile('tweets_tokens_clusters.pkl'):
    print('Clusters have already been found!\n')
else:
    print("Clustering Tweets...", end='')
    km = KMeans(n_clusters=K)
    km = km.fit(X_en)
    print("Done.")

    # add columns to df and save
    df["cluster"] = pd.Series(km.labels_, index=df.index)
    print("Saving clustered tweets...", end='')
    df.to_pickle("tweets_tokens_clusters.pkl")
    print("Done.")

    print("Deleting uneeded dataframe save...", end='')
    if os.path.isfile('tweets_tokens.pkl'):
        os.remove('tweets_tokens.pkl')
    print("Done.\n")

#%% Plot 2D PCA projection

if os.path.isfile('embedding_pca.png'):
    print('Plot of PCA projection already exists!\n')

else:

    # define the colormap
    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    # define the bins and no rmalize
    bounds = np.linspace(0, K, K + 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # Use PCA to project data to 2D
    print('Reducing dimension via PCA...', end='')
    pca = PCA(n_components=2, whiten=True)
    X_pca = pca.fit_transform(X_en)
    print('Done.')

    fig3  = plt.figure()
    ax3 = plt.subplot(111)
    scat = ax3.scatter(X_pca[:, 0],
                        X_pca[:, 1],
                        c=df['cluster'].to_list(),
                        cmap=cmap,
                        norm=norm,
                        s=1)
    # create the colorbar
    cb = plt.colorbar(scat, spacing='proportional', ticks=bounds)
    cb.set_label('Classes')
    cb.ax.set_alpha(1)
    plt.draw()
    ax3.set_title('Tweets - PCA')

    var_pct = np.round(100 * sum(pca.explained_variance_ratio_), 3)
    comp_pct = np.round(100 * 2 / X_en.shape[1], 3)
    plt.xlabel(('variance: {:3f}% from 2/{} ({}%) components'
                .format(var_pct, X_en.shape[1], comp_pct)))
    ax3.set_yticklabels([])
    ax3.set_xticklabels([])
    print('Saving PCA plot...', end='')
    plt.savefig("embedding_pca.png", dpi=300)
    print('Done.\n')

#%% 2D tSNE Projection

if os.path.isfile('embedding_tsne.png'):
    print('Plot of tSNE projection already exists!\n')
else:

    # define the colormap
    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    # define the bins and no rmalize
    bounds = np.linspace(0, K, K + 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # Use tSNE to project data to 2D
    print('Reducing dimension via tSNE...')
    p = 15
    tsne = TSNE(n_components=2, verbose=1, perplexity=p)
    X_t = tsne.fit_transform(X_en)
    print('Done.')

    fig4  = plt.figure()
    ax4 = plt.subplot(111)
    scat = ax4.scatter(X_t[:, 0],
                        X_t[:, 1],
                        c=df['cluster'].to_list(),
                        cmap=cmap,
                        norm=norm,
                        s=1)
    # create the colorbar
    cb = plt.colorbar(scat, spacing='proportional', ticks=bounds)
    cb.set_label('Classes')
    ax4.set_title('Tweets - tSNE with perplexity {}'.format(p))
    ax4.set_yticklabels([])
    ax4.set_xticklabels([])
    print('Saving tSNE plot...', end='')
    plt.savefig("embedding_tsne_{}.png".format(p), dpi=300)
    print('Done.\n')

#%%

print("\n\nProgram exiting.")