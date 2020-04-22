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
from gensim.models import word2vec

import matplotlib as mpl
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Dropout, ELU, LeakyReLU
from keras import regularizers, optimizers
from keras.models import Model#, save_model, load_model
from keras.callbacks import EarlyStopping, Callback


import itertools
fr_it = itertools.chain.from_iterable
from itertools import repeat

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# make pretty
plt.style.use('seaborn')

# reproducible results
np.random.seed(27)

#%% DATA

if os.path.isfile('tweets_tokens.pkl'):
    print("Tokens already exist, loading...", end='')
    df = pd.read_pickle('tweets_tokens.pkl')
    print('Done.\n')

else:
    # import data
    print("Loading raw data...\n")
    df = pd.read_csv("../../data/Twitter/pre_2020-03-12_working_updated.csv")
    df = df.sample(frac=0.10)
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

if os.path.isfile('feature_mat.npy'):
    print("Feature matrix already exists, loading...", end='')
    X = np.load('feature_mat.npy')
    print('Done. Features have shape {}.\n'.format(X.shape))
else:

    # Set hyper parameters #
    
    # Word vector dimensionality  
    n_features = 100
    # Context window size 
    window = 5
    # Minimum word count                                                                                              
    min_word_count = 1
    # Downsample setting for frequent words                    
    sample = 1e-3
    
    print("Training Word2Vec model...")
    w2v = word2vec.Word2Vec(df['tokens'].to_list(),
                            size=n_features,
                            window=window,
                            min_count=min_word_count,
                            sample=sample)
    print("Done.")
    
    # we take tweets to vec by measuring average word value
    def average_word_vectors(words, model, vocab, num_features):
        
        feature_vec = np.zeros((num_features,),dtype="float64")
        n_words = 0.
        
        for word in words:
            if word in vocab: 
                n_words = n_words + 1.
                feature_vec = np.add(feature_vec, model[word])
        
        if n_words:
            feature_vec = np.divide(feature_vec, n_words)
            
        return feature_vec
        
       
    def tweets2vec(corpus, model, num_features):
        vocab = set(model.wv.index2word)
        features = [average_word_vectors(tokenized_sentence, model, vocab, num_features)
                        for tokenized_sentence in corpus]
        return np.array(features)

    # get document level embeddings
    print('Vectorizing tweets...')
    X = tweets2vec(corpus=df['tokens'].to_list(),
                   model=w2v,
                   num_features=n_features)
    print('Done.')

    print("Saving feature matrix...", end='')
    np.save("feature_mat.npy", X)
    print('Done. Features have shape {}.\n'.format(X.shape))

#%% Let's define an autoencoder to do the heavy dimension reduction

if os.path.isfile('encoded_mat.npy'):
    print("Encoded features already exist, loading...", end='')
    X_en = np.load('encoded_mat.npy')
    print("Done.")

else:
    print('Features are Gaussian, normalizing...', end='')
    mu = np.mean(X, axis=0)
    s = np.std(X, axis=0)
    #X = (X - mu) / s
    M = np.max(np.abs(X), axis=0)
    X = X / M
    print('Done.\n')
    
    print("Initializing model...", end='')
    
    # basic hyperparameters
    n_epochs = 100
    l_batch = 64
    split = 0.25
    n_batches = round((1 - split) * X.shape[0] / l_batch)
    
    # early stopping parameter
    pat = 3
    
    # regularization parameters
    reg = 1e-5
    regulzr = regularizers.l2(reg)
    drop_pct = 0.01
    
    # unit counts    
    dims = [50, 25, 12, 6, 2]

    def generate_coders(dims, drop_rate, regularizer):
        
        # load in data
        input_img = Input(shape=(X.shape[-1],),
                        name='Feature_Layer')
        
        # pass through the encoder
        encoded = Dense(units=dims[0],
                        kernel_regularizer=regularizer, 
                        name='Encoder_0')(input_img)
        encoded = ELU()(encoded)
        encoded = Dropout(drop_rate, 
                          input_shape=(dims[0],), 
                          name='Encoder_Drop_0')(encoded)
        for i in range(len(dims[1:])):
            encoded = Dense(units=dims[i + 1],
                            kernel_regularizer=regularizer, 
                            name='Encoder_{}'.format(i + 1))(encoded)
            encoded = ELU()(encoded)
            encoded = Dropout(drop_rate, 
                              input_shape=(dims[i + 1],), 
                              name='Encoder_Drop_{}'.format(i + 1))(encoded)
        
        # pass through decoder
        decoded = Dense(units=dims[-2], 
                        kernel_regularizer=regularizer, 
                        name='Decoder_0')(encoded)
        decoded = ELU()(decoded)
        decoded = Dropout(drop_rate, 
                          input_shape=(dims[-1],),
                          name='Decoder_Drop_{}'.format(i + 1))(decoded)
        
        for i in range(len(dims[:-2])):
            decoded = Dense(units=dims[len(dims) - i - 3], 
                            kernel_regularizer=regularizer, 
                            name='Dense_{}'.format(i))(decoded)
            decoded = ELU()(decoded)
            decoded = Dropout(drop_rate, input_shape=(dims[len(dims) - i - 3],), 
                              name='Drop_{}'.format(i + 1))(decoded)
        
        # Exit the decoder
        output_img = Dense(units=X.shape[-1], 
                        activation='tanh', 
                        name='Exit_Layer')(decoded)
        autoencoder=Model(input_img, 
                          output_img, 
                          name='autoencoder')
        encoder = Model(input_img, 
                        encoded, 
                        name='encoder')
        
        return encoder, autoencoder

    encoder, autoencoder = generate_coders(dims, 
                                           drop_pct, 
                                           regularizer=regulzr)

    # print summary
    autoencoder.summary()

    # compile AE
    sgd = optimizers.SGD(lr=1,
                         momentum=0.1,
                         nesterov=True,
                         clipvalue=0.5)
    
    adam = optimizers.Adam(learning_rate=0.01, 
                           beta_1=0.9, 
                           beta_2=0.999, 
                           amsgrad=True)
    
    autoencoder.compile(optimizer=sgd,
                        loss='mse',
                        metrics=['mae'])

    print("Done.\n")

    earlystop = EarlyStopping(verbose=True,
                              patience=pat,
                              monitor='val_loss')
    
    class MyCallback(Callback):
        def on_train_begin(self, logs={}):
            self.losses = []
            self.maes = []
    
        def on_batch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))
            self.maes.append(logs.get('mae'))    
            
    batch_hist = MyCallback()

    fit_params = {"batch_size": l_batch,
                  "epochs": n_epochs,
                  "validation_split": split,
                  "shuffle": True,
                  "callbacks": [earlystop,
                                batch_hist]}

    print("Training model...", end='')
    
    hist = autoencoder.fit(X, X, **fit_params)
    print("Done.")

    # save weights
    print("Saving AE weights...", end='')
    autoencoder.save_weights('ae_weights.h5')
    print("Done.")

    print('Plotting training history...', end='')

    # plot acc and loss
    fig4 = plt.figure()
    ax4_1 = plt.subplot(211)
    ax4_2 = plt.subplot(212, sharex=ax4_1)
    
    ax4_1.semilogy(batch_hist.maes, label='train')
    ax4_1.semilogy(list(fr_it(repeat(v, n_batches) for 
                        v in hist.history['val_mae'])), label='validation')
    
    ax4_2.semilogy(batch_hist.losses, label='train')
    ax4_2.semilogy(list(fr_it(repeat(v, n_batches) for 
                        v in hist.history['val_loss'])), label='validation')
    
    ax4_1.set_title('Training History')
    plt.xlabel("batches")
    ax4_1.set_ylabel('mae')
    ax4_2.set_ylabel('loss')
    plt.setp(ax4_1.get_xticklabels(), visible=False)
    
    
    ax4_1.legend(loc='upper right')
    ax4_2.legend(loc='upper right')
    plt.savefig("training_history.png", dpi=300)

    print('Done.\n')

    # encode data to reduce dimension
    print("Encoding data to {} dimensions...".format(dims[-1]))
    X_en = encoder.predict(X, verbose=1)
    print("Done.")
    

    print('Saving encoded data...', end='')
    np.save('encoded_mat.npy', X_en)
    print('Done.\n')
    
#%% Clustering

if (os.path.isfile('tweets_tokens_clusters.pkl')):
    print('Clusters have already been found!\n')
else:
    print("Clustering Tweets...", end='')
    K = 7
    km = KMeans(n_clusters=8).fit(X_en)
    print("Done.")

    # add columns to df and save
    df["cluster"] = pd.Series(km.labels_, index=df.index)
    print("Saving clustered tweets...", end='')
    df.to_pickle("tweets_tokens_clusters.pkl")
    print("Done.")

#%% Plot 2D PCA projection

if os.path.isfile('embedding_pca.png'):
    print('Plot of PCA projection already exists!\n')

else:

    # define the colormap
    cmap = plt.cm.rainbow
    # extract all colors from the color map
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
    plt.xlabel(('variance: {}% from 2/{} ({}%) components'
                .format(var_pct, X_en.shape[1], comp_pct)))
    ax3.set_yticklabels([])
    ax3.set_xticklabels([])
    print('Saving PCA plot...', end='')
    plt.savefig("embedding_pca.png", dpi=300)
    print('Done.\n')

#%% 2D tSNE Projection

if os.path.isfile('embedding_AE.png'):
    print('Plot of AE projection already exists!\n')
else:

    # define the colormap
    cmap = plt.cm.rainbow
    # extract all colors from the map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    # define the bins and no rmalize
    K = len(df['cluster'].unique())
    bounds = np.linspace(0, K, K + 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    fig4  = plt.figure()
    ax4 = plt.subplot(111)
    scat = ax4.scatter(X_en[:, 0],
                        X_en[:, 1],
                        c=df['cluster'].to_list(),
                        cmap=cmap,
                        norm=norm,
                        s=1)
    # create the colorbar
    cb = plt.colorbar(scat, spacing='proportional', ticks=bounds)
    cb.set_label('Classes')
    ax4.set_title('Tweets - AE')
    ax4.set_yticklabels([])
    ax4.set_xticklabels([])
    print('Saving latent encoding plot...', end='')
    plt.savefig("embedding_AE.png", dpi=300)
    print('Done.\n')

#%%

print("\n\nProgram exiting.")