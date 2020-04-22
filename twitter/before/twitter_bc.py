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
from gensim.models import word2vec, Word2Vec

import matplotlib as mpl
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Dropout, ELU
from keras import regularizers, optimizers
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, Callback

import itertools
fr_it = itertools.chain.from_iterable
from itertools import repeat

from sklearn.cluster import KMeans
from kneed import KneeLocator

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# make pretty
plt.style.use('seaborn')

# reproducible results
np.random.seed(27)

# kmeans control
kmeans = True
if not kmeans:
    K = 8
else:
    pass

# tsne control
plot_tsne = False
#%% Data and Preprocessing

if os.path.isfile('tokenized_tweets.npy'):
    
    print('Tweets have already been generated, loading...', end='')
    tokenized = np.load('tokenized_tweets.npy', allow_pickle=True).tolist()
    print('Done.')

else:

    # import data
    print("Loading raw data...\n")
    df = pd.read_csv("../../data/Twitter/pre_2020-03-12_working_updated.csv")
    df = df.sample(frac=1.0)
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

#%% Build features from token data

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
    w2v = Word2Vec.load('w2v.model')
    print('Done.\n')
    
else:
    # Set hyper parameters #
    
    # Word vector dimensionality  
    n_features = 300
    # Context window size 
    window = 30
    # Minimum word count                                                                                              
    min_word_count = 2
    # Downsample setting for frequent words                    
    sample = 1e-3
    
    print("Training Word2Vec model...")
    w2v = word2vec.Word2Vec(tokenized,
                            size=n_features,
                            window=window,
                            min_count=min_word_count,
                            sample=sample)
    print("Done.\n")
    
    print('Saving Word2Vec model...', end='')
    w2v.save('w2v.model')
    print('Done.\n')

# get document level embeddings
print('Creating feature matrix...\n')
X = tweets2vec(corpus=tokenized, model=w2v)
print('\nDone.\n')
#%% Use an autoencoder to do the heavy dimension reduction
    
models_ = os.path.isfile('autoencoder.h5')
weights_ = os.path.isfile('ae_weights.h5')

if models_ and weights_:
    print("Autoencoder already exist, loading...", end='')
    autoencoder = load_model('autoencoder.h5')
    print("Done.")

    print('Compiling model...', end='')
    # compile AE for training
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
    print('Done.\n')

    #print summary
    autoencoder.summary()

    print("\nTrained weights already exist, loading...", end='')
    autoencoder.load_weights('ae_weights.h5')
    print("Done.\n")
    
    # define encoder based on autoencoder
    index = None
    for idx, layer in enumerate(autoencoder.layers):
        if layer.name == 'Latent_Layer_Drop':
            index = idx
            break
    encoder = Model(autoencoder.layers[0].input, 
                    autoencoder.layers[index].output, 
                    name='encoder')
    
    # encode data to reduce dimension
    print("Encoding data...")
    X_en = encoder.predict(X, verbose=1)
    print("Done.\n")
    
else:
    def generate_coders(dims, drop_rate, regularizer):
        
        n_dims = len(dims)
        
        # load in data
        input_img = Input(shape=(X.shape[-1],),
                        name='Feature_Layer')
        
        # pass through the encoder
        N = dims[0]
        encoded = Dense(units=N,
                        kernel_regularizer=regularizer, 
                        name='Encoder_0')(input_img)
        encoded = ELU()(encoded)
        encoded = Dropout(drop_rate, 
                          input_shape=(N,), 
                          name='Encoder_Drop_0')(encoded)
        
        for i in range(n_dims - 2):
            encoded = Dense(units=dims[i + 1],
                            kernel_regularizer=regularizer, 
                            name='Encoder_{}'.format(i + 1))(encoded)
            encoded = ELU()(encoded)
            encoded = Dropout(drop_rate, 
                              input_shape=(dims[i + 1],), 
                              name='Encoder_Drop_{}'.format(i + 1))(encoded)
            
        latent = Dense(units=dims[-1],
                        kernel_regularizer=regularizer, 
                        name='Latent_Layer')(encoded)
        latent = ELU()(latent)
        latent = Dropout(drop_rate, 
                          input_shape=(dims[-1],), 
                          name='Latent_Layer_Drop')(latent)
        
        # pass through decoder
        N = dims[-2]
        decoded = Dense(units=N, 
                        kernel_regularizer=regularizer, 
                        name='Decoder_0')(latent)
        decoded = ELU()(decoded)
        decoded = Dropout(drop_rate, 
                          input_shape=(N,),
                          name='Decoder_Drop_{}'.format(i + 1))(decoded)
        
        for i in range(n_dims - 2):
            decoded = Dense(units=dims[n_dims - i - 3], 
                            kernel_regularizer=regularizer, 
                            name='Dense_{}'.format(i))(decoded)
            decoded = ELU()(decoded)
            decoded = Dropout(drop_rate, input_shape=(dims[n_dims - i - 3],), 
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
    
    # model parameters
    reg = 1e-4
    regulzr = regularizers.l2(reg)
    drop_pct = 0.05
    
    # unit counts    
    dims = [150, 100, 33, 15, 7]

    # define model
    encoder, autoencoder = generate_coders(dims, 
                                           drop_pct, 
                                           regularizer=regulzr)
    print('Compiling model...', end='')
    # compile AE for training
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
    print('Done.\n')
    
    autoencoder.summary()

    print('\nFeatures are Gaussian, normalizing...', end='')
    mu = np.mean(X, axis=0)
    s = np.std(X, axis=0)
    #X = (X - mu) / s
    M = np.max(np.abs(X), axis=0)
    X = X / M
    print('Done.\n')
    
    print("Initializing model...", end='')
    
    # basic hyperparameters
    n_epochs = 1000
    l_batch = 64
    split = 0.25
    n_batches = round((1 - split) * X.shape[0] / l_batch)
    
    # early stopping parameter
    pat = 3    

    earlystop = EarlyStopping(verbose=True,
                              patience=pat,
                              monitor='val_loss')
    
    # define custom callbacks
    class MyCallback(Callback):
        def on_train_begin(self, logs={}):
            self.losses = []
            self.maes = []
    
        def on_batch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))
            self.maes.append(logs.get('mae'))    

    batch_hist = MyCallback()



    print("Done.\n")

    # Define fitting parameters and train
    fit_params = {"batch_size": l_batch,
                  "epochs": n_epochs,
                  "validation_split": split,
                  "shuffle": True,
                  "callbacks": [earlystop,
                                batch_hist]}

    print("Training model...", end='')
    hist = autoencoder.fit(X, X, **fit_params)
    print("Done.\n")
    
    # encode data to reduce dimension
    print("Encoding data to {} dimensions...".format(dims[-1]))
    X_en = encoder.predict(X, verbose=1)
    print("Done.\n")

    # save models
    autoencoder.reset_metrics()
    print('Saving models...', end='')
    autoencoder.save('autoencoder.h5')
    print('Done.')

    # save weights
    print('Saving weights...', end='')
    autoencoder.save_weights('ae_weights.h5')
    print('Done.')

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

#%% K-Means clustering - choosing K
if os.path.isfile('k_val.npy'):
    print('Optimum K already exists, loading...', end='')
    K = np.load('k_val.npy')
    print('Done.\n')
else:
    if kmeans:
        print("Running K - Means clustering...", end='')
    
        # figure out what K is good for the data set
        km_inertias = []
    
        n_k = 50
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
        print("Done.\n")
        
        # plot results
        kneedle.plot_knee()
        plt.ylabel("K-Means Inertia")
        plt.xlabel("K")
        plt.title("Inertia Plot for K-Means Algorithm - $K^* = {}$".format(K))
        plt.savefig("inertia.png", dpi=300)
    else:
        pass
print("Clustering Tweets...", end='')
km = KMeans(n_clusters=K)
km = km.fit(X_en)
print("Done.")

if os.path.isfile('labels.npy'):
    print('Labels already exist, loading', end='')
    labels = np.load('labels.npy')
    print('Done.\n')
else:
    print('Saving labels...', end='')
    labels = km.labels_
    np.save('labels.npy', labels)
    print('Done.\n')

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
                        c=labels,
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

if os.path.isfile('embedding_tsne.png'):
    print('Plot of tSNE projection already exists!\n')
else:
    if plot_tsne:
        # define the colormap
        cmap = plt.cm.rainbow
        # extract all colors from the map
        cmaplist = [cmap(i) for i in range(cmap.N)]
        # create the new map
        cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    
        # define the bins and no rmalize
        bounds = np.linspace(0, K, K + 1)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
        # Use tSNE to project data to 2D
        print('Reducing dimension via tSNE...')
        p = 1000
        tsne = TSNE(n_components=2, verbose=2, perplexity=p)
        X_t = tsne.fit_transform(X_en)
        print('Done.')
    
        fig4  = plt.figure()
        ax4 = plt.subplot(111)
        scat = ax4.scatter(X_t[:, 0],
                            X_t[:, 1],
                            c=labels,
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
        plt.savefig("embedding_tsne.png", dpi=300)
        print('Done.\n')
    else:
        pass

#%%

print("\n\nProgram exiting.")