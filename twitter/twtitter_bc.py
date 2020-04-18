# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 22:29:51 2020

@author: tharshi
"""

#%% Import Frameworks and Set-up

# import useful frameworks
import numpy as np
import re
import nltk
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from kneed import KneeLocator

from nltk.stem import PorterStemmer
import pickle

import scipy as sp

# make pretty
plt.style.use('seaborn')

# reproducible results
np.random.seed(27)

#%% DATA

# control modules, full run needs all of them on if you want to
# recalculate everything
import_data = True
need_tokens = True
need_features = True
need_model = True
train = True
test_k = True
cluster = False
viz = True

if import_data:
    # import data
    df = pd.read_csv("../data/Twitter/pre_2020-03-12_working_updated.csv")
    
    # drop uneeded columns
    df = df.drop(["favourites_count", 'is_retweet', 'is_quote',
                  "retweet_count", "place_full_name", 
                  "friends_count", "verified",
                  "followers_count",
                  "place_type", "created_at"], axis=1)
    
    df = df.sample(frac=0.25)
    
    print("Cleaning data...")
    df = df.dropna(axis=0)
    df = df.reset_index(drop=True)
    print("Done.")
#%% CLEAN DATA
if need_tokens:
    print("Tokenizing data...")
    
    tweets = df['text'].to_list()
    
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
    tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
    emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)
    
    # create stemmer
    ps = PorterStemmer()
    
    def tokenise(s):
        return tokens_re.findall(s)
     
    def preprocess(s, lowercase=True):
        # remove urls
        s = re.sub(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', '', s, flags=re.MULTILINE)
        
        # remove hashtags
        s = re.sub(r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", '', s, flags=re.MULTILINE)
        
        # remove handles
        s = re.sub(r'(?:@[\w_]+)', '', s, flags=re.MULTILINE)
       
        # remove numbers
        s = re.sub(r'(?:(?:\d+,?)+(?:\.?\d+)?)', '', s, flags=re.MULTILINE)
        
        tokens = tokenise(s)
        if lowercase:
            tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
        
        # stem the tokens
        tokens = [ps.stem(token) for token in tokens]
        
        return tokens
    
    # Get the tokenized value for each word
    tokenised = []
    for i in range(len(tweets)):
        tw = tweets[i]
        tokens = preprocess(tw)
        tokenised.append(tokens)
        df.at[i, 'tokens'] = tokens
    
    # save tokens
    print("Saving token data...")
    with open("tokens.txt", "wb") as fp:
        pickle.dump(tokenised, fp)
    print("Done.")
else:
    print("Loading token data...")
    with open("tokens.txt", "rb") as fp:
        tokenised = pickle.load(fp)
    print("Done.")

#%% Build features from clean data
if need_features:
    print("Creating features...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    s_words = nltk.corpus.stopwords.words('english')
    newStopWords = ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'k', 'l',
                    'm', 'n', 'o', 'p', 'r', 's',
                    't', 'u', 'v', 'w', 'x', 'y', 'j', "'"]
    s_words.extend(newStopWords)
    
    # we pass the list of tokens without worrying about sklearn tokenizing for us
    def dummy_fun(doc):
        return doc
    
    tfidf = TfidfVectorizer(
        analyzer='word',
        tokenizer=dummy_fun,
        preprocessor=dummy_fun,
        token_pattern=None,
        stop_words=s_words)
    
    print("Extracting features from tokenized words...")
    X = tfidf.fit_transform(tokenised)
    print("Saving feature matrix...")
    sp.sparse.save_npz("data_mat.npz", X)
    del tokenised
    del tweets
    print('Done. The data matrix has shape {}'.format(X.shape))
else:
    print("Loading feature matrix...")
    X = sp.sparse.load_npz("data_mat.npz")
    print("Done.")

#%% Data Viz - PCA

# We have many dimensions in this data set
# to maximize the effectiveness of manifold based methods
# let us first use PCA to reduce the number of dimensions
# from sklearn.decomposition import PCA

# n_comp = round(0.10 * X.shape[1])
# pca = PCA(n_components=n_comp, whiten=True)
# print("Running Dimension Reduction...")
# Y = pca.fit_transform(X.toarray())
# print("Done.")

#%% Plot SVD
# fig1, ax1 = plt.subplots()
# ax1.plot(Y[:, 0], Y[:, 1], 'o')
# ax1.set_title('Tweets SVD - {:.2f}% of components explain {:.2f}% of variance'.
#               format(100 * n_comp/X.shape[1], 
#                       100 * sum(pca.explained_variance_ratio_)))
# ax1.set_yticklabels([])
# ax1.set_xticklabels([])
# ax1.grid('False')
# plt.savefig("pca_embedding.png", dpi=300)
#%% Data Viz - tSNE

# tsne = TSNE(n_components=2)
# print("\nRunning tSNE...")
# Z = tsne.fit_transform(Y)
# del Y
# print("Done.")

#%% Plot tSNE

# fig2, ax2 = plt.subplots()
# ax2.plot(Z[:, 0], Z[:, 1], 'o')
# ax2.set_title('Tweets - 2D tSNE')
# ax2.set_yticklabels([])
# ax2.set_xticklabels([])
# ax2.grid('False')
# plt.savefig("tnse_embedding.png", dpi=300)

#%% Let's define an autoencoder to do the heavy lifiting since
# SVD and tSNE take way too long

if need_model:
    print("Initializing model...")
    from keras.layers import Input, Dense
    from keras.models import Model
    from keras import regularizers
    from keras.callbacks import EarlyStopping
    
    n_epochs = 10
    batch_size = X.shape[0] // 100
    reg = 1e-4
    split = 0.30 
    pat = 5
    
    N1 = X.shape[1] // 100
    N2 = N1 // 10
    N3 = max(N2 // 10, 10)
    
    # define autoencoder
    input_img= Input(shape=(X.shape[-1],))
    encoded = Dense(units=N1, activation='relu'
                    , kernel_regularizer=regularizers.l2(reg))(input_img)
    encoded = Dense(units=N2, activation='relu'
                    , kernel_regularizer=regularizers.l2(reg))(encoded)
    encoded = Dense(units=N3, activation='relu'
                    , kernel_regularizer=regularizers.l2(reg))(encoded)
    decoded = Dense(units=N2, activation='relu'
                    , kernel_regularizer=regularizers.l2(reg))(encoded)
    decoded = Dense(units=N1, activation='relu'
                    , kernel_regularizer=regularizers.l2(reg))(decoded)
    decoded = Dense(units=X.shape[-1], activation='sigmoid'
                    , kernel_regularizer=regularizers.l2(reg))(decoded)
    autoencoder=Model(input_img, decoded)
    encoder = Model(input_img, encoded)
    
    # compile AE
    autoencoder.compile(optimizer='adam', 
                        loss='binary_crossentropy', 
                        metrics=['accuracy'])
    
    # print summary
    autoencoder.summary()
    print("Done.")

    if not train:
        print("Loading AE weights...")
        autoencoder.load_weights('ae_weights.h5')
        print("Done.")
    else:
        fit_params = {"batch_size": batch_size,
                      "epochs": n_epochs, 
                      "verbose": True,
                      "validation_split": split,
                      "shuffle": True,
                      "callbacks": [EarlyStopping(verbose=True,
                                                  patience=pat,
                                                  monitor='val_loss')]}
            
        print("Training model...")
        history = autoencoder.fit(X, X, **fit_params)
        print("Done.")

#%% Save and plot training info

if train:
        # plot acc and loss
        fig4, ax4 = plt.subplots()
        ax4.plot(history.history['accuracy'], label='train')
        ax4.plot(history.history['val_accuracy'], label='validation')
        ax4.set_title("Model Accuracy")
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.legend()
        plt.savefig("acc_history.png", dpi=300)
        
        fig5, ax5 = plt.subplots()
        ax5.plot(history.history['loss'], label='train')
        ax5.plot(history.history['val_loss'], label='validation')
        ax5.set_title("Model Loss")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig("loss_history.png", dpi=300)
        
        # save data
        print("Saving AE weights...")
        autoencoder.save_weights('ae_weights.h5')
        print("Done.")
        
        print("Saving compressed data...")
        compressed_data = encoder.predict(X)
        np.save("encoded_data.npy", compressed_data)
        print("Done.")
        
#%% K-Means clustering - choosing K part 1

if test_k:
    
    if 'compressed_data' in locals():
        pass
    else:
        print("Loading encoded data...")
        compressed_data = np.load('encoded_data.npy')
        print("Done.")
    
    print("Running inertia analysis...")
    # figure out what K is good for the data set
    km_inertias = []
    
    # determine a reasonable number of K to test
    def num_k_trials(n):
        if n <= 10:
            return n
        elif n > int(4.5e5):
            return round((4.5e5 * 2**((10 - (4.5e5) / ((4.5e5)**0.785)))))
        else:
            return round(n * 2**((10 - n) / n**0.785))
        
    num_k = num_k_trials(compressed_data.shape[0])
    print("{} values of K to test.".format(num_k))
    k_range = range(1, num_k + 1)
    for k in k_range:
        # find clusters for given k
        km = KMeans(n_clusters=k)
        km = km.fit(compressed_data)
        
        # calculate measure of distance
        inertia = km.inertia_
        km_inertias.append(inertia)
        print("k = {} ... inertia = {}".format(k, inertia))
        
    
    km_inertias = np.array(km_inertias)
    print("Done.")
else:
    print("Loading compressed data...")
    compressed_data = np.load('encoded_data.npy')
    print("Done.")
#%% K-Means clustering - choosing K part 2

if test_k:
    # use the kneedle algorithm to find "elbow point"
    kneedle = KneeLocator(k_range, km_inertias, S=1.0,
                          curve='convex', direction='decreasing')
    K = round(kneedle.elbow)
    print("Saving K and K-Means labels...")
    np.save('k_val.npy', K)
    np.save('km_labels.npy', km.labels_)
    print("Done.")
    print("Estimated optimal K = {}".format(K))
    
    # plot results
    kneedle.plot_knee()
    plt.ylabel("K-Means Inertia")
    plt.xlabel("K")
    plt.title("Inertia Plot for K-Means Algorithm")
    plt.savefig("inertia.png", dpi=300)
else:
    print("Loading K and K-Means labels...")
    K = np.load('k_val.npy')
    labels = np.load('km_labels.npy')
    print("Done.")
    
#%% Final analysis with chosen K

# cluster
if cluster and import_data:
    print("Clustering Tweets...")
    km = KMeans(n_clusters=K)
    km = km.fit(compressed_data)
    print("Done")
    
    # add columns to df and save
    df["cluster"] = pd.Series(km.labels_, index=df.index)
    df.to_pickle("labeled_tweets")
else:
    df = pd.read_pickle("labeled_tweets")

#%% Plot 2D grid of all compressed dimensions
if viz:
    from mpl_toolkits.mplot3d import Axes3D
    
    if 'compressed_data' in locals():
        pass
    else:
        print("Loading compressed data...")
        compressed_data = np.load('encoded_data.npy')
        print("Done.")
        
    n_dim = compressed_data.shape[1]
    
    if 'labels' in locals():
        pass
    else:
        print("Loading K-Means labels...")
        labels = np.load('km_labels.npy')
        print("Done.")
    
    # define the colormap
    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    
    # define the bins and no rmalize
    bounds = np.linspace(0, K, K + 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        
    # make the scatterplot
    fig3, ax3 = plt.subplots()
    scat = ax3.scatter(compressed_data[:, 0],
                        compressed_data[:, 5],
                        c=labels, cmap=cmap,
                        norm=norm, s=6)
    # create the colorbar
    cb = plt.colorbar(scat, spacing='proportional', ticks=bounds)
    cb.set_label('Classes')
    ax3.set_title('Tweets - First 3 axes of AE with class labels')
    ax3.set_yticklabels([])
    ax3.set_xticklabels([])
    plt.savefig("AE_embedding_km.png", dpi=300)

print("\nProgram exit.")