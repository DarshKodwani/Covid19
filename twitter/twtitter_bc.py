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
import_data = False
need_tokens = False
need_features = False
need_model = True
train = True
test_k = True
cluster = True
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
    
    df = df.sample(frac=1)
    
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
    with open("tokens.txt", "wb") as fp:
        pickle.dump(tokenised, fp)
    
    print("Done.")
else:
    with open("tokens.txt", "rb") as fp:
        tokenised = pickle.load(fp)

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
    sp.sparse.save_npz("data_mat.npz", X)
    del tokenised
    del tweets
    print('Done. The data matrix has shape {}'.format(X.shape))
else:
    X = sp.sparse.load_npz("data_mat.npz")

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
# plt.savefig("pca_embedding.png", dpi=600)
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
# plt.savefig("tnse_embedding.png", dpi=600)

#%% Let's define an autoencoder to do the heavy lifiting since
# SVD and tSNE take way too long

if need_model:
    print("Initializing model...")
    from keras.layers import Input, Dense
    from keras.models import Model
    
    n_epochs = 10
    batch_size = X.shape[0] // 100
    N1 = X.shape[1] // 100
    N2 = max(N1 // 10, 100)
    N3 = max(N2 // 10, 10)
    
    # define autoencoder
    input_img= Input(shape=(X.shape[-1],))
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
    print("Done.")

    if not train:
        autoencoder.load_weights('ae_weights.h5')
    else:
        autoencoder.fit(X, X, 
                        epochs=n_epochs, 
                        batch_size=batch_size, 
                        shuffle=True)
        autoencoder.save_weights('ae_weights.h5')
    

#%% K-Means clustering - choosing K part 1

if test_k:
    
    compressed_data = encoder.predict(X)
    np.save('comp_data.npy', compressed_data)
    
    print("Running inertia analysis...")
    # figure out what K is good for the data set
    km_inertias = []
    
    num_k = max(round(0.01 * compressed_data.shape[0]), 10)
    print("{} values of K to test.".format(num_k))
    k_range = range(1, num_k)
    for k in k_range:
        # find clusters for given k
        km = KMeans(n_clusters=k)
        km = km.fit(compressed_data)
        
        # calculate measure of distance
        inertia = km.inertia_
        km_inertias.append(inertia)
        print("k = {} ... inertia = {}".format(k, inertia))

    km_inertias = np.array(km_inertias)
else:
    compressed_data = np.load('comp_data.npy')

#%% K-Means clustering - choosing K part 2

if test_k:
    # use the kneedle algorithm to find "elbow point"
    kneedle = KneeLocator(k_range, km_inertias, S=1.0,
                          curve='convex', direction='decreasing')
    K = round(kneedle.elbow)
    np.save('k_val.npy', K)
    np.save('km_labels.npy', km.labels_)
    print("Done.")
    print("Estimated optimal K = {}".format(K))
    
    # plot results
    kneedle.plot_knee()
    plt.ylabel("K-Means Inertia")
    plt.xlabel("K")
    plt.title("Inertia Plot for K-Means Algorithm")
    plt.savefig("inertia.png", dpi=600)
else:
    K = np.load('k_val.npy')
    labels = np.load('km_labels.npy')
    
#%% Final analysis with chosen K

# cluster
if cluster:
    print("Clustering Tweets...")
    km = KMeans(n_clusters=K)
    km = km.fit(compressed_data)
    print("Done")
    
    # add columns to df and save
    df["cluster"] = pd.Series(km.labels_, index=df.index)
    df.to_pickle("labeled_tweets")
else:
    df = pd.read_pickle("labeled_tweets")

if viz:
    from mpl_toolkits.mplot3d import Axes3D
    import itertools
    
    if 'compressed_data' in locals():
        Z = compressed_data
    else:
        compressed_data = np.load('comp_data.npy')
        
    n_dim = Z.shape[1]
    
    if 'labels' in locals():
        pass
    else:
        labels = np.load('km_labels.npy')
    
    for comb in itertools.combinations(range(n_dim), 3):
        print(comb)
    
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
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, projection='3d')
    scat = ax3.scatter(Z[:, 0], Z[:, 1], Z[:, 2], 
                       c=labels, cmap=cmap,
                       norm=norm, s=6)
    # create the colorbar
    cb = plt.colorbar(scat, spacing='proportional', ticks=bounds)
    cb.set_label('Classes')
    ax3.set_title('Tweets - First 3 axes of AE with class labels')
    ax3.set_yticklabels([])
    ax3.set_xticklabels([])
    plt.savefig("AE_embedding_km.png", dpi=600)

print("\nProgram exit.")