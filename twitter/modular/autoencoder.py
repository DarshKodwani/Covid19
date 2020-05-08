# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 20:46:18 2020

@author: tharshi
"""


#%% Import Frameworks and Set-up

# import useful frameworks
import numpy as np
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Dropout, ELU
from keras import regularizers, optimizers
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, Callback

import itertools
fr_it = itertools.chain.from_iterable
from itertools import repeat

# make pretty
plt.style.use('seaborn')

# reproducible results
np.random.seed(27)

X = np.load('features.npy')

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
    drop_pct = 0.01
    
    # unit counts    
    dims = [50, 25, 12, 6, 3, 2]

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
    
    autoencoder.compile(optimizer='sgd',
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
    pat = 100

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

    print('Saving encoding...', end='')
    np.save('encoded.npy', X_en)
    print('Done.')

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