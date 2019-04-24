import pandas as pd
import numpy as np
import tensorflow as tf
import datetime
from datetime import datetime as time
from random import shuffle


"""Import bitoin price dataset from https://coinmarketcap.com"""
now = time.now()
bitcoin_market_info = pd.read_html("https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130428&end="+time.strftime(now, "%Y%m%d"))[0]
bitcoin_market_info = bitcoin_market_info.assign(Date=pd.to_datetime(bitcoin_market_info['Date']))
bitcoin_market_info.loc[bitcoin_market_info['Volume']=="-",'Volume']=0
bitcoin_market_info['Volume'] = bitcoin_market_info['Volume'].astype('int64')
bitcoin_market_info.columns = bitcoin_market_info.columns.str.replace("*", "")
bitcoin_market_info.head()


bitcoin_array = bitcoin_market_info["Close"].as_matrix(columns = None) 
#Bitcoin closing prices as a numpy array


#%%
"""Model hyperparameters 
"""
epochs = 100               # Complete passes over the training data
in_window = 100             #Input sequence length
out_window = 10           #Number of prediction steps
lstm_size = 256            #Number of LSTM units 
batch_size = 30            #Minibatch size
num_layers = 2             #Number of LSTM layers
keep_prob = 0.5           #Dropout probability
learning_rate = 0.0001
grad_clip = 5              #gradient clip to avoid exploding gradients
l2_reg = 2e-6            #l2 regularization to combat overfitting
path = "model/bitcoin_lstm" #Path to save model
checkpoint = False          #Resume training from a saved checkpoint or restart