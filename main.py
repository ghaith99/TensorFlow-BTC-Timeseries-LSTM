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