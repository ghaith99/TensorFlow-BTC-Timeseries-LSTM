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
checkpoint = False          #Resume training from a saved checkpoint or restart\

mini = np.min(bitcoin_array)
maxi = np.max(bitcoin_array)
bitcoin_array_normalized = (bitcoin_array-mini)/(maxi-mini)
bitcoin_array_normalized


x = np.zeros([bitcoin_array.size-out_window,in_window])
y = np.zeros([bitcoin_array.size-out_window,out_window])
for i in range(bitcoin_array.size-out_window):
    y[i,:] = bitcoin_array_normalized[i+1:i+1+out_window]
    if i< in_window:
        x[i,-(i+1):] = bitcoin_array_normalized[:i+1]
        
    else :
        x[i,:] = bitcoin_array_normalized[i-in_window+1:i+1]


x_test = x[-300:]
y_test = y[-300:]
x = x[:-300]
y = y[:-300]
x_train = x[:1300]
y_train = y[:1300]
x_valid = x[1300:]
y_valid = y[1300:]
ind_list = np.arange(x_train.shape[0])
shuffle(ind_list)
x_train = x_train[ind_list]
y_train = y_train[ind_list]   
x_valid.shape


def get_batches(x,y, batch_size):
    n_batches = x.shape[0]//batch_size
    for i in range(0,n_batches*batch_size,batch_size):
        x_ = np.expand_dims(x[i:i+batch_size],2)
        y_ = np.expand_dims(y[i:i+batch_size],2)
        yield x_,y_
    

def build_inputs(input_length = in_window, output_length = out_window):
    
    enc_inp = [
        tf.placeholder(tf.float32, shape=(None, 1), name="input{}".format(t))  
      for t in range(input_length)
    ]

    targets = [tf.placeholder(tf.float32, shape=(None, 1), name="target{}".format(t))
          for t in range(output_length) ]
    
    dec_inp = [ tf.zeros_like(enc_inp[0], dtype=np.float32, name="GO") for t in range(output_length)] 
    return enc_inp, targets, dec_inp

def build_lstm(lstm_size, num_layers, batch_size, keep_prob):
    
    def build_cell(lstm_size, keep_prob):
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        return drop
    
    cell = tf.contrib.rnn.MultiRNNCell([build_cell(lstm_size, keep_prob) for _ in range(num_layers)])
    return cell


def build_loss(reshaped_outputs, expected_sparse_output, l2_reg):
    with tf.variable_scope('Loss'):

        output_loss = tf.losses.mean_squared_error(expected_sparse_output,reshaped_outputs )
        reg_loss = 0
        for tf_var in tf.trainable_variables():
            if not ("Bias" in tf_var.name or "Output_" in tf_var.name):
                reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

        loss = output_loss + l2_reg * reg_loss
        return loss
    
    

def build_optimizer(loss, learning_rate, grad_clip):
    tvars = tf.trainable_variables() 
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip) 
    train_op = tf.train.AdamOptimizer(learning_rate) 
    optimizer = train_op.apply_gradients(zip(grads, tvars))
    
    return optimizer



tf.reset_default_graph()
#sess.close()
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
enc_inp, targets, dec_inp = build_inputs()     
cell = build_lstm(lstm_size, num_layers, batch_size, keep_prob)  
dec_outputs, dec_memory = tf.contrib.legacy_seq2seq.basic_rnn_seq2seq(enc_inp, dec_inp,  cell)
outputs = [tf.contrib.layers.fully_connected(lstm_output, 1 , 
                                             activation_fn = None) for lstm_output in dec_outputs]
loss = build_loss(outputs, targets, l2_reg) 
optimizer = build_optimizer(loss, learning_rate, grad_clip)  

saver = tf.train.Saver()
if checkpoint:
    saver.restore(sess, path)
    print("Checkpoint restored")
else:              
    tf.global_variables_initializer().run()


counter = 0
for e in range(epochs):
    train_y = []
    train_losses = []
    train_outputs = []
    for x_in, y_in in get_batches(x_train, y_train, batch_size):
        feed_dict = {enc_inp[t]: x_in[:,t] for t in range(len(enc_inp))} 
        feed_dict.update({targets[t]: y_in[:,t] for t in range(len(targets))})
        target, outputs_, loss_, _ = sess.run([targets , outputs, loss, optimizer], 
                                                   feed_dict = feed_dict) 
        train_losses.append(loss_)
        train_outputs.append(outputs_)
        train_y.append(target)
        counter += 1
        if (counter % 200 == 0):
            saver.save(sess, path)
            
        
    val_pred =[]
    val_y = []
    val_losses = []
        
    for x_in, y_in in get_batches(x_valid, y_valid, batch_size):
        feed_dict = {enc_inp[t]: x_in[:,t] for t in range(len(enc_inp))} 
        feed_dict.update({targets[t]: y_in[:,t] for t in range(len(targets))})
        target, outputs_, loss_  = sess.run([targets, outputs, loss], feed_dict = feed_dict) 
        val_y.append(target)
        val_pred.append(outputs_)
        val_losses.append(loss_)
    print("==================================================================") 
    print("Epoch {}/{}: Training loss: {}....Validation loss: {}".format(e+1, epochs, np.mean(train_losses),
                                                                           np.mean(val_losses)))
    #print("Final test loss : {:.8f}".format(np.mean(test_losses))) 
    
test_pred = []
test_y = []
test_losses = []
for x_in, y_in in get_batches(x_test, y_test, batch_size):
    feed_dict = {enc_inp[t]: x_in[:,t] for t in range(len(enc_inp))} 
    feed_dict.update({targets[t]: y_in[:,t] for t in range(len(targets))})
    target, outputs_, loss_  = sess.run([targets, outputs, loss], feed_dict = feed_dict) 
    test_y.append(target)
    test_pred.append(outputs_)
    test_losses.append(loss_)
            
print("==================================================================")    
print("Final test loss : {:.8f}".format(np.mean(test_losses)))    
             

sess.close()
