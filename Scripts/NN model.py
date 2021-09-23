import tensorflow as tf
from keras.models import Sequential, Model
from keras import backend as K
from keras.utils import to_categorical
from keras.layers import Dense, LSTM, Input, Activation, GRU, Conv1D
from keras.layers import Flatten, TimeDistributed, concatenate, Reshape
from keras.layers import Embedding, Dot, Add, Multiply, Dropout
from keras.optimizers import Adam, Nadam
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from keras.utils import multi_gpu_model

import mlflow
import mlflow.keras

def nn_model(kernel_size, filter_size):
    vocab_size = CHANNELS + 1
  
    ### Channels
    input_layer_e = Input(shape=(CUTOFF,),name='event_input')   
    embedding_event = Embedding(input_dim=vocab_size,output_dim=EMBED_DIM,input_length=CUTOFF,name='sequence_embedding')(input_layer_e)

    ### Devices
    input_layer_d = Input(shape=(CUTOFF,),name='device_input')
    embedding_device = Embedding(input_dim=NO_DEVICES[0]+1,output_dim=EMBED_DIM,input_length=CUTOFF,name='device_embedding')(input_layer_d)
    
    ### City
    input_layer_c = Input(shape=(CUTOFF,),name='city_input')
    embedding_city = Embedding(input_dim=NO_CITIES[0]+1,output_dim=EMBED_DIM,input_length=CUTOFF,name='city_embedding')(input_layer_c)
     
    ### Hits
    hits_input_layer = Input(shape=(CUTOFF,NO_COMMON_HITS,),name='hit_input')
    hit_convolution = Conv1D(filters=filter_size,kernel_size=kernel_size,padding="same",name='hit_conv')(hits_input_layer)
      
    ## Time on site
    tos_input_layer = Input(shape=(CUTOFF,),name='tos_input')
    tos_transformed = Reshape([CUTOFF,1])(tos_input_layer)
    
    ## Visit No
    visitno_input_layer = Input(shape=(CUTOFF,),name='visitno_input')
    visitno_transformed = Reshape([CUTOFF,1])(visitno_input_layer)
    
    emb_vector = [embedding_event]
    if USE_DEVICE_CITY:
      emb_vector.append(embedding_device)
      emb_vector.append(embedding_city)
    if USE_HITS:
      emb_vector.append(hit_convolution)      
    if USE_TOS:
      emb_vector.append(tos_transformed)
    emb_vector.append(visitno_transformed)
      
    input_embeddings = concatenate(emb_vector)
    embedding_phi = Embedding(input_dim=vocab_size,output_dim=1,input_length=CUTOFF,name='phi_embedding')(input_layer_e)
    embedding_mu = Embedding(input_dim=vocab_size,output_dim=1,input_length=CUTOFF,name='mu_embedding')(input_layer_e)
        
    time_input_layer = Input(shape=(CUTOFF,1),name='time_input')
    
    multiply = Multiply(name='multiplication')([embedding_mu,time_input_layer])
    added = Add(name='addition')([embedding_phi,multiply])
    time_attention = Activation(activation='sigmoid',name='attention_activation')(added)
    
    product = Multiply(name='product')([input_embeddings,time_attention])
    
    if USE_TIME:
      lstm_1 = LSTM(LSTM_DIM, return_sequences=True,activation=ACTI)(product)
    else:
      lstm_1 = LSTM(LSTM_DIM, return_sequences=True,activation=ACTI)(input_embeddings)
    drop_1 = Dropout(DROPOUT)(lstm_1) 
    lstm_2 = LSTM(LSTM_DIM, return_sequences=True,activation=ACTI)(drop_1)    
    drop_f = Dropout(DROPOUT)(lstm_2)
    
    input_attention_layer = TimeDistributed(Dense(1))(drop_f)
    attention_output_layer = TimeDistributed(Dense(1, activation='softmax'))(input_attention_layer)
    attention_product = Multiply(name='attention_product')([drop_f, attention_output_layer])

    flattened_output = Flatten()(attention_product)
    output_layer = Dense(1, activation='sigmoid')(flattened_output)
    
    model = Model(inputs=[input_layer_e,input_layer_d,input_layer_c,hits_input_layer,time_input_layer,tos_input_layer,visitno_input_layer],outputs=[output_layer])
    
    opt = Nadam(lr=LEARNING_RATE) #, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    return model
