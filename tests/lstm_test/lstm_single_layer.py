#LSTM Single Layer 

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12,8)
import numpy as np
import tensorflow as tf
import keras
import pandas as pd
from keras_tqdm import TQDMNotebookCallback

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.layers import LSTM
from keras import regularizers
from keras.utils import plot_model

def data_generator(batch_size, tfrecord, start_frac=0, end_frac=1):
    '''
    Shuffles the Audioset training data and returns a generator of training data and boolean gunshot labels
    batch_size: batch size for each set of training data and labels
    tfrecord: filestring of the tfrecord file to train on
    start_frac: the starting point of the data set to use, as a fraction of total record length (used for CV)
    end_frac: the ending point of the data set to use, as a fraction of total record length (used for CV)
    '''
    max_len=10

    #tfrecord holds data in binary sequence string. 
    records = list(tf.python_io.tf_record_iterator(tfrecord))  #records holds the array of the tfrecord file
    print("Total audioframes in training dataset:", len(records))
        
    records = records[int(start_frac*len(records)):int(end_frac*len(records))]  # Make train_set & CV_set 
    print("After fractioning:")
    print("Total audioframes in training dataset:", len(records))
    
    rec_len = len(records)  # this is your train set, rest is CV_set
    
    shuffle = np.random.permutation(range(rec_len))
    num_batches = rec_len//batch_size - 1                      
    j = 0
    
    gun_labels = [426,427,428,429,430,431]
    
    while True:
        X = []
        y = []
        for idx in shuffle[j*batch_size:(j+1)*batch_size]:
            example = records[idx]
            tf_seq_example = tf.train.SequenceExample.FromString(example)
            example_label = list(np.asarray(tf_seq_example.context.feature['labels'].int64_list.value))
            value_x = any((True for x in example_label if x in gun_labels))
            if(value_x==True):
                y.append(1)
            else:
                y.append(0)
                
            n_frames = len(tf_seq_example.feature_lists.feature_list['audio_embedding'].feature)
            audio_frame = []
            for i in range(n_frames):
                # audio_frame gets 128 8 bit numbers on each for loop iteration
                audio_frame.append(np.frombuffer(tf_seq_example.feature_lists.feature_list['audio_embedding'].
                                                         feature[i].bytes_list.value[0],np.uint8).astype(np.float32)) 

            pad = [np.zeros([128], np.float32) for i in range(max_len-n_frames)] 
            # if clip is less than 10 sec, audio_frame is padded with zeros for 
            #rest of the secs to make it to 10 sec.
            
            audio_frame += pad
            X.append(audio_frame)

        j += 1
        if j >= num_batches:
            shuffle = np.random.permutation(range(rec_len))
            j = 0

        X = np.array(X)
        yield X, np.array(y)



#Trainer 
def lstm_trainer(train_tfrecord, train_lr, train_epochs):
    # Building the model
    lstm_model = Sequential()
    lstm_model.add(BatchNormalization(input_shape=(None, 128)))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(LSTM(128, activation='relu',
            kernel_regularizer=regularizers.l2(0.01),
            activity_regularizer=regularizers.l2(0.01)))
    lstm_model.add(Dense(1, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    lstm_model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])


    # Fitting the LSTM model
    batch_size=32
    CV_frac = 0.1
    train_gen = data_generator(batch_size, train_tfrecord, 0, 1-CV_frac)
    val_gen = data_generator(20,train_tfrecord, 1-CV_frac, 1)
    rec_len = 17662
    lstm_h = lstm_model.fit_generator(train_gen,steps_per_epoch=int(rec_len*(1-CV_frac))//batch_size, epochs=train_epochs, 
                           validation_data=val_gen, validation_steps=int(rec_len*CV_frac)//20,
                           verbose=1, callbacks=[TQDMNotebookCallback()])
    
    # Plot the model architecture
    plot_model(lstm_model, to_file='LSTM_Loss=BinCE_Epochs{}_lt={}.png'.format(train_epochs,train_lr))

    # Save the lstm model
    lstm_model.save('1LayerLSTM__Loss=BinCE_lr={}_Epochs={}.h5'.format(train_lr, train_epochs))
    return lstm_h


#Main function
if __name__ == "__main__":
    #setting hyperparameters
    train_path = '../../data/preprocessed/bal_gunspotting_in_school_subset.tfrecord'
    epochs = 2 #25
    learning_rate = 3 #0.001     #0.001 lr is the trick that works. 

    print("Training Logistic Regression:")
 
    #train logistic regression with learn rate = 0.1 and epochs 10
    lstm_h = lstm_trainer(train_path, learning_rate, epochs)    

    #Plotting the training performance of the LSTM
    plt.plot(lstm_h.history['acc'], 'o-', label='train_acc')
    plt.plot(lstm_h.history['val_acc'], 'x-', label='val_acc')
    plt.xlabel('Epochs({})'.format(epochs), size=20)
    plt.ylabel('Accuracy', size=20)
    plt.legend()
    plt.savefig('LSTM__Loss=BinCE_{}Epochs_lr={}_performance.png'.format(epochs,learning_rate), dpi = 300)


    # Metrics for the LSTM
    print("Epochs = {}".format(epochs))
    print("Average Training loss =", sum(lstm_h.history['loss'])/len(lstm_h.history['loss']))
    print("Average Training accuracy=", sum(lstm_h.history['acc'])/len(lstm_h.history['acc'])*100)
    print("Average validation loss =", sum(lstm_h.history['val_loss'])/len(lstm_h.history['val_loss']))
    print("Average validation accuracy=", sum(lstm_h.history['val_acc'])/len(lstm_h.history['val_acc'])*100)




