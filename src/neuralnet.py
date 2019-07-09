# Single layer NN classifier 

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12,8)
import numpy as np
import tensorflow as tf
import keras
import pandas as pd
from keras_tqdm import TQDMNotebookCallback
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Flatten
from keras.regularizers import l2
from keras.utils import plot_model      

# data generator function which creates a numpy array of audioframes (X) and corresponding labels (y)
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
    if(tfrecord == '../data/preprocessed/bal_gunspotting_in_school_subset.tfrecord'):
        print("Total audioframes in training dataset:", len(records))
    elif(tfrecord == '../data/preprocessed/eval_gunspotting_in_school_subset.tfrecord'):
        print("Total audioframes in eval dataset:", len(records))
        
    records = records[int(start_frac*len(records)):int(end_frac*len(records))]  # Make train_set & CV_set 
    print("After fractioning:")
    if(tfrecord == '../data/preprocessed/bal_gunspotting_in_school_subset.tfrecord'):
        print("Total audioframes in training dataset:", len(records))
    elif(tfrecord == '../data/preprocessed/eval_gunspotting_in_school_subset.tfrecord'):
        print("Total audioframes in eval dataset:", len(records))
    
    rec_len = len(records)  # this is your train set, rest is CV_set
    
    shuffle = np.random.permutation(range(rec_len))
    num_batches = rec_len//batch_size - 1                      
    j = 0
    
    gun_labels = [426,427,428,429,430,431]
    
    while True:
        X = []
        y = []  #add g=[],f=[],b=[],other=[]
        for idx in shuffle[j*batch_size:(j+1)*batch_size]:
            example = records[idx]
            tf_seq_example = tf.train.SequenceExample.FromString(example)
            example_label = list(np.asarray(tf_seq_example.context.feature['labels'].int64_list.value))
            value_x = any((True for x in example_label if x in gun_labels)) #add f_bin, b_bin, other_bin
            if(value_x==True):
                y.append(1)      #[1,1,0,1,0...] 
            else:
                y.append(0)
                
            n_frames = len(tf_seq_example.feature_lists.feature_list['audio_embedding'].feature)
            audio_frame = []
            for i in range(n_frames):
                audio_frame.append(np.frombuffer(tf_seq_example.feature_lists.feature_list['audio_embedding'].
                                                         feature[i].bytes_list.value[0],np.uint8).astype(np.float32)) # audio_frame gets 128 8 bit numbers on each for loop iteration
            pad = [np.zeros([128], np.float32) for i in range(max_len-n_frames)] 
            # if clip is less than 10 sec, audio_frame is padded with zeros for 
            #rest of the secs to make it to 10 sec.
            
            audio_frame += pad
            X.append(audio_frame) #eg: X[5] will output 5th audioframe 

        j += 1
        if j >= num_batches:
            shuffle = np.random.permutation(range(rec_len))
            j = 0

        X = np.array(X)
        yield X, np.array(y)


def main():
    adam = optimizers.Adam(lr=0.4)

    nn_model = Sequential()
    nn_model.add(BatchNormalization(input_shape=(10, 128)))
    nn_model.add(Flatten())
    # nn_model.add(Dense(100, activation='relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    nn_model.add(Dense(100, activation='relu'))
    nn_model.add(Dense(1, activation='sigmoid'))                 
    nn_model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])


    batch_size=  40 
    CV_frac = 0.1
    train_gen = data_generator(batch_size,'../data/preprocessed/bal_gunspotting_in_school_subset.tfrecord', 0, 1-CV_frac)
    val_gen = data_generator(20,'../data/preprocessed/bal_gunspotting_in_school_subset.tfrecord', 1-CV_frac, 1)
    rec_len = 17662 
    nn_h = nn_model.fit_generator(train_gen,steps_per_epoch=int(rec_len*(1-CV_frac))//batch_size, epochs=20,validation_data=val_gen, validation_steps=int(rec_len*CV_frac)//20,verbose=1, callbacks=[TQDMNotebookCallback()])

    #Plot the training performance graphs for the NN classifier
    plt.plot(nn_h.history['acc'], 'o-', label='train_acc')
    plt.plot(nn_h.history['val_acc'], 'x-', label='val_acc')
    plt.xlabel('Epochs(50)', size=20)
    plt.ylabel('Accuracy', size=20)
    plt.legend()
    plt.savefig('../results/training_results/NeuralNet/SingleLayerNN__Loss=BinaryCE_50Epochs_lr0.4performance.png', dpi = 300)

    print("Epochs = 20")
    print("val_loss length:",len(nn_h.history['val_loss']))
    print("val_acc length:",len(nn_h.history['val_acc']))
    print("loss length:",len(nn_h.history['loss']))
    print("acc length:",len(nn_h.history['acc']))

    print("Average Training loss =", sum(nn_h.history['loss'])/len(nn_h.history['loss']))
    print("Average Training accuracy=", sum(nn_h.history['acc'])/len(nn_h.history['acc'])*100)
    print("Average validation loss =", sum(nn_h.history['val_loss'])/len(nn_h.history['val_loss']))
    print("Average validation accuracy=", sum(nn_h.history['val_acc'])/len(nn_h.history['val_acc'])*100)
    plot_model(nn_model, to_file='../results/training_results/NeuralNet/SingleLayerNN__Loss=BinaryCE_50Epochs_lr0.4Model.png')


if __name__ == "__main__":
    main()