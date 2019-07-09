# Logistic Regression Model training


# %matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12,8)
import numpy as np
import tensorflow as tf
import keras
import pandas as pd
from keras_tqdm import TQDMNotebookCallback

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Flatten
from keras import optimizers
from keras.preprocessing.sequence import pad_sequences
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
    
    #records holds the array of the tfrecords from the tfrecord file
    records = list(tf.python_io.tf_record_iterator(tfrecord))  
    print("Total audioframes in the dataset:", len(records))
    
    # Make train_set & CV_set    
    records = records[int(start_frac*len(records)):int(end_frac*len(records))]   
    print("After fractioning:")
    print("Total audioframes left:", len(records))

    #This is your train set, rest is CV_set
    rec_len = len(records)  
    
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
            # rest of the secs to make it to 10 sec.
            
            audio_frame += pad
            X.append(audio_frame)

        j += 1
        if j >= num_batches:
            shuffle = np.random.permutation(range(rec_len))
            j = 0

        X = np.array(X)
        yield X, np.array(y)



#Trainer 
def trainer(train_tfrecord, train_lr, train_epochs):

    #Logistic Regression Model 
    # Adam optimizer with Loss = Binary Cross Entropy
    lr_model = Sequential()
    lr_model.add(BatchNormalization(input_shape=(10, 128)))
    lr_model.add(Flatten())
    lr_model.add(Dense(1, activation='sigmoid')) 
    adam = optimizers.Adam(lr=train_lr)
    lr_model.compile(loss = 'binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])


    #Training the model
    # Hyperparameters : Epochs = train_epochs, batch_size = 40, learning rate = train_lr
    batch_size=  40 
    CV_frac = 0.1
    train_gen = data_generator(batch_size, train_tfrecord, 0, 1-CV_frac)
    val_gen = data_generator(20, train_tfrecord, 1-CV_frac, 1)
    rec_len = 17662 
    lr_h = lr_model.fit_generator(train_gen, steps_per_epoch=int(rec_len*(1-CV_frac))//batch_size, epochs=train_epochs,validation_data=val_gen, validation_steps=int(rec_len*CV_frac)//20,verbose=1, callbacks=[TQDMNotebookCallback()])
    
    #Save the model architecture image always at same path
    plot_model(lr_model, to_file='results/LogisticReg/model_LogisticRegression_BinaryCE_Adam_lr={}_Epochs{}.png'.format(train_lr, train_epochs))

    #Save the model at the same path
    lr_model.save('models/LogisticRegression_BinCE_Adam_lr={}_Epochs={}.h5'.format(train_lr,train_epochs))
    return lr_h


#Main function
if __name__ == "__main__":
    
    #setting hyperparameters
    train_path = '../data/preprocessed/bal_gunspotting_in_school_subset.tfrecord'
    epochs = 10
    learning_rate = 0.1

    print("Training Logistic Regression:")
 
    #train logistic regression with learn rate = 0.1 and epochs 10
    lr_h = trainer(train_path, learning_rate, epochs)

    #plotting the Training accuracy, Validation Accuracy for Logistic Regression 
    plt.plot(lr_h.history['acc'], 'o-', label='train_acc')
    plt.plot(lr_h.history['val_acc'], 'x-', label='val_acc')
    plt.xlabel('Epochs({})'.format(epochs), size=20)
    plt.ylabel('Accuracy', size=20)
    plt.legend()
    plt.savefig('results/LogisticReg/LogisticRegression_BinCEAdam_lr={}_Epochs{}.png'.format(learning_rate, epochs), dpi = 300)

    #Printing the Training and Validation metrics - Loss & Accuracy
    print("Epochs = {}".format(epochs))

    print("Average Training loss =", sum(lr_h.history['loss'])/len(lr_h.history['loss']))
    print("Average Training accuracy=", sum(lr_h.history['acc'])/len(lr_h.history['acc'])*100)
    print("Average validation loss =", sum(lr_h.history['val_loss'])/len(lr_h.history['val_loss']))
    print("Average validation accuracy=", sum(lr_h.history['val_acc'])/len(lr_h.history['val_acc'])*100)



