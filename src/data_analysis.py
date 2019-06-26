# Data exploration: Plot the PCA and tSNE 

%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12,8)
import numpy as np
import tensorflow as tf
import keras
import pandas as pd
from keras_tqdm import TQDMNotebookCallback

#Loading the training data subset

records_subset = list(tf.python_io.tf_record_iterator('../data/preprocessed/bal_gunspotting_in_school_subset.tfrecord')) #records holds the array of the tfrecord file
rec_len = len(records_subset)
print(rec_len)


from bitstring import BitArray  #To perform bit manipulation
from keras.preprocessing.sequence import pad_sequences

def datamatrix_multiclass(tfrecord, start_frac=0, end_frac=1):
    '''
    Shuffles the Audioset training data and returns a generator of training data and boolean gunshot and no gunshot labels
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
    
    
    classes = ["gun","fireworks","glass","hammer", "school","background","tools", "multiclass_nogun" ]
    dict_classes = dict.fromkeys(classes)
    
    gun_labels = [426,427,428,429,430,431]
    fireworks_labels = [432,433,434]
    glass_labels = [441,442,443]
    tools_labels = [418,420,421,422,423,424,425]
    background_labels= [506,507,508,509,510,511,512,513,514,515,516,517,518,519,520,521,522,523,524,525,526]
    hammer_labels=[419]
    school_labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
    
    
    while True:
        X = []
        y = []  #add g=[],f=[],b=[],other=[]
        
        for idx in shuffle[0:rec_len]:
            example = records[idx]
            tf_seq_example = tf.train.SequenceExample.FromString(example)
            example_label = list(np.asarray(tf_seq_example.context.feature['labels'].int64_list.value))
            dict_classes = {
                "gun":int(any((True for x in example_label if x in gun_labels))),
                "fireworks":int(any((True for x in example_label if x in fireworks_labels))),
                "glass":int(any(True for x in example_label if x in glass_labels)),
                "hammer":int(any(True for x in example_label if x in hammer_labels)),
                "school":int(any(True for x in example_label if x in school_labels)),
                "tools":int(any(True for x in example_label if x in tools_labels)),
                "background":int(any(True for x in example_label if x in background_labels))
            }
            
            # Class number: gun=0, fireworks=1,glass=2,hammer=3,school=4,tools=5,background=6,multi_nogun=7
            #Handling Mutually exclusive case
            if(sum(dict_classes.values())==1):
                if(dict_classes["gun"]==1):
                    y.append(0)
                if(dict_classes["fireworks"]==1):
                    y.append(1)
                if(dict_classes["glass"]==1):
                    y.append(2)
                if(dict_classes["hammer"]==1):
                    y.append(3)
                if(dict_classes["school"]==1):
                    y.append(4)
                if(dict_classes["tools"]==1):
                    y.append(5)
                if(dict_classes["background"]==1):
                    y.append(6)
               
            #Two classes occuring at once -- class which have low no. of samples data is given high priority  
            if((sum(dict_classes.values())==2)):
                if(dict_classes["gun"] == 1):
                    y.append(0)
                    print("co-occurrence of gun class with someother class")
                elif(dict_classes["hammer"]==1):
                    y.append(3)
                elif(dict_classes["fireworks"]==1):
                    y.append(1)
                elif(dict_classes["glass"]==1):
                    y.append(2)
                elif(dict_classes["tools"]==1):
                    y.append(5)
                elif(dict_classes["school"]==1):
                    y.append(4)
                else:
                    y.append(6) #background
            
              #For >3 class co-occurrence
            if((sum(dict_classes.values())>=3)):
                if(dict_classes["gun"] == 1):
                    y.append(0)
                    print("co-occurrence of gun class with >=3 class")
                else:
                    y.append(7)
               
            n_frames = len(tf_seq_example.feature_lists.feature_list['audio_embedding'].feature)
            audio_frame = []
            for i in range(n_frames):
                audio_frame.append(np.frombuffer(tf_seq_example.feature_lists.feature_list['audio_embedding'].
                                                         feature[i].bytes_list.value[0],np.uint8).astype(np.float32)) # audio_frame gets 128 8 bit numbers on each for loop iteration
            #averaging audio frame
            avg_audioframe = np.mean(audio_frame, axis = 0)
            X.append(avg_audioframe) 


        X = np.array(X)
        print("size of X",len(X))
        print("size of y",len(y))
        return X, np.array(y)


audio_train,labels_train = datamatrix_multiclass('../data/preprocessed/bal_gunspotting_in_school_subset.tfrecord')
print(labels_train[labels_train==0].shape) 
print("Dimension of training matrix:",len(audio_train),"x",len(audio_train[0]))


# PCA 
# import mdtraj as md
from sklearn.decomposition import PCA
import seaborn as sns

#PCA in 2 dimensions
df_pca  = pd.DataFrame(audio_train)
df_pca['y'] = labels_train
pca = PCA(n_components=3)
pca_result = pca.fit_transform(df_pca) #audio_train)
df_pca['pca-one'] = pca_result[:,0]
df_pca['pca-two'] = pca_result[:,1] 
df_pca['pca-three'] = pca_result[:,2]
# Now you have 3 more coloumns on your df_pca dataframe
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

# For reproducability of the results
np.random.seed(42)
rndperm = np.random.permutation(df_pca.shape[0])

colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple","black", "red","tan"]
sns.palplot(sns.xkcd_palette(colors))
plt.figure(figsize=(16,10))
sns.set_style("whitegrid")
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue= "y",
    palette = sns.xkcd_palette(colors),
    data=df_pca.loc[rndperm,:],
    legend="full",
)

plt.savefig("PCA_training_data.png", dpi=400)




#tSNE analysis

#tSNE analysis perplexity = 40, iteration 250
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=250)
tsne_results = tsne.fit_transform(audio_train)
df_tsne = pd.DataFrame()
df_tsne['tsne-2d-one'] = tsne_results[:,0]
df_tsne['tsne-2d-two'] = tsne_results[:,1]
df_tsne['class'] = labels_train
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="class",
    palette = sns.xkcd_palette(colors),
    data=df_tsne,
    legend="full",
#     alpha=0.3
)
plt.savefig("tSNE_iter250_perplexity40.png")
print('tsne plotting finished')


#tSNE analysis perplexity = 50, iteration 250
tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=250)
tsne_results = tsne.fit_transform(audio_train)
df_tsne = pd.DataFrame()
df_tsne['tsne-2d-one'] = tsne_results[:,0]
df_tsne['tsne-2d-two'] = tsne_results[:,1]
df_tsne['class'] = labels_train
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="class",
    palette=sns.color_palette("hls", 8),
    data=df_tsne,
    legend="full",
#     alpha=0.3
)
plt.savefig("tSNE_iter250_perplexity50.png")
print('tsne plotting finished')


#tSNE analysis perplexity = 30, iteration 250
tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=250)
tsne_results = tsne.fit_transform(audio_train)
df_tsne = pd.DataFrame()
df_tsne['tsne-2d-one'] = tsne_results[:,0]
df_tsne['tsne-2d-two'] = tsne_results[:,1]
df_tsne['class'] = labels_train
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="class",
    palette=sns.color_palette("hls", 8),
    data=df_tsne,
    legend="full",
#     alpha=0.3
)
plt.savefig("tSNE_iter250_perplexity30.png")
print('tsne plotting finished')


#tSNE analysis perplexity = 2, iteration 250
tsne = TSNE(n_components=2, verbose=1, perplexity=2, n_iter=250)
tsne_results = tsne.fit_transform(audio_train)
df_tsne = pd.DataFrame()
df_tsne['tsne-2d-one'] = tsne_results[:,0]
df_tsne['tsne-2d-two'] = tsne_results[:,1]
df_tsne['class'] = labels_train
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="class",
    palette=sns.color_palette("hls", 8),
    data=df_tsne,
    legend="full",
#     alpha=0.3
)
plt.savefig("tSNE_iter250_perplexity2.png")
print('tsne plotting finished')


#tSNE analysis perplexity = 5, iteration 250
tsne = TSNE(n_components=2, verbose=1, perplexity=5, n_iter=250)
tsne_results = tsne.fit_transform(audio_train)
df_tsne = pd.DataFrame()
df_tsne['tsne-2d-one'] = tsne_results[:,0]
df_tsne['tsne-2d-two'] = tsne_results[:,1]
df_tsne['class'] = labels_train
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="class",
    palette=sns.color_palette("hls", 8),
    data=df_tsne,
    legend="full",
#     alpha=0.3
)
plt.savefig("tSNE_iter250_perplexity5.png")
print('tsne plotting finished')






