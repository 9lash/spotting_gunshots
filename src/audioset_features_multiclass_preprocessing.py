#Date: June 17, 2019
# This code takes the audioset dataset and explores the dataset csv files and audio_embeddings. 
# It creates a balanced training dataset containing 22832 [5708 x4 classes]  audio embedding samples of gunshot, fireworks, background 
# and other classes out of the total dataset provided by Google.
# It also creates a balanced evaluation dataset containing 644 [161 x4 classes] audio embedding samples of gunshot, fireworks, background
# and other classes out of the total eval dataset.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import glob

import seaborn as sns

#read the unbalanced train dataset csv file & gunshot_labels.csv file 
labels = pd.read_csv('../data/raw/unbalanced_train_segments.csv',header=2, quotechar=r'"',skipinitialspace=True)

# There are multiple labels like artillery gun, shot gun qualify for gun. All of those classes are 
# put together into gun_class. Likewise for fireworks, background and other class

gunshot_labels = pd.read_csv('../data/raw/classes/gunshot_labels.csv',names=['num','label','description'])
g_str = '|'.join(gunshot_labels['label'].values)
print(g_str)

firework_labels = pd.read_csv('../data/raw/classes/fireworks.csv', names = ['num', 'label', 'description'])
#print(firework_labels)
f_str = '|'.join(firework_labels['label'].values)
print(f_str)

background_labels = pd.read_csv('../data/raw/classes/background.csv', names = ['num', 'label', 'description'])
#print(background_labels)
b_str = '|'.join(background_labels['label'].values)
print(b_str)

#create a one hot encoded labelling for 4 classes
labels['fireworks'] = labels['positive_labels'].str.contains(f_str) & ~labels['positive_labels'].str.contains(g_str) & ~labels['positive_labels'].str.contains(b_str) 

labels['gunshots'] = labels['positive_labels'].str.contains(g_str) & ~labels['positive_labels'].str.contains(f_str) & ~labels['positive_labels'].str.contains(b_str)

labels['background'] = labels['positive_labels'].str.contains(b_str) & ~labels['positive_labels'].str.contains(g_str) & ~labels['positive_labels'].str.contains(f_str)

labels['other'] = True
labels['other'] = labels['other'] & ~labels['positive_labels'].str.contains(b_str) & ~labels['positive_labels'].str.contains(g_str) & ~labels['positive_labels'].str.contains(f_str)

#save this unbalanced training one hot encoded dataset to csv file. 
labels.to_csv(r'../data/preprocessed/unbalanced_train_tagged.csv')


#Checking out whats the distribution of the classes in the training dataset
labels = pd.read_csv(r'../data/raw/unbalanced_train_segments.csv', header = 2, quotechar = r'"', skipinitialspace=True)
labels['gunshots'] = labels['positive_labels'].str.contains(g_str)
labels['fireworks'] = labels['positive_labels'].str.contains(f_str)
labels['background'] = labels['positive_labels'].str.contains(b_str)
labels['other'] = True
labels['other'] = labels['other'] & ~labels['positive_labels'].str.contains(b_str) & ~labels['positive_labels'].str.contains(g_str) & ~labels['positive_labels'].str.contains(f_str)

gun_positive = labels[labels['gunshots']==True]
fireworks_positive = labels[labels['fireworks']==True]
background_positive = labels[labels['background']== True]
other_positive = labels[labels['other']==True]

d = {'Class' : ['Gunshots','Fireworks','Background','Other'], 'Total_samples' : [gun_positive.shape[0],fireworks_positive.shape[0],background_positive.shape[0],other_positive.shape[0]]}
train_df = pd.DataFrame(data=d)
train_df


#Checking out whats the distribution of the classes in the evaluation dataset
labels = pd.read_csv(r'../data/raw/eval_segments.csv', header = 2, quotechar = r'"', skipinitialspace=True)
labels['gunshots'] = labels['positive_labels'].str.contains(g_str)
labels['fireworks'] = labels['positive_labels'].str.contains(f_str)
labels['background'] = labels['positive_labels'].str.contains(b_str)
labels['other'] = True
labels['other'] = labels['other'] & ~labels['positive_labels'].str.contains(b_str) & ~labels['positive_labels'].str.contains(g_str) & ~labels['positive_labels'].str.contains(f_str)

gun_positive = labels[labels['gunshots']==True]
fireworks_positive = labels[labels['fireworks']==True]
background_positive = labels[labels['background']== True]
other_positive = labels[labels['other']==True]

d = {'Class' : ['Gunshots','Fireworks','Background','Other'], 'Total_samples' : [gun_positive.shape[0],fireworks_positive.shape[0],background_positive.shape[0],other_positive.shape[0]]}
eval_df = pd.DataFrame(data=d)
eval_df

sns.set(style="whitegrid")
sns.barplot(x='Class', y='Total_samples', data=eval_df.reset_index())


#Creating a balanced evaluation or Validation dataset. Make sure the validation set are equal and randomly sampled.
#output: tfrecord file which has YTID, start, end and positive labels of gun, fireworks, background and other.
#Name of the tfrecord file : eval_spotting_gunshots_subset.tfrecord
import os

labels = pd.read_csv('../data/raw/eval_segments.csv', header = 2, quotechar = r'"', skipinitialspace=True)
labels['gunshots'] = labels['positive_labels'].str.contains(g_str)
labels['fireworks'] = labels['positive_labels'].str.contains(f_str)
labels['background'] = labels['positive_labels'].str.contains(b_str)
labels['other'] = True
labels['other'] = labels['other'] & ~labels['positive_labels'].str.contains(b_str) & ~labels['positive_labels'].str.contains(g_str) & ~labels['positive_labels'].str.contains(f_str)

# gun_positive = labels[labels['gunshots']==True]
# fireworks_positive = labels[labels['fireworks']==True]
# background_positive = labels[labels['background']== True]
# other_positive = labels[labels['other']==True]
# print("gun samples:")
# print(gun_positive.shape[0])
# print("fireworks samples")
# print(fireworks_positive.shape[0])
# print("background samples")
# print(background_positive.shape[0])
# print("other samples")
# print(other_positive.shape[0])

# We found that there are 292 gun samples, 161 fireworks, 2053 background, 17890 Other  class in Evaluation set. 
# Lets make an evaluation dataset of 161 samples of each class. 

fireworks_positive = labels[labels['fireworks']==True]
gun_positive = labels[labels['gunshots']==True].sample(fireworks_positive.shape[0])
background_positive = labels[labels['background']==True].sample(fireworks_positive.shape[0])
other_positive = labels[labels['other']==True].sample(fireworks_positive.shape[0])

subset = gun_positive.append(fireworks_positive, ignore_index=True)
subset = subset.append(background_positive, ignore_index=True)
subset = subset.append(other_positive,ignore_index=True)


print(subset.shape[0])
#eval_df['first2'] = eval_df['# YTID'].str[:2]

subset['first2'] = subset['# YTID'].str[:2]
sorted_subset = subset.sort_values(by=['first2'])
print(sorted_subset)
print('length of the subset is ', len(sorted_subset))

raw_dir = '../data/raw/audioset_v1_embeddings/eval'
writer = tf.python_io.TFRecordWriter('../data/preprocessed/eval_spotting_gunshots_subset.tfrecord')
i=0 
while i<len(sorted_subset):
    first2 = sorted_subset['first2'][i]
    ytids = set([sorted_subset['# YTID'][i]])
    while sorted_subset['first2'][i+1] == first2:
        i += 1
        ytids.add(sorted_subset['# YTID'][i])
    if os.path.exists(f'{raw_dir}/{first2}.tfrecord'):
        fn = f'{raw_dir}/{first2}.tfrecord'
    else:
        fn = f'{raw_dir}/{first2}-1.tfrecord'
    for example in tf.python_io.tf_record_iterator(fn):
        tf_example = tf.train.Example.FromString(example)
        if tf_example.features.feature['video_id'].bytes_list.value[0].decode(encoding='UTF-8'):
            writer.write(example)
            # Add the whole details in a new tfrecord file
            i=i+1
writer.close()


# Now open the unbalanced_train data and create a balanced training dataset
#output: tfrecord file which has YTID, start, end and positive labels of gun, fireworks, background and other. 
# This tfrecord file  would contain 22832 audioembeddings belonging to all 4 classes. Each class of size 5708 samples. 

from tqdm import tqdm 

labels = pd.read_csv('../data/raw/unbalanced_train_segments.csv',header=2, quotechar=r'"',skipinitialspace=True)
print(labels.shape[0])

labels['gunshots'] = labels['positive_labels'].str.contains(g_str)
labels['fireworks'] = labels['positive_labels'].str.contains(f_str)
labels['background'] = labels['positive_labels'].str.contains(b_str)
labels['other'] = True
labels['other'] = labels['other'] & ~labels['positive_labels'].str.contains(b_str) & ~labels['positive_labels'].str.contains(g_str) & ~labels['positive_labels'].str.contains(f_str)

fireworks_positive = labels[labels['fireworks']==True]
print("Fireworks sample: ",fireworks_positive.shape[0])
gun_positive = labels[labels['gunshots']==True].sample(fireworks_positive.shape[0])
print("gun sample: ",gun_positive.shape[0])
background_positive = labels[labels['background']==True].sample(fireworks_positive.shape[0])
print("Background sample: ",background_positive.shape[0])
other_positive = labels[labels['other']==True].sample(fireworks_positive.shape[0])
print("other sample: ",other_positive.shape[0])

#firework samples = 5708, gunshot samples = 8831 , background_samples = 194974, other = 1833813
#There is a strong class imbalance in this problem. Therefore, for now,we will handle this class
#imbalance by making sure all classes is of size fireworks. 
# Name of the tfrecord file : unbaltrain_spotting_gunshots_subset.tfrecord

subset = gun_positive.append(fireworks_positive, ignore_index=True)
subset = subset.append(background_positive, ignore_index=True)
subset = subset.append(other_positive,ignore_index=True)

print(subset)
print(subset.shape[0])

subset['first2'] = subset['# YTID'].str[:2]
sorted_subset = subset.sort_values(by=['first2'])
print(sorted_subset)
print('length of the subset is ', len(sorted_subset))

raw_dir = '../data/raw/audioset_v1_embeddings/unbal_train'
writer = tf.python_io.TFRecordWriter('../data/preprocessed/unbaltrain_spotting_gunshots_subset.tfrecord')
i=0 
while i<len(sorted_subset):
    first2 = sorted_subset['first2'][i]
    ytids = set([sorted_subset['# YTID'][i]])
    while sorted_subset['first2'][i+1] == first2:
        i += 1
        ytids.add(sorted_subset['# YTID'][i])
    if os.path.exists(f'{raw_dir}/{first2}.tfrecord'):
        fn = f'{raw_dir}/{first2}.tfrecord'
    else:
        fn = f'{raw_dir}/{first2}-1.tfrecord'
    for example in tf.python_io.tf_record_iterator(fn):
        tf_example = tf.train.Example.FromString(example)
        if tf_example.features.feature['video_id'].bytes_list.value[0].decode(encoding='UTF-8'):
            writer.write(example)  # Add the whole details in a new tfrecord file
            i=i+1
writer.close()


