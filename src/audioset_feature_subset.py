# Audiofeatures preprocessing 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import glob

#!pip install seaborn
import seaborn as sns

import os
from tqdm import tqdm   #for displaying the progress of training 

labels = pd.read_csv('../data/raw/unbalanced_train_segments.csv',header=2, quotechar=r'"',skipinitialspace=True)
gunshot_labels = pd.read_csv('../data/raw/classes/gunshot_labels.csv',names=['num','label','description'])
g_str = '|'.join(gunshot_labels['label'].values)
no_gun_labels = pd.read_csv('../data/raw/classes/no_gunshot.csv', names = ['num', 'label', 'description'])
ng_str = '|'.join(no_gun_labels['label'].values)
labels['no_gun'] = labels['positive_labels'].str.contains(ng_str) & ~labels['positive_labels'].str.contains(g_str)
labels['gunshots'] = labels['positive_labels'].str.contains(g_str) 
labels.to_csv(r'../data/preprocessed/unbalanced_train_tagged.csv')


#Checking out whats the distribution of the classes in the training dataset
print("Analyzing the distribution of classes in Training dataset")
labels = pd.read_csv(r'../data/raw/unbalanced_train_segments.csv', header = 2, quotechar = r'"', skipinitialspace=True)
labels['gunshots'] = labels['positive_labels'].str.contains(g_str)
labels['no_gun'] = labels['positive_labels'].str.contains(ng_str) & ~labels['positive_labels'].str.contains(g_str)
gun_positive = labels[labels['gunshots']==True]
no_gun_positive = labels[labels['no_gun']==True]
d = {'Class' : ['gunshots','no_gun'], 'Total_samples' : [gun_positive.shape[0],no_gun_positive.shape[0]]}
train_df = pd.DataFrame(data=d)
train_df.to_csv(r'training_distribution.csv')
print("...")
print("training_distribution.png saved in disk")

#Checking out whats the distribution of the classes in the evaluation dataset
print("Analyzing the distribution of the classes in Eval dataset")
labels = pd.read_csv(r'../data/raw/eval_segments.csv', header = 2, quotechar = r'"', skipinitialspace=True)
labels['gunshots'] = labels['positive_labels'].str.contains(g_str)
labels['no_gun'] = labels['positive_labels'].str.contains(ng_str)
gun_positive = labels[labels['gunshots']==True]
no_gun_positive = labels[labels['no_gun']==True]
d = {'Class' : ['gunshots','no_gun'], 'Total_samples' : [gun_positive.shape[0],no_gun_positive.shape[0]]}
df_checkeval = pd.DataFrame(data=d)
df_checkeval.to_csv(r'evaluationclass_distribution.csv')

#Evaluation Set creation
#Creating an evaluation or Validation dataset. Make sure the validation set are equal and randomly sampled.
#output: tfrecord file which has YTID, start, end and positive labels of gunshot audio clips and no_gunshot audioclips.
#Total number of audioclips 

labels = pd.read_csv('../data/raw/eval_segments.csv', header = 2, quotechar = r'"', skipinitialspace=True)
labels['gunshots'] = labels['positive_labels'].str.contains(g_str)
labels['no_gun'] = (labels['positive_labels'].str.contains(ng_str)& ~labels['positive_labels'].str.contains(g_str))

# We found that there are 292 gunshot samples, 6945 other sounds in evaluation set. 
gun_positive = labels[labels['gunshots']==True] 
ng_positive = labels[labels['no_gun']==True].sample(gun_positive.shape[0])
subset = gun_positive.append(ng_positive, ignore_index=True)
subset.to_csv('../data/preprocessed/eval_gunspotting_in_school_subset.csv')
print(subset.shape[0])
print(subset)

#subset has all the dataset which contains gun_positive, background_
#Now lets take a look at the audio embedding files [tfrecord files]
files = glob.glob('../data/raw/audioset_v1_embeddings/eval/*')
# print(files)
subset_ids = subset['# YTID'].values

i=0
writer = tf.python_io.TFRecordWriter('../data/preprocessed/eval_gunspotting_in_school_subset.tfrecord')
for tfrecord in tqdm(files):
    for example in tf.python_io.tf_record_iterator(tfrecord):
        tf_example = tf.train.Example.FromString(example)
        vid_id = tf_example.features.feature['video_id'].bytes_list.value[0].decode(encoding = 'UTF-8')
        if vid_id in subset_ids:
            writer.write(example)
            i= i+1
print(i)
writer.close()


# Train set creation
# Now open the unbalanced_train data

labels = pd.read_csv('../data/raw/unbalanced_train_segments.csv',header=2, quotechar=r'"',skipinitialspace=True)
print(labels.shape[0])

labels['gunshots'] = labels['positive_labels'].str.contains(g_str)
labels['no_gun'] = labels['positive_labels'].str.contains(ng_str) & ~labels['positive_labels'].str.contains(g_str)
gun_positive = labels[labels['gunshots']==True]
print(gun_positive.shape[0])
no_gun_positive = labels[labels['no_gun']==True].sample(gun_positive.shape[0])
print(no_gun_positive.shape[0])
subset = gun_positive.append(no_gun_positive, ignore_index=True)
subset.to_csv('../data/preprocessed/spotting_gunshots_inschool_baltraining_subset.csv')

print(subset)
print(subset.shape[0])



import glob
files = glob.glob('../data/raw/audioset_v1_embeddings/unbal_train/*')
subset_ids= subset['# YTID'].values

i=0
writer = tf.python_io.TFRecordWriter('../data/preprocessed/unbal_spotgun_subset.tfrecord')
for tfrecord in tqdm(files):
    for example in tf.python_io.tf_record_iterator(tfrecord):
        tf_example = tf.train.Example.FromString(example)
        vid_id = tf_example.features.feature['video_id'].bytes_list.value[0].decode(encoding = 'UTF-8')
        if vid_id in subset_ids:
            writer.write(example)
            i+=1
print(i)
writer.close()

# This bal_spotgun_subset.tfrecord just contains [start, end, vid, labels] of gunshot, no gun class [tools, hammer, background, common place #sounds,fireworks, ]

