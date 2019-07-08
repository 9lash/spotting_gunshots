# This python code creates a subset of the whole audioset by creating two classes of youtube audio clip embeddings:
# 1. audio embeddings containing gunshots
# 2. audio embeddings of sound clips from people talking, school environment, background noise, glass break, hammer,
# tools, fireworks. 

# The second class is tagged as no_gun class.
# The samples which have co-occurrences of gunshot sample and the no_gun sample are eliminated during training.   

# Before running audioset_feature_subset.py; one should run the dl_audioset.sh. This is essential so that
# audioset_feature_subset.py is able to locate the necessary csv files and the tfrecord files to create a 
# subset tfrecord file.

# Input: 4 CSV files and complete audioset in tfrecord format.
# CSV files that are downloaded directly from Audioset dataset are in the directory data/raw/: 
# 1. unbalanced_train_segments.csv 
# 2  eval_segments.csv
# 3. Additionally this python file also expects the descriptions of the classes in terms of labels. For eg:
#    Under /raw/data/classes/ there are two files gunshot.csv and no_gunshot.csv
#    gunshot.csv describes the classes in audioset which are categorized as gunshots. 
#    In our case, the youtube videos which are tagged as gunshot, cap gun, machine gun, gunfire, fusillade, artillery
#    fire & explosion are considered as gunshot class.  
#    Like wise, the no_gunshot.csv contains all the labels which define this class. 
# 4. This script also expects that the complete dataset of the audioset is downloaded under 'data/raw/audioset_v1_embeddings'
#	 This folder should contain the tfrecord files of all the youtube 10-sec audio embeddings which contain the labels.

# Ouput: 
# Outputs will be stored in the dir: data/preprocessed/
# 
# To make your own classes, one could add their own labels here by referring to the audioset class_labels_indices.csv.
# This will help form your own subset dataset. 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import glob

#!pip install seaborn
import seaborn as sns
import os

#for displaying the progress of training subset creation
from tqdm import tqdm   



# creates an unbalanced training segment subset which will contain only gunshot class & no_gunshot class
# the whole audioset
# Input: path of 1.unbalanced_train_segments.csv 2. gunshot_labels.csv 3.no_gunshot.csv
# Ouptut: CSV file containing youtube ids which contain gunshot labels or no_gunshot labels and 
# eliminate all other classes present in the vast audioset.
# def trainsubset_csv(unbaltrain_path, gunlabels_path, nogunlables_path, unbaltagged_csvpath):
def trainsubset_csv(unbaltrain_path, unbaltagged_csvpath, gunlabels_path, nogunlables_path):
	labels = pd.read_csv(unbaltrain_path,header=2, quotechar=r'"',skipinitialspace=True)
	gunshot_labels = pd.read_csv(gunlabels_path, names=['num','label','description'])
	g_str = '|'.join(gunshot_labels['label'].values)
	no_gun_labels = pd.read_csv(nogunlables_path, names = ['num', 'label', 'description'])
	ng_str = '|'.join(no_gun_labels['label'].values)
	labels['no_gun'] = labels['positive_labels'].str.contains(ng_str) & ~labels['positive_labels'].str.contains(g_str)
	labels['gunshots'] = labels['positive_labels'].str.contains(g_str) 
	labels.to_csv(unbaltagged_csvpath)
	print("Created a tagged training data CSV file at data/preprocessed/unbalanced_train_tagged.csv")


#Evaluation Set creation
#Creating an evaluation dataset. Make sure the evaluation set are equal and randomly sampled.
#output: tfrecord file which has YTID, start, end and positive labels of gunshot audio clips and no_gunshot audioclips.
#Total number of audioclips = 494 (292 gunshots and 292 other class)

def create_evalsubset(eval_path,evalsubset_csvpath,evalsubset_tf, raw_evaltf, gunlabels_path, nogunlables_path):

	labels = pd.read_csv(eval_path, header = 2, quotechar = r'"', skipinitialspace=True)
	gunshot_labels = pd.read_csv(gunlabels_path, names=['num','label','description'])
	g_str = '|'.join(gunshot_labels['label'].values)
	no_gun_labels = pd.read_csv(nogunlables_path, names = ['num', 'label', 'description'])
	ng_str = '|'.join(no_gun_labels['label'].values)
	labels['gunshots'] = labels['positive_labels'].str.contains(g_str)
	labels['no_gun'] = (labels['positive_labels'].str.contains(ng_str)& ~labels['positive_labels'].str.contains(g_str))

	# We found that there are 292 gunshot samples, 6945 other sounds in total evaluation set.
	# out of which we choose 292 gunshots and 292 other sounds to create a eval subset. 
	gun_positive = labels[labels['gunshots']==True] 
	ng_positive = labels[labels['no_gun']==True].sample(gun_positive.shape[0])
	subset = gun_positive.append(ng_positive, ignore_index=True)
	subset.to_csv(evalsubset_csvpath)
	print(subset.shape[0])
	print(subset)

	#subset has all the dataset which contains gun_positive, background
	#Now lets take a look at the audio embedding files [tfrecord files]
	files = glob.glob(raw_evaltf)
	# print(files)
	subset_ids = subset['# YTID'].values

	i=0
	writer = tf.python_io.TFRecordWriter(evalsubset_tf)
	for tfrecord in tqdm(files):
	    for example in tf.python_io.tf_record_iterator(tfrecord):
	        tf_example = tf.train.Example.FromString(example)
	        vid_id = tf_example.features.feature['video_id'].bytes_list.value[0].decode(encoding = 'UTF-8')
	        if vid_id in subset_ids:
	            writer.write(example)
	            i= i+1
	print(i)
	writer.close()
	print("New tfrecord created: eval_gunspotting_in_school_subset.tfrecord")



# Train set creation
# This bal_spotgun_subset.tfrecord just contains [start, end, video_id, labels] of gunshot samples, 
# no gun samples contains samples ofpeople talking, school environment, background noise, glass break,
# hammer, tools, fireworks. 

#output: tfrecord file which has YTID, start, end and positive labels of gunshot audio clips and no_gunshot audioclips.
#Total number of samples in each class  = 8831 samples
def create_trainsubset(unbaltrain_path, trainsubset_csvpath, trainsubset_tf, raw_traintf, gunlabels_path, nogunlables_path):

	labels = pd.read_csv(unbaltrain_path,header=2, quotechar=r'"',skipinitialspace=True)
	print(labels.shape[0])
	gunshot_labels = pd.read_csv(gunlabels_path, names=['num','label','description'])
	g_str = '|'.join(gunshot_labels['label'].values)
	no_gun_labels = pd.read_csv(nogunlables_path, names = ['num', 'label', 'description'])
	ng_str = '|'.join(no_gun_labels['label'].values)	

	labels['gunshots'] = labels['positive_labels'].str.contains(g_str)
	labels['no_gun'] = labels['positive_labels'].str.contains(ng_str) & ~labels['positive_labels'].str.contains(g_str)
	gun_positive = labels[labels['gunshots']==True]
	print(gun_positive.shape[0])
	no_gun_positive = labels[labels['no_gun']==True].sample(gun_positive.shape[0])
	print(no_gun_positive.shape[0])
	subset = gun_positive.append(no_gun_positive, ignore_index=True)
	subset.to_csv(trainsubset_csvpath)

	print(subset)
	print(subset.shape[0])
	
	files = glob.glob(raw_traintf)
	subset_ids= subset['# YTID'].values

	i=0
	writer = tf.python_io.TFRecordWriter(trainsubset_tf)
	for tfrecord in tqdm(files):
	    for example in tf.python_io.tf_record_iterator(tfrecord):
	        tf_example = tf.train.Example.FromString(example)
	        vid_id = tf_example.features.feature['video_id'].bytes_list.value[0].decode(encoding = 'UTF-8')
	        if vid_id in subset_ids:
	            writer.write(example)
	            i+=1
	
	print("New tfrecord created: eval_gunspotting_in_school_subset.tfrecord")
	writer.close()


# def main():
# 	print("Hi")
# 	#create a subset of training data which contains gunshot and no_gunshot class in csv file
# 	trainsubset_csv(unbaltrain_path, unbaltagged_csvpath)
# 	#create an eval subset
# 	create_evalsubset(eval_path,evalsubset_csvpath,evalsubset_tf,raw_evaltf)
# 	#create a training subset
# 	create_trainsubset(unbaltrain_path, trainsubset_csvpath, trainsubset_tf, raw_traintf)
	
if __name__ == "__main__":
	# The following are the path of the csv files, raw tfrecords and final subset tfrecords:

	#All 4 Input CSV paths
	unbaltrain_path = '../data/raw/unbalanced_train_segments.csv'
	eval_path = '../data/raw/eval_segments.csv'
	gunlabels_path = '../data/raw/classes/gunshot_labels.csv'
	nogunlables_path = '../data/raw/classes/no_gunshot.csv'

	# raw tfrecords from audioset
	raw_evaltf = '../data/raw/audioset_v1_embeddings/eval/*'
	raw_traintf = '../data/raw/audioset_v1_embeddings/unbal_train/*'

	# All 3 ouput CSV files
	unbaltagged_csvpath = '../data/preprocessed/unbalanced_train_tagged.csv'
	evalsubset_csvpath = '../data/preprocessed/eval_gunspotting_in_school_subset.csv'
	trainsubset_csvpath = '../data/preprocessed/spotting_gunshots_inschool_baltraining_subset.csv'

	# 2 output tfrecord subsets 
	trainsubset_tf = '../data/preprocessed/unbal_spotgun_subset.tfrecord'
	evalsubset_tf = '../data/preprocessed/eval_gunspotting_in_school_subset.tfrecord'

	#create a subset of training data which contains gunshot and no_gunshot class in csv file
	trainsubset_csv(unbaltrain_path, unbaltagged_csvpath, gunlabels_path, nogunlables_path)
	#create an eval subset
	create_evalsubset(eval_path, evalsubset_csvpath, evalsubset_tf, raw_evaltf, gunlabels_path, nogunlables_path)
	#create a training subset
	create_trainsubset(unbaltrain_path, trainsubset_csvpath, trainsubset_tf, raw_traintf, gunlabels_path, nogunlables_path)




#Additional features

# #Checking the distribution of the classes in the training dataset
# def training_distribution(unbaltrain_path):	
# 	print("Analyzing the distribution of classes in Training subset")
# 	labels = pd.read_csv(unbaltrain_path, header = 2, quotechar = r'"', skipinitialspace=True)
# 	labels['gunshots'] = labels['positive_labels'].str.contains(g_str)
# 	labels['no_gun'] = labels['positive_labels'].str.contains(ng_str) & ~labels['positive_labels'].str.contains(g_str)
# 	gun_positive = labels[labels['gunshots']==True]
# 	no_gun_positive = labels[labels['no_gun']==True]
# 	d = {'Class' : ['gunshots','no_gun'], 'Total_samples' : [gun_positive.shape[0],no_gun_positive.shape[0]]}
# 	train_df = pd.DataFrame(data=d)
# 	train_df.to_csv(r'training_distribution.csv')
# 	print("...")
# 	print("training_distribution.png saved in disk")

# #Checking out whats the distribution of the classes in the evaluation dataset
# def eval_distribution():
# 	print("Analyzing the distribution of the classes in Eval dataset")
# 	labels = pd.read_csv(r'../data/raw/eval_segments.csv', header = 2, quotechar = r'"', skipinitialspace=True)
# 	labels['gunshots'] = labels['positive_labels'].str.contains(g_str)
# 	labels['no_gun'] = labels['positive_labels'].str.contains(ng_str)
# 	gun_positive = labels[labels['gunshots']==True]
# 	no_gun_positive = labels[labels['no_gun']==True]
# 	d = {'Class' : ['gunshots','no_gun'], 'Total_samples' : [gun_positive.shape[0],no_gun_positive.shape[0]]}
#	df_checkeval = pd.DataFrame(data=d)
# 	df_checkeval.to_csv(r'evaluationclass_distribution.csv')
