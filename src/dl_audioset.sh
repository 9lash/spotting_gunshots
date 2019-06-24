#!/bin/bash
clear

echo "Downloading the Audio Embedding features provided by Google Audioset"
echo "features.tar.gz will be saved in data/raw/"
echo "Downloading start ==>"

# Download the features data from Google Audioset for training 
wget -P ../data/raw http://storage.googleapis.com/us_audioset/youtube_corpus/v1/features/features.tar.gz

feature_tar=../data/raw/features.tar.gz
if test -f "$feature_tar"; then
	feature=1
    echo "$feature_tar Downloaded successfully"
fi

echo "Downloading unbalanced_train_segments.csv"
# #Download the unbalanced_train_segments.csv describing the start & end seconds of Youtube video  
wget -P ../data/raw http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv

dir=../data/raw/unbalanaced_train_segments.csv
if test -f "$dir"; then
	unbal=1
    echo "$dir Downloaded successfully"
fi

echo "Downloading balanced_train_segments.csv"
# #Download the balanced_train_segments.csv
wget -P ../data/raw http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv

dir=../data/raw/balanced_train_segments.csv
if test -f "$dir"; then
	bal=1
    echo "$dir Downloaded successfully"
fi


echo "Downloading eval_segments.csv"
# #Download the eval_segments.csv
wget -P ../data/raw http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv

dir=../data/raw/eval_segments.csv
if test -f "$dir"; then
	eval=1
    echo "$dir Downloaded successfully"
fi


echo "Unzipping the feature.tar.gz"
#Download the 
tar xvzf ../data/raw/features.tar.gz -C ../data/raw/

dir=../data/raw/audioset_v1_embeddings
if test -f "$dir"; then
	zip=1
    echo "$dir exists, extraction successful"
fi



