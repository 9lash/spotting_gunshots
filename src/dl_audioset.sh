#!/bin/bash
# clear

#Check if feature_tar.gz exists, if not download
file_name=../data/raw/features.tar.gz
if [ -f $file_name ]
then 
	echo "$file_name already exists"
else   
  echo "$file_name not exists"
  echo "Downloading the Audio Embedding features provided by Google Audioset"
  echo "features.tar.gz will be saved in data/raw/"
  echo "Downloading start ==>"

  # Download the features data from Google Audioset for training 
  wget -P ../data/raw http://storage.googleapis.com/us_audioset/youtube_corpus/v1/features/features.tar.gz
fi

feature_tar=../data/raw/features.tar.gz
if test -f "$feature_tar"; then
	feature=1
    echo "$feature_tar Downloaded successfully"
fi


#Check if unbalanced_train_segments.csv exists, if not download
file_name=../data/raw/unbalanced_train_segments.csv
if [ -f $file_name ]
then 
	echo "$file_name already exists"
else   
  echo "$file_name not exists"
  echo "Downloading unbalanced_train_segments.csv"
   #Download the unbalanced_train_segments.csv describing the start & end seconds of Youtube video  
   wget -P ../data/raw http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv
fi

#Test if download success
dir=../data/raw/unbalanced_train_segments.csv
if test -f "$dir"; then
	unbal=1
    echo "$dir Downloaded successfully or already present"
fi




#Check if balanced_train_segments.csv exists, if not download
file_name=../data/raw/balanced_train_segments.csv
if [ -f $file_name ]
then 
	echo "$file_name already exists"
else   
  echo "$file_name not exists"
  echo "Downloading balanced_train_segments.csv"
   # #Download the balanced_train_segments.csv
  wget -P ../data/raw http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv
fi

dir=../data/raw/balanced_train_segments.csv
if test -f "$dir"; then
	bal=1
    echo "$dir Downloaded successfully or already present"
fi


#Check if eval_segments.csv exists, if not download
file_name=../data/raw/eval_segments.csv
if [ -f $file_name ]
then 
	echo "$file_name already exists"
else   
  echo "$file_name not exists"
  echo "Downloading eval_segments.csv"
# #Download the eval_segments.csv
  wget -P ../data/raw http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv
fi

dir=../data/raw/eval_segments.csv
if test -f "$dir"; then
	eval=1
    echo "$dir Downloaded successfully or already present"
fi



#Check if eval_segments.csv exists, if not download
file_name=../data/raw/audioset_v1_embeddings
if [ -d $file_name ]
then 
	echo "$file_name already exists"
else   
  echo "$file_name not exists"
  echo "Unzipping the feature.tar.gz"
  #Download the 
  tar xvzf ../data/raw/features.tar.gz -C ../data/raw/

fi


dir=../data/raw/audioset_v1_embeddings
if test -d "$dir"; then
   echo "$dir exists"
   emb=1
else
   echo "extraction unsuccessful, try manually extracting"
fi

echo "
      All checks done.
      Please goto data/raw and check the size of audioset_v1_embeddings folder. 
      If the size is 2.9GB to 3.1GB then extraction was successful. 
      If the size = 1.27GB, then extraction was unsuccessful. Please 
      manually extract using a RAR package and place the folder at the exact 
      location.
      "
