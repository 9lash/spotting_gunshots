
# Spotting Gunshots in Noisy Audio

![](extras/california_sunday_magazine.png)

AudioSet consists of an expanding ontology of 632 audio event classes and a collection of 2,084,320 human-labeled 10-second audio clips drawn from YouTube videos. This project focuses on using this dataset to train a classification model on audio embeddings to detect gunshots against other sounds like fireworks, background noise, speech, glass break, tools, hammer and other extraneous noises. This is achieved by using a pretrained CNN architecture, which we call VGGish to extract the distinct features from the audio clip and then passing this features to a LSTM model to predict the gunshot score. 


On submitting a 10 second audio clip to this python package, you can expect a gunshot prediction score on your terminal stating whether there is a gunshot present in the audio clip. 


# Getting started:

## Setting up the environment

1. Clone the spotting_gunshots repository
2. Goto /src and run ```./dl_vggish_components.sh ```<br/>
      This will download vggish_model.ckpt & VGG_PCA_Parameters from AudioSet and place it to the src/VGGish directory. This cehckpoint file is essential for inference.

3. Setup a new conda environment by installing the python requirements: ```pip install -r requirements.txt``` 
   
   If the installing all requirements was not successful, then these are the general commands which fix issues. 
   ```conda install -c anaconda pygpu ```<br/>
   Ignore issues with incompatible homeassistant package.<br/> 
   And then retry ```pip install -r requirements.txt ``` <br/>

## Run inference on an existing sound clip

To run the algorithm on a demo wav file, 

	1. In your terminal goto src/,
		run python inference_pipeline.py --wav_file ../inference_samples/gunshot_samples/pistol_shot.wav 
	2. To try some other sounds, 
		run python inference_pipeline.py --wav_file ../inference_samples/other_samples/footseps_shuffle.wav

This should print out an output about gunshot probability score on your terminal. Multiple demo wav files are present in the inference_samples folder. 

# Spotting Gunshots Framework: 

## Pipeline: 

![](extras/pipeline.png)

The audio files (.wav) are first preprocessed and converted to the mel-spectrogram representation in the initial preprocessing stage. Then these preprocessed mel-spectrograms are passed  as images to a VGGish model to form  high level audio embedding representation of each sec over a period of 10 seconds. These audio embeddings are post processed and quantized to form 128 8bit numbers which describe each second. 10 such embedding vectors describe a 10 sec audio clip. Then these embeddings are post processed by applying a PCA followed by whitening and as well as quantization to 8 bits per embedding element. 

Now these post processed audio embeddings are passed through a trained 1 layer Long short term Memory (LSTM) model to predict whether a clip has gunshot or no gunshot. 

## Audioset

For training purposes, Audioset can be downloaded into two forms: 

	1. CSV containing the Youtube ID of the audio clip and the labels annotated in the clip.
	2. Audio Embeddings of each clip annotated with the event labels present in the clip.

	To download the entire dataset, goto src/,
	run ./dl_audioset.sh
This will download all the necessary csv files and the features.tar file containing the audio embeddings of all the sound clips in the form of tfrecord format in data/raw. 


## Audio Embeddings: 

The audio embedding that are provided in the dataset are basically a semantically meaningful high level representation of a raw audio clip in 128 dimensional embedding. For instance, if an audio file of 10 seconds long is recorded at 1000 bits/sec resolution, the audio embedding representation would be a [10 x 128] feature vector. Every second in this clip is described by [1x128] high level embedding feature vector. Each feature vector contains 128 8bit numbers. 

These embeddings were generated by a audio classification model called VGGish. VGGish was used as feature extractor to convert raw audio input features into a semantically meaningful, high-level 128-D embedding. So we have a list of Youtube IDs and the corresponding audio embeddings. 

##  Define the two classes: gunshot.csv and no_gunshot.csv
In this case, the gunshot class is a set of Youtube 10 sec audio emebddings which contains labels such as gunshot, gun fire, explosion, artillery fire,  machine gun, fusillade, cap gun. The no_gunshot class contains a range of classes such as small room, large room, public spaces, background noises, hammer, fireworks, burst, pop, human speech, children shouting, radio, television, echo, static, tools, etc. It is essential to define these two classes under data/raw/class folder. So make two csv files called gunshot.csv and no_gunshot.csv which lists the labels of the respective classes using the audioset classes_indices.csv.  


## Creating a subset dataset containing only the above two classes

To create a binary (or multi) classifer, we will first create a balanced subset dataset. Here we create a balanced training subset and evaluation subset from the tfrecord files that we downloaded.

	To create a subset goto src/, run python audioset_feature_subset.py

This will create two files: 
1. bal_spotting_gunshots_subset.tfrecord contains audio embeddings of 8831 gunshot clips & 8831 other class samples
2. eval_spotting_gunshots_subset.tfrecord contains audio embeddings of 292 gunshot and 292 other class samples
3. Additionally it also creates a unbal_train_tagged.csv file which labels those clips which exclusively has gunshots as gunshot class and the samples which exclusively belong to other class [fireworks, glass break, hammer, tools, background noises, usual sounds in school environment like kids chatting, teaching and other outdoor public spaces sounds] as no_gunshot. This exclusiveness helps model the two class and avoids co-occurences of both class. 

All the above dataset and csv files will be placed in data/preprocessed. 

The selection of the other class is crucial to make the classifier robust and was chosen such that the other class has a mix of hard negatives and easy negatives. There are some sounds like people talking which are easily distinguisable as compared to gunshots but other sounds like glass break, footsteps, fireworks, hammer are hard to distinguish form the gunshot sounds. This form of dataset makes the classifier more robust for real world scenario.The samples in raw dataset could contain co-occurences of both the labels. To handle this, gunshot samples were selected such that the clip contains gun class label and does not contain all the other labels stated above. Each sample is thus, mutually exclusive in the training dataset. 



## Lets Build Model
This can be done either locally or on the cloud


	To train a 1 layer Long short Term Memory (LSTM) model 
	run python lstm_single_layer.py 
	This training script will store the 1layer LSTM model (h5 file) under models/ folder. 
	The results of the training score, validation score and the graphs will be stored in the results folder.


	To train the logisitic regression model
		goto src and run python logistic_regression_training.py

You can then use the logisitic regression model present in the models/ directory to infer by replacing the lstm model with logistic regression model in the inference_pipeline.py. The results of the logisitic regression model training and validation performance will be saved in the results folder.


## Data Exploration: 

	To run PCA analysis, t-SNE analysis and plotting goto /src,
		run python data_analysis.py

 This helps you to visualize the training dataset after applying dimensionality reduction techniques like PCA and tSNE. PCA here captures the average value of embeddings of different sound clips and plots in 2 dimensional space. Whereas the t-SNE captures the non linear behviours and preserves neighborhood relationships of sounds.tSNE groups the glass break, gunshots and fireworks are very close to each other. That shows that these classes are hard to distinguish. 


```

## Requisites
- package managers - 
conda version : 4.6.14 or higher
pip 19.1.1

#### Dependencies
- VGGish from 

#### Installation
To install the package above, please run:
```shell
./dl_vggish_components.sh
pip install -r requirements.txt

```

![](extras/tsne.png)
**Figure 3: tSNE shows how distribution of hard negatives classes (fireworks and glass breaks) are with gunshot classes. Each dot is an audio clip ** 


## Analysis
- The sound clips in our dataset are described by a 128 dimensional vector at each second for a duration of 10 sec. Thus forming a 128 x 10 dimensional vector. Principal component Analysis (PCA) was performed on the average of the audio embeddings over the 10 sec duration to form a 3 dimensional representation of sound clips. This was then visualized in a 2D chart. 
- On applying a t-SNE on the audio embeddings, the glass break, gunshots and fireworks class form clusters very close to each other. To checkout the t-SNE graphs, please run python data_analysis.py.
- As a base line, the logistic regression and a single hidden layer neural network was trained on audio embeddings to detect gunshots or no-gunshots. Logistic regression gave a avergage performance of about 84% accuracy. But the training accuracy and validation accuracy of logisitic classifier doesnt flatten out. Graphs show an unstable performance. The single layer neural net also gives a similar reponse. 
- As opposed to Logistic regression and single layer neural net, a single layer LSTM performs much better. The overall accuracy goes upto 92% (highest) when the learning rate is set to 0.001. LSTM takes advantage of its nature to identify a sequence of patterns as opposed to neural nets or logistic regression.
- On trying multiple layers of LSTM (3 layers), the accuracy remained the same. It did not add value to the performance of the classifer but the size of the model increased. 
- Overall, it makes sense to select recall as a metric to decide between the models because it is more expensive to miss a gunshot than to minimize false positives. I see that a 1 layer LSTM works pretty well along with the VGGish feature extractor giving an inference time of about ~7secs on a CPU. <br/>


## Current Limitations of Dataset:
Note that the dataset available has been tagged as gunshot or any other label like tool/ public speaking over the complete 10 secs of time. For the time being there is no frame level labeling. There is only a sample level labelling available for this dataset. This limits the capabilites of the model to perform real time solutions. Currently model has to wait for 10 secs and then process over those set of frames to infer whether there is gunshot or not. 

## References:
Figure 1 - Listening for gunshots, California Sunday Magazine.<br/>
Thanks for audioset team in Google for releasing the curated youtube audio dataset, the VGGish model and the supporting scripts to run it.<br/> 
