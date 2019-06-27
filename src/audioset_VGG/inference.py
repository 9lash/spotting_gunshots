from __future__ import print_function

import time
start_time = time.time()

import sys

if(len(sys.argv) <2):
  print("please pass the relative file path of sound file")
  sys.exit("Error")

import vggish_slim
import vggish_params
import vggish_input
import vggish_postprocess

import numpy
import librosa 
import os

import numpy as np
from scipy.io import wavfile
import six
import vggish_slim
import vggish_params
import vggish_input

import tensorflow as tf
import keras


#Creating a VGGish network
def CreateVGGishNetwork(hop_size=0.96):   # Hop size is in seconds.
  """Define VGGish model, load the checkpoint, and return a dictionary that points
  to the different tensors defined by the model.
  """
  vggish_slim.define_vggish_slim()
  checkpoint_path = 'vggish_model.ckpt' #'audioset_VGG/vggish_model.ckpt'
  vggish_params.EXAMPLE_HOP_SECONDS = hop_size
  
  vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)

  features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
  embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)

  layers = {'conv1': 'vggish/conv1/Relu',
            'pool1': 'vggish/pool1/MaxPool',
            'conv2': 'vggish/conv2/Relu',
            'pool2': 'vggish/pool2/MaxPool',
            'conv3': 'vggish/conv3/conv3_2/Relu',
            'pool3': 'vggish/pool3/MaxPool',
            'conv4': 'vggish/conv4/conv4_2/Relu',
            'pool4': 'vggish/pool4/MaxPool',
            'fc1': 'vggish/fc1/fc1_2/Relu',
            'fc2': 'vggish/fc2/Relu',
            'embedding': 'vggish/embedding',
            'features': 'vggish/input_features',
         }
  g = tf.get_default_graph()
  for k in layers:
    layers[k] = g.get_tensor_by_name( layers[k] + ':0')
    
  return {'features': features_tensor,
          'embedding': embedding_tensor,
          'layers': layers,
         }


# Post Process with VGGish output by applying a PCA transfromation and as well 
# as quantization to 8 bits per embedding element. 

def ProcessWithVGGish(vgg, x, sr):
  '''Run the VGGish model, starting with a sound (x) at sample rate
  (sr). Return a whitened version of the embeddings. Sound must be scaled to be
  floats between -1 and +1.'''

  # Produce a batch of log mel spectrogram examples.
  input_batch = vggish_input.waveform_to_examples(x, sr)
  # print('Log Mel Spectrogram example: ', input_batch[0])

  [embedding_batch] = sess.run([vgg['embedding']],feed_dict={vgg['features']: input_batch})

  # Postprocess the results to produce whitened quantized embeddings.
  pca_params_path = 'vggish_pca_params.npz' #'audioset_VGG/vggish_pca_params.npz'

  pproc = vggish_postprocess.Postprocessor(pca_params_path)
  postprocessed_batch = pproc.postprocess(embedding_batch)
  # print('Postprocessed VGGish embedding: ', postprocessed_batch[0])
  return postprocessed_batch[0]


def EmbeddingsFromVGGish(vgg, x, sr):
  '''Run the VGGish model, starting with a sound (x) at sample rate
  (sr). Return a dictionary of embeddings from the different layers
  of the model.'''
  # Produce a batch of log mel spectrogram examples.
  input_batch = vggish_input.waveform_to_examples(x, sr)
  # print('Log Mel Spectrogram example: ', input_batch[0])

  layer_names = vgg['layers'].keys()
  tensors = [vgg['layers'][k] for k in layer_names]
  
  results = sess.run(tensors,
                     feed_dict={vgg['features']: input_batch})

  resdict = {}
  for i, k in enumerate(layer_names):
    resdict[k] = results[i]
    
  return resdict


#---- main code ----#
# def main():


# loading the sound file 
project_dir = os.path.dirname(os.path.abspath(__file__))
sound_file_path = os.path.join(project_dir, sys.argv[1])      #'../../inference_sample_data/asg_sti_10sec.mp3')
print(sound_file_path)

y,sr = librosa.load(sound_file_path)
print("sampling rate:",  sr)

tf.reset_default_graph()
sess = tf.Session()

vgg = CreateVGGishNetwork(0.01)

resdict = EmbeddingsFromVGGish(vgg, y,sr)
for k in resdict:
  print(k, resdict[k].shape)
print("Embedding", resdict['embedding'])

model = keras.models.load_model('../Models/1LayerLSTM__Loss=BinCE_40Epochs_highestacc.h5')
p = model.predict(np.expand_dims(resdict['embedding'], axis=0))
print("Gunshot prediction probability", p*100)


# main()
print("--- %s seconds ---" % (time.time() - start_time))