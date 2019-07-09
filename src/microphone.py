from __future__ import print_function

import sys
sys.path.insert(0, 'VGGish/')


import time
start_time = time.time()


# if(len(sys.argv) <2):
#   print("please pass the relative file path of sound file")
#   sys.exit("Error")

import vggish_slim
import vggish_params
import vggish_input
import vggish_postprocess

import librosa 
import os

import numpy as np
from scipy.io import wavfile
import six

import tensorflow as tf
import keras
from keras.preprocessing.sequence import pad_sequences
import itertools
from keras.models import load_model

from sys import byteorder
from array import array
from struct import pack

import pyaudio
import wave

THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 44100

flags = tf.app.flags

flags.DEFINE_string(
    'wav_file', None,
    'Path to a wav file. Should contain signed 16-bit PCM samples. '
    'If none is provided, a synthetic sound is used.')

flags.DEFINE_string(
    'checkpoint', 'VGGish/vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

flags.DEFINE_string(
    'pca_params', 'VGGish/vggish_pca_params.npz',
    'Path to the VGGish PCA parameters file.')

flags.DEFINE_string(
    'tfrecord_file', None,
    'Path to a TFRecord file where embeddings will be written.')

FLAGS = flags.FLAGS

def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD

def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)
    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    "Trim the blank spots at the start and end"
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    r = array('h', [0 for i in range(int(seconds*RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds*RATE))])
    return r

def record():
    """
    Record a word or words from the microphone and 
    return the data as an array of signed shorts.

    Normalizes the audio, trims silence from the 
    start and end, and pads with 0.5 seconds of 
    blank sound to make sure VLC et al can play 
    it without getting chopped off.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > 30:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r

def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()
    # infer()


def main(_):

  print("please speak a word into the microphone")
  record_to_file('demo.wav')

  y,sr = librosa.load('demo.wav')
  print("sampling rate:",  sr)
  print("Recorded sound wave: ")
  print(sr)

  wav_file = 'demo.wav'
  examples_batch = vggish_input.wavfile_to_examples(wav_file)
  # print(examples_batch)
  # Prepare a postprocessor to munge the model embeddings.
  pproc = vggish_postprocess.Postprocessor(FLAGS.pca_params)

  # If needed, prepare a record writer to store the postprocessed embeddings.
  writer = tf.python_io.TFRecordWriter(FLAGS.tfrecord_file) if FLAGS.tfrecord_file else None

  with tf.Graph().as_default(), tf.Session() as sess:
    # Define the model in inference mode, load the checkpoint, and
    # locate input and output tensors.
    vggish_slim.define_vggish_slim(training=False)
    vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)
    features_tensor = sess.graph.get_tensor_by_name(
        vggish_params.INPUT_TENSOR_NAME)
    embedding_tensor = sess.graph.get_tensor_by_name(
        vggish_params.OUTPUT_TENSOR_NAME)

    # Run inference and postprocessing.
    [embedding_batch] = sess.run([embedding_tensor],
                                 feed_dict={features_tensor: examples_batch})
    # print(embedding_batch)
    postprocessed_batch = pproc.postprocess(embedding_batch)
    # print(postprocessed_batch)

    # Write the postprocessed embeddings as a SequenceExample, in a similar
    # format as the features released in AudioSet. Each row of the batch of
    # embeddings corresponds to roughly a second of audio (96 10ms frames), and
    # the rows are written as a sequence of bytes-valued features, where each
    # feature value contains the 128 bytes of the whitened quantized embedding.
    tf_seq_example = tf.train.SequenceExample(
        feature_lists=tf.train.FeatureLists(
            feature_list={
                vggish_params.AUDIO_EMBEDDING_FEATURE_NAME:
                    tf.train.FeatureList(
                        feature=[
                            tf.train.Feature(
                                bytes_list=tf.train.BytesList(
                                    value=[embedding.tobytes()]))
                            for embedding in postprocessed_batch
                        ]
                    )
            }
        )
    )
    # print(tf_seq_example)
    if writer:
      writer.write(tf_seq_example.SerializeToString())

  if writer:
    writer.close()

  X=[]
  max_len=10
  n_frames = len(tf_seq_example.feature_lists.feature_list['audio_embedding'].feature)
  # print("number of frames = ", n_frames)

  audio_frame = []
  for i in range(n_frames):
      audio_frame.append(np.frombuffer(tf_seq_example.feature_lists.feature_list['audio_embedding'].feature[i].bytes_list.value[0],np.uint8).astype(np.float32))
  

  pad = [np.zeros([128], np.float32) for i in range(max_len-n_frames)]
  audio_frame += pad
  X.append(audio_frame)

  X = np.array(X)
  # print("Dimension before adding newaxis", X.shape)

  # X = X[newaxis,:,:]
  # print("Dimension after adding newaxis", X.shape)

  #Loading LSTM model
  m4 = load_model('src/models/1LayerLSTM__Loss=BinCE_20Epochs_july02.h5')
  p4 = m4.predict(X)

  print("Gunshot score for inference_sample: ====> ", float(p4*100),"percent confidence")
  if(p4>= 0.51):
    print("Gunshot present in the clip")
  else:
    print("Gunshot is not present in the clip")



if __name__ == '__main__':
  tf.app.run()
