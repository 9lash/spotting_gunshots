# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""A simple demonstration of running VGGish in inference mode.

This is intended as a toy example that demonstrates how the various building
blocks (feature extraction, model definition and loading, postprocessing) work
together in an inference context.

A WAV file (assumed to contain signed 16-bit PCM samples) is read in, converted
into log mel spectrogram examples, fed into VGGish, the raw embedding output is
whitened and quantized, and the postprocessed embeddings are optionally written
in a SequenceExample to a TFRecord file (using the same format as the embedding
features released in AudioSet).

Usage:
  # Run a WAV file through the model and print the embeddings. The model
  # checkpoint is loaded from vggish_model.ckpt and the PCA parameters are
  # loaded from vggish_pca_params.npz in the current directory.
  $ python vggish_inference_demo.py --wav_file /path/to/a/wav/file

  # Run a WAV file through the model and also write the embeddings to
  # a TFRecord file. The model checkpoint and PCA parameters are explicitly
  # passed in as well.
  $ python vggish_inference_demo.py --wav_file /path/to/a/wav/file \
                                    --tfrecord_file /path/to/tfrecord/file \
                                    --checkpoint /path/to/model/checkpoint \
                                    --pca_params /path/to/pca/params

  # Run a built-in input (a sine wav) through the model and print the
  # embeddings. Associated model files are read from the current directory.
  $ python vggish_inference_demo.py
"""

from __future__ import print_function
import sys
sys.path.append('VGGish/')

import numpy as np
from numpy import newaxis
from scipy.io import wavfile
import six
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # {'0', '1', '2', '3'}
                                          # 0 = all messages are logged (default behavior)
                                          # 1 = INFO messages are not printed
                                          # 2 = INFO and WARNING messages are not printed
                                          # 3 = INFO, WARNING, and ERROR messages are not printed

import tensorflow as tf
#to remove contrib WARNING
if type(tf.contrib) != type(tf): tf.contrib._warning = None 
tf.logging.set_verbosity(tf.logging.FATAL)   # DEBUG, INFO, WARN, ERROR, or FATAL - To remove warnings

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim

import keras
from keras.preprocessing.sequence import pad_sequences
import itertools
from keras.models import load_model


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


def main(_):
  # In this simple example, we run the examples from a single audio file through
  # the model. If none is provided, we generate a synthetic input.
  if FLAGS.wav_file:
    wav_file = FLAGS.wav_file
    print("wav_file Found")
  else:
    print("======== wav_file not passed or not found ========")
    print("To pass a wav_file run:")
    print("python inference_pipeline.py --wav_file ../inference_samples/gunshot_samples/pistol_shot.wav")
    print()
    return 0

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
  m4 = load_model('models/1LayerLSTM__Loss=BinCE_20Epochs_july02.h5')
  p4 = m4.predict(X)

  print("Gunshot score for inference_sample: ====> ", float(p4*100),"percent confidence")
  if(p4>= 0.51):
    print("Gunshot present in the clip")
  else:
    print("Gunshot is not present in the clip")

if __name__ == '__main__':
  tf.app.run()



