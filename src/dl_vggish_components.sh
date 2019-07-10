#!/bin/bash
# clear

if test -f "VGGish/vggish_model.ckpt"; then
echo vggish_model.ckpt is already downloaded.
else
echo downloading vggish_model.ckpt
wget -P VGGish/ https://storage.googleapis.com/audioset/vggish_model.ckpt
fi

if test -f "VGGish/vggish_pca_params.npz"; then
echo vggish_pca_params.npz is already downloaded.
else
echo downloading vggish_pca_params.npz
wget -P VGGish/ https://storage.googleapis.com/audioset/vggish_pca_params.npz
fi


echo " Checked if vggish_model.ckpt & vggish_pca_params.npz are present under src/VGGish. Success!"
