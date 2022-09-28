Pre-trained VLAAI models
==========================

This folder contains a pre-trained VLAAI model, in three different formats:

1. TensorFlow SavedModel format ([pretrained_model/vlaai](./pretrained_models/vlaai))
2. HDF5 format ([pretrained_model/vlaai.h5](./pretrained_models/vlaai.h5))
3. ONNX format ([pretrained_model/vlaai.onnx](./pretrained_models/vlaai.onnx))


# Dataset
This model was trained on the single-speaker stories dataset, 80 subjects that 
listened to 1 hour and 46 minutes on average (approximately 15 minutes per
recording) for a total of 144 hours of EEG data. 

# Preprocessing

The preprocessing used in this notebook is the same as proposed in the [paper](./#):
* For __EEG__: 
  1. High-pass filtering using a 1st order Butterworth filter with a cutoff frequency of 0.5Hz (using filtfilt)
  2. Downsampling to 1024 Hz
  3. Eyeblink artefact removal using a Multichannel Wiener filter
  4. Common average re-referencing
  5. Downsampling to 64Hz

* For __Speech__:
  1. Envelope extraction using a gamma-tone filterbank
  2. Downsampling to 1024 Hz
  3. Downsampling to 64 Hz

Finally, data was split per recording into a training, validation and test set,
following a 80/10/10 split. The validation and test set were extracted from the
middle of the recording, to avoid any edge effects. Data is standardized
per recording, using the mean and standard deviation of the training set.

# Training procedure

The model was trained for at most 1000 epochs, with a batch size of 64.
The [Adam optimizer](https://arxiv.org/abs/1412.6980) was used with a 
learning rate of 0.001 and negative Pearson r as a loss function on segments
of 5 seconds.