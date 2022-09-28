Datasets used for evaluation
=============================

This directory contains the datasets used for evaluation of the VLAAI network.

# DTU dataset

The [DTU](./DTU) folder contains a preprocessed subset of 
[Fuglsang et al.](https://zenodo.org/record/1199011). This dataset contains 
18 subjects who listened to one of two competing speech audio streams of 
50 seconds with different levels of reverberation. 
In this repository, only the single-speaker trials
(approximately 10 per subject, 500 seconds in total) were saved/used.

The preprocessing used in this notebook is the same as proposed in the [paper](./#):

* For __EEG__: 
  1. High-pass filtering using a 1st order Butterworth filter with a cutoff frequency of 0.5Hz
  2. Downsampling to 1024 Hz
  3. Eyeblink artefact removal using a Multichannel Wiener filter
  4. Common average re-referencing
  5. Downsampling to 64Hz

* For __Speech__:
  1. Envelope extraction using a gamma-tone filterbank
  2. Downsampling to 1024 Hz
  3. Downsampling to 64 Hz