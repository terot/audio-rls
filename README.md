# audio-rls

This code implements [Reqursive Least Squares (RLS) filter](https://en.wikipedia.org/wiki/Recursive_least_squares_filter) for audio. RLS is a fast converging linear filter with online learning. In audio processsing it is used in e.g. adative noise cancellation and echo cancellation. The model takes audio file as an input outputs predicted signal and error signal as audio files. 

## Requirements
- [scipy](https://www.scipy.org/scipylib/index.html)
- [numpy](http://www.numpy.org/)
- [theano](http://deeplearning.net/software/theano/)

## Usage

Basic usage:

`python audio_rls.py input.wav --prediction_output=output.wav --error_output=error.wav`

Further help:
`python audio_rls.py --help`

