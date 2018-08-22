#!/usr/bin/env python3

import argparse
import scipy.io.wavfile as wav
import numpy as np
from rls import RLS

##
# Argument parsing
#

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # required positional argument
    parser.add_argument('input_file', type=str,
                        default=None, help="model")

    # optional arguments
    parser.add_argument('--dim_input', type=int,
                        default=1000, help="The input window size")
    parser.add_argument('--print_every', type=positive_int,
                        default=1000, help="Report average squared error every nth sample")
    parser.add_argument('--prediction_distance', type=positive_int,
                        default=1, help="How far into future to predict") # >=1
    parser.add_argument('--prediction_output', type=str,
                        default=None, help="Write predicted data to wav file")
    parser.add_argument('--error_output', type=str,
                        default=None, help="Write prediction error to wav file")
    parser.add_argument('--rls_lambda', type=restricted_float,
                        default=1.0, help="RLS lambda parameter, aka. forgetting factor")
    parser.add_argument('--rls_delta', type=positive_float,
                        default=0.001, help="RLS delta parameter for approximate initialization")
    parser.add_argument('--verbose', action="store_true",
                        help="Verbose printing")

    return parser.parse_args()

def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
         raise argparse.ArgumentTypeError("{} is an invalid value for positive int".format(value))
    return ivalue

def restricted_float(value):
    fvalue = float(value)
    if fvalue < 0 or fvalue > 1:
         raise argparse.ArgumentTypeError("{} is not in range [0,1]".format(value))
    return fvalue

def positive_float(value):
    fvalue = float(value)
    if fvalue <= 0:
        raise argparse.ArgumentTypeError("{} is an invalid value for positive float".format(value))
    return fvalue

##
# Main
#

def main(args):
  if args.verbose:
    print("args: {}".format(vars(args)))

  # init model
  rls = RLS(args.dim_input, args.rls_delta)

  # load data
  sample_rate, input_data = wav.read(args.input_file)
  if args.verbose:
    print("sample rate: {} Hz".format(sample_rate))
    print("num samples: {}".format(input_data.size))

  # number of samples to predict
  tot_size = input_data.size - args.dim_input - args.prediction_distance + 1

  # init predicted output and error arrays to empty
  outputs = np.array([], dtype=float)
  errors = np.array([], dtype=float)

  error_sqr_sum = 0.0
  for t in range(tot_size):

    if t>0 and t%args.print_every==0:
      print("{}%\t{}".format(round(100.0*t/tot_size,2), round(error_sqr_sum/args.print_every,5)))
      error_sqr_sum = 0.0

    # compute predicted output and error, and train model
    input = input_data[t:t+args.dim_input]/32767.0
    target = input_data[t+args.dim_input+args.prediction_distance-1]/32767.0
    output, error = rls.observe(input, target, args.rls_lambda)
    error_sqr_sum += error*error

    # append predicted output and error to corresponding arrays
    outputs = np.append(outputs, output)
    errors = np.append(errors, error)

  # write prediction output wav file
  if args.prediction_output is not None:
    wav.write(args.prediction_output, sample_rate, np.int16(output_data*32767.0))
    if args.verbose:
      print("Wrote {}".format(args.prediction_output))

  # write prediction error wav file
  if args.error_output is not None:
    wav.write(args.error_output, sample_rate, np.int16(error_data*32767.0))
    if args.verbose:
      print("Wrote {}".format(args.error_output))

if __name__ == "__main__":
  main(get_args())
