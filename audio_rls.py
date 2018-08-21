#!/usr/bin/env python3

import argparse
import scipy.io.wavfile as wav
import numpy as np
from rls import RLS

def parse_positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
         raise argparse.ArgumentTypeError("{} is an invalid value for positive int".format(value))
    return ivalue

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # required positional argument
    parser.add_argument('input_file', type=str,
                        default=None, help="model")

    # optional arguments
    parser.add_argument('--dim_input', type=int,
                        default=1000, help="The input window size")
    parser.add_argument('--print_every', type=parse_positive_int,
                        default=1000, help="Report average squared error every nth sample")
    parser.add_argument('--prediction_distance', type=parse_positive_int,
                        default=1, help="How far into future to predict") # >=1
    parser.add_argument('--prediction_output', type=str,
                        default=None, help="Write predicted data to wav file")
    parser.add_argument('--error_output', type=str,
                        default=None, help="Write prediction error to wav file")
    parser.add_argument('--rls_gamma', type=float,
                        default=0.001, help="RLS gamma parameter")
    parser.add_argument('--rls_lambda', type=float,
                        default=1.0, help="RLS lambda parameter")
    parser.add_argument('--verbose', action="store_true",
                        help="Verbose printing")
    return parser.parse_args()

def main(args):
  if args.verbose:
    print("args: {}".format(vars(args)))

  rls = RLS(args.dim_input, args.rls_gamma)
  sample_rate, input_data = wav.read(args.input_file)

  if args.verbose:
    print("sample rate: {} Hz".format(sample_rate))
    print("num samples: {}".format(input_data.size))

  tot_size = input_data.size - args.dim_input - args.prediction_distance + 1

  output_data = np.array([], dtype=float)
  error_data = np.array([], dtype=float)
  error_sqr_sum = 0.0

  for t in range(tot_size):

    if t%args.print_every==0:
      print("{}%\t{}".format(round(100.0*t/tot_size,2), round(error_sqr_sum/args.print_every,5)))
      error_sqr_sum = 0.0

    input = input_data[t:t+args.dim_input]/32767.0
    target = input_data[t+args.dim_input+args.prediction_distance-1]/32767.0
    output, error = rls.observe(input, target, args.rls_lambda)
    error_sqr_sum += error*error

    output_data = np.append(output_data, output)
    error_data = np.append(error_data, error)

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
