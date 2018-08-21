#!/usr/bin/env python3

import sys
import scipy.io.wavfile as wav
import numpy as np
from rls import RLS

def parse_arguments():
  usage = "./audio_rls <input.wav> <output.wav> <error.wav>" 
  if len(sys.argv) != 4:
    print("Invalid number of arguments")
    print(usage)
    sys.exit(1)

  input_file = sys.argv[1]
  output_file = sys.argv[2]
  error_file = sys.argv[3]
  return input_file, output_file, error_file
    

input_file, output_file, error_file = parse_arguments()

# parameters
dim_input = 1000
prediction_distance = 1 # 0 is invalid, since it is known
rls_gamma = 0.001 #0.001
rls_lambda = 1.5 #1.0

rls = RLS(dim_input, rls_gamma)
sample_rate, input_data = wav.read(input_file)
tot_size = input_data.size - dim_input - prediction_distance + 1

output_data = np.array([], dtype=float)
error_data = np.array([], dtype=float)
error_sqr_sum = 0.0

epoch_size = 1000

for t in range(tot_size):

  if t%epoch_size==0:
    print(str(round(100.0*t/tot_size,2)) + "%\t" +
      str(round(error_sqr_sum/epoch_size,5)))
    error_sum = 0.0
  
  input = input_data[t:t+dim_input]/32767.0
  target = input_data[t+dim_input+prediction_distance-1]/32767.0
  
  output, error = rls.observe(input, target,rls_lambda)

  output_data = np.append(output_data, output)
  error_data = np.append(error_data, error)

  error_sqr_sum += error*error

wav.write(output_file, sample_rate, np.int16(output_data*32767.0))
wav.write(error_file, sample_rate, np.int16(error_data*32767.0))
