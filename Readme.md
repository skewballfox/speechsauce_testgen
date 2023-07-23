# SauceTest

Just a helper program for generating testing data for [SpeechSauce](https://github.com/secretsauceai/mfcc-rust) that I'm thinking of making more generic to help with other ndarray-rust libs, or any project which requires compatibility with a set of python numpy-centric functions.

## generated data layout

here's what I'm thinking right now:

- function_name as dir
  - hash_of_nonarray_args as directory
    - non_ndarray_args as json
    - test_uuid as dir
      - input.npy
      - output.npy

the goal is to have a series of individual input and outputs for each set of parameters. rust-side the json file is deserialized and used to set all other inputs for the function. Then it loops over the input/output pairs confirming the output of the rust function matches the stored output of the python function within some acceptable error bounds. tools may be added for analyzing incorrect results.

there may be a file that associates a set of batch or spectific test uuids with a set of tests, so that you can't avoid running the full suite each time.

## Getting Started

obviously this is a WIP, but if you want to use it, create 3 things:

- an input array
- a dictionary of arguments that will be saved as a json file
- a lambda which takes a numpy array and the dictionary of arguments

pass all 3 into `gen_fidelity_test`. you can randomly create those inputs in a loop. I plan to have functons for generating data for more than one test, just working out how to do so in a way that is generic enough to be useful by someone other than me.
