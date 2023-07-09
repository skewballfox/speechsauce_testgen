# SauceTest

Just a helper program for generating testing data for [SpeechSauce](https://github.com/secretsauceai/mfcc-rust) that I'm thinking of making more generic to help with other ndarray-rust libs, or any project which requires compatibility with a set of python numpy-centric functions.

## generated data layout

here's what I'm thinking right now:

- function name as dir
  - batch_uuid labeled directory
    - non-ndarray input as json
    - test_uuid_input.npy
    - test_uuid_output.npy

the goal is to have a series of individual input and outputs for each set of parameters. rust-side the json file is desierialized and used to set all other inputs for the function. Then it loops over the input/output pairs confirming the output of the rust function matches the stored output of the python function within some acceptable error bounds.

there may be a file that associates a set of batch or spectific test uuids with a set of tests, so that you can't avoid running the full suite each time.
