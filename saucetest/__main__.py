from typing import Callable, Dict
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from librosa.core import stft
from hashlib import md5
import json
import uuid

from saucetest.test_gen import gen_fidelity_test, array_func


win_length = n_fft = 512
hop_length = win_length // 2


stft_args = {
    "n_fft": 512,
    "hop_length": hop_length,
    "win_length": win_length,
    "center": True,
    # "window": "hann",
    # "pad_mode": "constant",
}

stft_func: array_func = lambda x, args: stft(
    x,
    n_fft=args["n_fft"],
    hop_length=args["hop_length"],
    # win_length=args["win_length"],
    center=args["center"],
)


if __name__ == "__main__":
    test_path = Path("testData")
    func_name = "stft"
    func_callable = stft_func
    func_args = stft_args
    input_array = np.random.rand(2, 1024)
    gen_fidelity_test(test_path, func_name, func_callable, func_args, input_array)
