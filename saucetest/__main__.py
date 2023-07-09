import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from librosa.core import stft


def save_func(input_array: NDArray, func_name: str) -> None:
    """Save the input array to a file with the given name."""
    np.save(f"{func_name}.npy", input_array)


win_length = n_fft = 512
hop_length = win_length // 4

stft_args = {
    "n_fft": 512,
    "hop_length": hop_length,
    "win_length": win_length,
    # "window": "hann",
    # "pad_mode": "constant",
}

stft_func = lambda x, args: stft(
    x,
    n_fft=args["n_fft"],
    hop_length=args["hop_length"],
    # win_length=args["win_length"],
    center=True,
)
