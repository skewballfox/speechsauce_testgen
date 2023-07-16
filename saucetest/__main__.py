from typing import Callable, Dict
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from librosa.core import stft
from hashlib import md5
import json
import uuid


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


def save_array_pair(
    input_array: NDArray, output_array: NDArray, test_path: Path, test_uuid: uuid.UUID
):
    """Save the input and output arrays to files with the given uuid."""
    np.save(test_path / f"{test_uuid}_input.npy", input_array)
    np.save(test_path / f"{test_uuid}_output.npy", output_array)


def gen_fidelity_test(
    test_path: Path,
    func_name: str,
    func_callable: Callable[[NDArray, Dict], NDArray],
    func_args: Dict,
    input_array: NDArray,
):
    # first generate the directory if it doesn't exist
    full_test_path = (
        test_path
        / func_name
        / md5(json.dumps(func_args, sort_keys=True).encode("utf-8")).hexdigest()
    )
    if not full_test_path.exists():
        full_test_path.mkdir(parents=True, exist_ok=True)
        # save the args to a json file
        with open(full_test_path / "args.json", "w") as f:
            json.dump(func_args, f, indent=4)
    # generate a uuid for the input output pair
    test_uuid = uuid.uuid4()
    # get the output array
    output_array = func_callable(input_array, func_args)
    save_array_pair(input_array, output_array, full_test_path, test_uuid)


if __name__ == "__main__":
    test_path = Path("testData")
    func_name = "stft"
    func_callable = stft_func
    func_args = stft_args
    input_array = np.random.rand(1024)
    gen_fidelity_test(test_path, func_name, func_callable, func_args, input_array)
