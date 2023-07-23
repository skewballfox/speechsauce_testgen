from typing import Callable, Dict
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from librosa.core import stft
from hashlib import md5
import json
import uuid

array_func = Callable[[NDArray, Dict], NDArray]


def save_array_pair(
    input_array: NDArray, output_array: NDArray, test_path: Path, test_uuid: uuid.UUID
):
    """Save the input and output arrays to files with the given uuid."""
    current_test_path = test_path / str(test_uuid)
    current_test_path.mkdir(parents=True, exist_ok=False)
    np.save(current_test_path / "input.npy", input_array)
    np.save(current_test_path / "output.npy", output_array)


def gen_fidelity_test(
    test_path: Path,
    func_name: str,
    func_callable: array_func,
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
