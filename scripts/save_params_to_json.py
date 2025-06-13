import json
import numpy as np


def convert(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    return obj


def save_dict_to_json(data: dict, filepath: str):
    with open(filepath, "w") as f:
        json.dump(data, f, default=convert, indent=2)


# Example usage
data = {
    "rgb": {
        "resolution": (1920, 1080),
        "fx": 1121.560547,
        "fy": 1121.922607,
        "cx": 939.114746,
        "cy": 534.709290,
        "px": 0.00125,
        "dist": np.array(
            [0.079764, -0.108468, -0.000167, -0.000509, 0.044803], dtype=np.float64
        ),
    },
    "depth": {
        "resolution": (1024, 1024),
        "fx": 504.752167,
        "fy": 504.695465,
        "cx": 517.601746,
        "cy": 508.529358,
        "px": 0.035,
        "dist": np.array(
            [
                12.565655,
                6.072149,
                0.000045,
                0.000020,
                0.208180,
                12.887096,
                10.294142,
                1.362217,
            ],
            dtype=np.float64,
        ),
    },
    "T": [
        [0.994052, 0.002774, 0.005608, -32.665543 / 1000.0],
        [-0.003367, 0.994064, 0.108743, -0.986931 / 1000.0],
        [-0.005723, -0.108760, 0.994054, 2.863724 / 1000.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
}


save_dict_to_json(data, "./params/femto_mega.json")
