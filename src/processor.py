import json
import numpy as np
import re
import os
from typing import Any, List, Tuple
from pathlib import Path
import cv2

from src.preprocessing import PreprocessingManager
from src.projection import ProjectionManager


class ImageProcessor:
    def __init__(self, params_path: str, image_dir: str = None, output_dir: str = None):
        self.output_dir = output_dir
        self.params = self._load_params_from_json(params_path)
        self.image_paths = self._list_image_pairs(image_dir) if image_dir else None
        self.preprocessing = PreprocessingManager(params=self.params)
        self.projection = ProjectionManager(
            rgb_params=self.params["rgb"],
            depth_params=self.params["depth"],
            transformation_matrix=self.params["T"],
        )
        if not image_dir or not output_dir:
            print(
                "WARNING: input or output directory not specified. \n Some functionality may not work correctly."
            )

    def _load_params_from_json(self, filepath: str) -> dict:
        def convert(obj: Any) -> Any:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == "resolution" and isinstance(v, list):
                        obj[k] = tuple(v)
                    elif k in ("dist", "T") and isinstance(v, list):
                        obj[k] = np.array(v)
                    elif isinstance(v, dict):
                        obj[k] = convert(v)
            return obj

        with open(filepath, "r") as f:
            data = json.load(f)
        return convert(data)

    def _list_image_pairs(self, image_dir: str) -> List[Tuple[str, str]]:
        """Find matching RGB/depth PNG pairs in a directory.

        Expects names like rgb_0.png and depth_0.png.

        Args:
        image_dir (str): Directory to search.

        Returns:
        List[Tuple[str, str]]: List of (rgb_path, depth_path) pairs.
        """
        pattern = re.compile(r"(rgb|depth)_(\d+)\.png$", re.IGNORECASE)
        files = Path(image_dir).iterdir()

        grouped = {}
        for file in files:
            if file.is_file() and file.suffix.lower() == ".png":
                match = pattern.match(file.name)
                if match:
                    label, index = match.groups()
                    grouped.setdefault(index, {})[label.lower()] = str(file.resolve())

        return [
            (pair["rgb"], pair["depth"])
            for index, pair in grouped.items()
            if "rgb" in pair and "depth" in pair
        ]

    def _process_image_pair(self, image_pair_path: List[Tuple[str, str]]):
        """Load, preprocess, and align a single RGB-depth image pair.

        Args:
            image_pair_path (Tuple[str, str]): (rgb_path, depth_path).

        Returns:
            tuple: (RGB image array, aligned depth image array).
        """
        rgb_img = cv2.imread(image_pair_path[0])
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        depth_img = cv2.imread(image_pair_path[1], cv2.IMREAD_UNCHANGED)

        preprocessed_depth_img = self.preprocessing.get_processed_image(
            depth_img, rgb_img
        )
        depth_img_aligned = self.projection.get_aligned_depth_img(
            preprocessed_depth_img, rgb_img
        )

        return rgb_img, depth_img_aligned

    def _create_output_dir_for_batch(self):
        os.makedirs(self.output_dir, exist_ok=True)

        numbered_folders = [
            int(name)
            for name in os.listdir(self.output_dir)
            if name.isdigit() and os.path.isdir(os.path.join(self.output_dir, name))
        ]
        next_number = max(numbered_folders, default=-1) + 1
        new_folder_path = os.path.join(self.output_dir, str(next_number))
        os.makedirs(new_folder_path)
        return new_folder_path

    def process_and_save_all_images(self):
        """Process every image pair in the input directory and save results.

        Saves into a new numbered batch folder under `self.output_dir`.
        """
        cnt = 0
        output_dir = self._create_output_dir_for_batch()
        for pair in self.image_paths:
            rgb_img, depth_img = self._process_image_pair(pair)
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{output_dir}/rgb_{cnt}.png", rgb_img)
            cv2.imwrite(f"{output_dir}/depth_{cnt}.png", depth_img)
            cnt += 1

    def process_single_img_pair(self, rgb_img_path: str, depth_image_path: str):
        """Process one image pair and save into a new batch folder.

        Args:
            rgb_img_path (str): Path to RGB image.
            depth_image_path (str): Path to depth image.
        """
        output_dir = self._create_output_dir_for_batch()
        rgb_img, depth_img = self._process_image_pair((rgb_img_path, depth_image_path))
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{output_dir}/rgb_0.png", rgb_img)
        cv2.imwrite(f"{output_dir}/depth_0.png", depth_img)

    def process_single_img_pair_no_save(self, rgb_img_path: str, depth_image_path: str):
        """Process one image pair and return results without saving.

        Args:
            rgb_img_path (str): Path to RGB image.
            depth_image_path (str): Path to depth image.

        Returns:
            tuple: (BGR-formatted RGB image, aligned depth image).
        """
        rgb_img, depth_img = self._process_image_pair((rgb_img_path, depth_image_path))
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        return rgb_img, depth_img
