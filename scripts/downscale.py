import os
from PIL import Image


def downscale_images(input_dir: str, output_dir: str, size=(256, 144)) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".png"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                with Image.open(input_path) as img:
                    if img.size != (1920, 1080):
                        print(f"Skipping {filename}: not 1920x1080")
                        continue
                    img = img.resize(size, Image.LANCZOS)
                    img.save(output_path, format="PNG")
            except Exception as e:
                print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    input_folder = "output/1/rgb_hr"
    output_folder = "output/1/rgb_lr"
    downscale_images(input_folder, output_folder)
