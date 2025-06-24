from src.processor import ImageProcessor


x = ImageProcessor(
    params_path="./params/femto_mega.json",
    image_dir="./rendered",
    output_dir="./output",
)

x.process_and_save_all_images()
