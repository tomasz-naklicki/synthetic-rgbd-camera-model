from src.processor import ImageProcessor


x = ImageProcessor(
    params_path="./params/femto_mega.json",
    image_dir="./rendered",
    output_dir="./output",
)
x.process_single_img_pair(
    rgb_img_path="./rendered/rgb_0.png", depth_image_path="./rendered/depth_0.png"
)
