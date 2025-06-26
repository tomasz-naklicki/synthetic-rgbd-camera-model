# Blender image generation
Script used in Blender for scene generation and rendering. It creates a base scene and places objects from YCB dataset into it. This pipeline can be followed by adding the script to a Blender project with some changes to settings.

## Compositing
![blender_compositing](https://github.com/user-attachments/assets/fbe78e4c-e2af-4b1c-9c7a-fbab9f2d22a9)
To acquire depth data, Blender’s Compositing settings were used. The following nodes were used:
- Render Layers - used to acquire depth data,
- Multiply - data was multiplied by 1000, to emulate the internal multiplier used in Orbbec
Femto Mega camera,
- Divide - data was divided by 65535, to fit inside 16-bit range,
- Composite - managed the rendering output.

## Other settings
Additionally, other settings were used as follows:
- in the Render settings, Render Engine was set to Cycles
- in the View Layer settings, Z-pass was added
- in the Output settings, Color Depth was set to 16-bit

## Paths
In the script, two paths need to be declared:
- ```BASE_DIR``` - path to the folder including YCB models,
- ```RENDER_PATH``` - render output path
