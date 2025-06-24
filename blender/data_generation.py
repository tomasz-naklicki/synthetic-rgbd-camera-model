import bpy
import os
import random
import math
from pathlib import Path
import mathutils
import numpy as np

BASE_DIR = Path("./models/ycb").resolve()
RENDER_PATH = Path("./rendered").resolve()

depth_calib_data = {
    "resolution": (1024, 1024),
    "fx": 504.752167,
    "fy": 504.695465,
    "cx": 517.601746,
    "cy": 508.529358,
}
rgb_calib_data = {
    "resolution": (1920, 1080),
    "fx": 1121.560547,
    "fy": 1121.922607,
    "cx": 939.114746,
    "cy": 534.709290,
}


def clean_scene():
    """
    Clean the entire Blender scene of objects, collections, lights, cameras,
    worlds, and other datablocks while preserving materials.
    """
    if bpy.context.active_object and bpy.context.active_object.mode == "EDIT":
        bpy.ops.object.editmode_toggle()
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    for collection in list(bpy.data.collections):
        bpy.data.collections.remove(collection)

    for world in list(bpy.data.worlds):
        bpy.data.worlds.remove(world)

    for cam in list(bpy.data.cameras):
        if cam.users == 0:
            bpy.data.cameras.remove(cam)

    for light in list(bpy.data.lights):
        if light.users == 0:
            bpy.data.lights.remove(light)

    def remove_unlinked(data_block_collection):
        for datablock in list(data_block_collection):
            if datablock.users == 0:
                data_block_collection.remove(datablock)

    remove_unlinked(bpy.data.meshes)
    remove_unlinked(bpy.data.curves)
    remove_unlinked(bpy.data.textures)
    remove_unlinked(bpy.data.images)
    remove_unlinked(bpy.data.particles)
    remove_unlinked(bpy.data.actions)
    remove_unlinked(bpy.data.node_groups)

    bpy.ops.world.new()
    bpy.context.scene.world = bpy.data.worlds["World"]


pixel_size_depth = 0.0035  # mm
width_depth, height_depth = depth_calib_data["resolution"]
sensor_width_depth = width_depth * pixel_size_depth
sensor_height_depth = height_depth * pixel_size_depth

shitf_x_depth = (depth_calib_data["cx"] - (width_depth / 2)) / width_depth
shitf_y_depth = (depth_calib_data["cy"] - (height_depth / 2)) / height_depth


pixel_size_rgb = 0.00125  # mm
width_rgb, height_rgb = rgb_calib_data["resolution"]
sensor_width_rgb = width_rgb * pixel_size_rgb
sensor_height_rgb = height_rgb * pixel_size_rgb

shitf_x_rgb = (rgb_calib_data["cx"] - (width_rgb / 2)) / width_rgb
shitf_y_rgb = (rgb_calib_data["cy"] - (height_rgb / 2)) / height_rgb


camera_locations_rotations = [
    ((0, -1.5, 1), (1.04, 0, 0)),
    ((0, -1.5, 1.7), (0.7, 0, 0)),
    ((0, -1.9, 0.5), (1.3, 0, 0)),
    ((0, -1.9, 0.2), (1.57, 0, 0)),
]


n_renders = 5

for i in range(n_renders):
    clean_scene()
    obj_paths = []
    for root, _, files in os.walk(BASE_DIR):
        if "textured.obj" in files:
            obj_paths.append(os.path.join(root, "textured.obj"))

    num_objects = random.randint(2, min(5, len(obj_paths)))
    selected_objs = random.sample(obj_paths, num_objects)

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # Scene generation
    # Material parameters are optional, used for coloring

    bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
    plane = bpy.context.object
    plane.scale = (0.485, 0.537, 0.021)
    material = bpy.data.materials["Material.003"]
    plane.active_material = material

    bpy.ops.mesh.primitive_plane_add(location=(0, 0, -0.512))
    plane = bpy.context.object
    plane.scale = (6.49, 7.954, 1.0)
    material = bpy.data.materials["Material.005"]
    plane.active_material = material

    bpy.ops.mesh.primitive_plane_add(location=(0, 2.0, 0))
    plane = bpy.context.object
    plane.rotation_euler.x = np.pi / 2
    plane.scale = (6.49, 7.954, 1.0)
    material = bpy.data.materials["Material.004"]
    plane.active_material = material

    bpy.ops.mesh.primitive_plane_add(location=(-6.38681, -3.43925, 0))
    plane = bpy.context.object
    plane.rotation_euler.x = np.pi / 2
    plane.rotation_euler.z = np.pi / 2
    plane.scale = (6.49, 7.954, 1.0)
    material = bpy.data.materials["Material.004"]
    plane.active_material = material

    bpy.ops.mesh.primitive_plane_add(location=(6.38681, -3.43925, 0))
    plane = bpy.context.object
    plane.rotation_euler.x = np.pi / 2
    plane.rotation_euler.z = np.pi / 2
    plane.scale = (6.49, 7.954, 1.0)
    material = bpy.data.materials["Material.004"]
    plane.active_material = material

    def place_object(obj):
        obj.location.x = random.uniform(-0.3, 0.3)
        obj.location.y = random.uniform(-0.35, 0.35)
        obj.location.z = 0.024
        obj.rotation_euler.x = 0
        obj.rotation_euler.z = random.uniform(0, math.radians(360))

    imported_objects = []
    for obj_path in selected_objs:
        bpy.ops.wm.obj_import(filepath=obj_path)
        imported_obj = bpy.context.selected_objects[0]
        place_object(imported_obj)
        imported_objects.append(imported_obj)

    loc, rot = random.choice(camera_locations_rotations)
    bpy.ops.object.camera_add(location=loc, rotation=rot)
    depth_cam = bpy.context.object
    depth_cam.name = "Depth_camera"
    depth_cam.data.name = "Depth_camera"
    depth_cam.data.type = "PERSP"

    loc, rot = random.choice(camera_locations_rotations)
    bpy.ops.object.camera_add(location=loc, rotation=rot)
    rgb_cam = bpy.context.object
    rgb_cam.name = "RGB_camera"
    rgb_cam.data.name = "RGB_camera"
    rgb_cam.data.type = "PERSP"
    bpy.context.scene.camera = rgb_cam

    bpy.ops.object.light_add(type="POINT", location=(1.45405, -5.35799, 5.32072))
    light = bpy.context.object
    light.data.energy = 4000.0

    rgb_cam = bpy.data.objects.get("RGB_camera")
    depth_cam = bpy.data.objects.get("Depth_camera")

    if rgb_cam is None or depth_cam is None:
        print("Cameras not found!")
    else:
        T_dc_opt = mathutils.Matrix(
            (
                (0.994052, 0.002774, 0.005608, -32.665543 / 1000.0),
                (-0.003367, 0.994064, 0.108743, -0.986931 / 1000.0),
                (-0.005723, -0.108760, 0.994054, 2.863724 / 1000.0),
                (0.0, 0.0, 0.0, 1.0),
            )
        )
        opt2b = mathutils.Matrix(
            (
                (-1, 0, 0, 0),
                (0, 1, 0, 0),
                (0, 0, -1, 0),
                (0, 0, 0, 1),
            )
        )

        T_bl = opt2b @ T_dc_opt @ opt2b

        T_depth2color_opt = T_bl

        rgb_cam.parent = None
        rgb_cam.matrix_world = depth_cam.matrix_world @ T_depth2color_opt

        depth_world = depth_cam.matrix_world
        rgb_world = rgb_cam.matrix_world
        T_blender = depth_world.inverted() @ rgb_world

        W_rgb, H_rgb = rgb_calib_data["resolution"][0], rgb_calib_data["resolution"][1]
        fx_rgb, fy_rgb = rgb_calib_data["fx"], rgb_calib_data["fy"]
        fov_x_rgb = 2 * math.atan(W_rgb / (2 * fx_rgb))
        fov_y_rgb = 2 * math.atan(H_rgb / (2 * fy_rgb))

        W_d, H_d = depth_calib_data["resolution"][0], depth_calib_data["resolution"][1]
        fx_d, fy_d = depth_calib_data["fx"], depth_calib_data["fy"]
        f_x_mm_d = depth_calib_data["fx"] * pixel_size_depth
        f_y_mm_d = depth_calib_data["fy"] * pixel_size_depth

        bpy.data.cameras["Depth_camera"].type = "PANO"
        bpy.data.cameras["Depth_camera"].panorama_type = "FISHEYE_EQUISOLID"
        bpy.data.cameras["Depth_camera"].fisheye_fov = math.radians(120)
        bpy.data.cameras["Depth_camera"].fisheye_lens = f_x_mm_d
        bpy.data.cameras["Depth_camera"].sensor_height = sensor_height_depth
        bpy.data.cameras["Depth_camera"].sensor_width = sensor_width_depth
        bpy.data.cameras["Depth_camera"].shift_x = -shitf_x_depth
        bpy.data.cameras["Depth_camera"].shift_y = shitf_y_depth
        bpy.data.cameras["Depth_camera"].clip_start = 0.1
        bpy.data.cameras["Depth_camera"].clip_end = 10.0

        bpy.data.cameras["RGB_camera"].sensor_height = sensor_height_rgb
        bpy.data.cameras["RGB_camera"].sensor_width = sensor_width_rgb
        bpy.data.cameras["RGB_camera"].lens_unit = "FOV"
        bpy.data.cameras["RGB_camera"].angle_x = fov_x_rgb
        bpy.data.cameras["RGB_camera"].angle_y = fov_y_rgb
        bpy.data.cameras["RGB_camera"].shift_x = -shitf_x_rgb
        bpy.data.cameras["RGB_camera"].shift_y = shitf_y_rgb
        bpy.data.cameras["RGB_camera"].clip_start = 0.1
        bpy.data.cameras["RGB_camera"].clip_end = 10.0

    # RGB render
    bpy.context.scene.camera = bpy.data.objects["RGB_camera"]
    rgb_resolution = rgb_calib_data["resolution"]
    bpy.context.scene.render.resolution_x = rgb_resolution[0]
    bpy.context.scene.render.resolution_y = rgb_resolution[1]
    bpy.context.scene.render.image_settings.color_mode = "RGB"
    bpy.context.scene.use_nodes = False
    bpy.context.scene.render.filepath = str(RENDER_PATH / f"rgb_{i}.png")
    bpy.ops.render.render(write_still=True)

    # Depth render
    bpy.context.scene.camera = bpy.data.objects["Depth_camera"]
    depth_resolution = depth_calib_data["resolution"]
    bpy.context.scene.render.resolution_x = depth_resolution[0]
    bpy.context.scene.render.resolution_y = depth_resolution[1]
    bpy.context.scene.use_nodes = True
    bpy.context.scene.render.image_settings.color_mode = "BW"
    bpy.context.scene.render.filepath = str(RENDER_PATH / f"depth_{i}.png")
    bpy.ops.render.render(write_still=True)
