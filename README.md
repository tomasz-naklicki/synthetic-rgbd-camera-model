# Depth Simulation Publisher

This repository contains a ROS2 node that publishes simulated RGB and depth image pairs. It is designed to read pre-rendered images from a shared folder, apply processing to better mimic real depth images, and publish them as ROS2 topics.

## Getting Started with Docker

To test the node using Docker:

### 1. Build the Docker Image

Run the provided script:

```bash
./docker_build.sh
```

### 2. Set the Shared Folder
Edit the ```docker_run.sh``` script and set the ```IMAGE_FOLDER``` variable to the path of the folder where your rendered images will be placed.

### 3. Run the Docker Container
Start the container with:
```
./docker_run.sh
```
### 4. Start the ROS2 Node
Inside the Docker container, run:

```
ros2 run depth_simulation_publisher render_publisher
```
### 5. (Optional) Visualize in RVIZ2
In another terminal within the container:

```
ros2 run rviz2 rviz2
```
The images are published on ```/camera/color``` and ```/camera/depth``` topics
### 6. Provide Image Input
Place your image pairs in the shared folder specified earlier. The files must follow this naming convention:

```rgb_N.png```
```depth_N.png```

Where ```N``` is an increasing number. The node will always publish the latest (highest-numbered) pair.

## Notes
The shared folder allows a rendering tool to output images directly where the ROS2 node can access and publish them.

Only the most recent image pair is published at any time.
