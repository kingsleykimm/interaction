# Following the ECCV navigation tutorial

import math
import os
import random

import git
import imageio
import magnum as mn
import numpy as np

from matplotlib import pyplot as plt

# function to display the topdown map
from PIL import Image

import habitat_sim
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut
from habitat.utils.visualizations import maps
data_path = "data"
test_scene = os.path.join(
    data_path, "scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb"
)

mp3d_scene_dataset = os.path.join(data_path + "scene_datasets/mp3d_example/mp3d.scene_dataset_config.json")

rgb_sensor = True
depth_sensor = True
semantic_sensor = True

sim_settings = {
    "scene": test_scene,  # Scene path
    "scene_dataset" : mp3d_scene_dataset,
    "default_agent": 0,  # Index of the default agent
    "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
    "width": 256,  # Spatial resolution of the observations
    "height": 256,
    "color_sensor" : rgb_sensor,
    "semantic_sensor" : semantic_sensor,
    "depth_sensor" : depth_sensor,
    "seed" : 1, # for random navigation
    "enable_physics" : False # interesting -- for later use
}



# one for the simulator backend
# one for the agent, where you can attach a bunch of sensors
def display_sample(rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([])):
    from habitat_sim.utils.common import d3_40_colors_rgb

    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

    arr = [rgb_img]
    titles = ["rgb"]
    if semantic_obs.size != 0:
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        arr.append(semantic_img)
        titles.append("semantic")

    if depth_obs.size != 0:
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
        arr.append(depth_img)
        titles.append("depth")

    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)
    plt.show(block=False)

def display_map(topdown_map, key_points=None):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map
    if key_points is not None:
        for point in key_points:
            plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
    plt.show(block=False)


def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.scene_dataset_config_file = settings["scene_dataset"]
    sim_cfg.enable_physics = settings["enable_physics"]

    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    sensor_specs = []
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(rgb_sensor_spec)

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(depth_sensor_spec)

    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_senesor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(semantic_sensor_spec)
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward" : habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left" : habitat_sim.agent.ActionSpec(
            "turn_left" , habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        "turn_right" : habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
        )
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


cfg = make_simple_cfg(sim_settings)
sim = habitat_sim.Simulator(cfg)

# so up to now, we've set up the simulator and are trying to work with it now

# initialize the agent in the sim
agent = sim.initialize_agent(sim_settings["default_agent"]) # only putting in the index of the agent
agent_state = habitat_sim.AgentState() # agent state only needs a position and a rotation
agent_state.position = np.array([-0.6, 0.0, 0.0])
agent.set_state(agent_state)

# just setting up the agent's positions
agent_state = agent.get_state()
print("agent_state position", agent_state.position, "agent_state rotation", agent_state.rotation)

action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
print("Default action space", action_names)

# All navigation is done through the nav module

height = sim.pathfinder.get_bounds()[0][1]
meters_per_pixel = 0.1  # resolution

if not sim.pathfinder.is_loaded:
    print("Pathfinder not loaded, aborting")

else:
    sim_topdown_map = sim.pathfinder.get_topdown_view(meters_per_pixel, height)
    habitat_topdown_map = maps.get_topdown_map(
        sim.pathfinder, height, meters_per_pixel=meters_per_pixel
    )
    recolor_map = np.array(
        [255, 255, 255], [128, 128, 128], [0, 0, 0], dtype=np.uint8
    ) # this looks like a color palette for wall, floor and obstacle

    topdown_map = recolor_map[habitat_topdown_map]
    
    print("Clean map of topdown map")
    display_map(topdown_map)

    map_filename = os.path.join(os.getcwd(), "/top_down_map.png")
    imageio.imsave(map_filename, topdown_map)

# start queries on navmesh now after verification of topdown map

if not sim.pathfinder.is_loaded:
    print("Pathfinder not loaded, aborting")
else:

    seed = random.seed()
    sim.pathfinder.seed(seed)

    sample1 = sim.pathfinder.get_random_navigable_point()    
    sample2 = sim.pathfinder.get_random_navigable_point()    

    path = habitat_sim.ShortestPath()
    path.requested_start = sample1
    path.requested_end = sample2

    found_path = sim.pathfinder.find_path(path)

    geo_dist = found_path.geodesic_distance
    path_points = path.points

    if found_path:
        meters_per_pixel = 0.025
        scene_bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
        height = scene_bb.y().min

