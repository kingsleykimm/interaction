import matplotlib.pyplot as plt
import os
from functools import partial
from habitat_sim.utils import viz_utils as vut
import numpy as np
from habitat.utils.visualizations import maps
from habitat_sim.nav import ShortestPath
from huggingface_hub import HfApi
import pickle

def convert_points_to_topdown(pathfinder, points, meters_per_pixel):
    points_topdown = []
    bounds = pathfinder.get_bounds()
    for point in points:
        # convert 3D x,z to topdown x,y
        px = (point[0] - bounds[0][0]) / meters_per_pixel
        py = (point[2] - bounds[0][2]) / meters_per_pixel
        points_topdown.append(np.array([px, py]))
    return points_topdown
def display_agents(env, agent_pos):
    """Agent_pos should be in the format [possible_human_pos, possible_robot_pos]"""
    xy_vis_points = convert_points_to_topdown(
            env.sim.pathfinder, agent_pos, meters_per_pixel=0.01
        )
    top_down_map = maps.get_topdown_map(
        env.sim.pathfinder, height=agent_pos[1], meters_per_pixel=0.01
    )
    recolor_map = np.array(
        [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
    )
    top_down_map = recolor_map[top_down_map]
    display_map(top_down_map, xy_vis_points)

# display a topdown map with matplotlib
def display_map(topdown_map, key_points=None):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map
    if key_points is not None:
        for point in key_points:
            plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
    plt.show()

def record_video(observations, sensor_name, sensor_type, video_name):
    vut.make_video(
        observations, 
        sensor_name,
        sensor_type,
        video_name,
        open_vid=False,
    )

def record_images(nav_observations):
    observation_to_image = partial(vut.observation_to_image, depth_clip=10.0)
    save_dir = os.path.join(os.getcwd(), "gesture_ahead")
    os.makedirs(save_dir, exist_ok=True)
    for ind, ob in enumerate(nav_observations):
        img = vut.make_video_frame(
            ob,
            "agent_1_articulated_agent_arm_rgb",
            "color",
            video_dims=None,
            overlay_settings=None,
            observation_to_image=observation_to_image,
        )
        img.save(os.path.join(save_dir, f"{ind}.png"))

def simulate(sim, dt, get_observations=False):
    r"""Runs physics simulation at 60FPS for a given duration (dt) optionally collecting and returning sensor observations."""
    # From the test/test_humanoid.py file
    observations = []
    target_time = sim.get_world_time() + dt
    while sim.get_world_time() < target_time:
        sim.step_physics(0.1 / 60.0)
        if get_observations:
            observations.append(sim.get_sensor_observations())
    return observations

def get_next_closest_point(agent_pos, final_targ, pathfinder):
    path = ShortestPath()
    path.requested_start = agent_pos
    path.requested_end = final_targ
    found_path = pathfinder.find_path(path)
    if not found_path:
        return [agent_pos, final_targ]
    return path.points


def create_scenario_dataset():
    api = HfApi()
    api.upload