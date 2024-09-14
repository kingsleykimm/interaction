
import argparse
import os
import os.path as osp
import time
from collections import defaultdict
from typing import Any, Dict, List
import imageio
import pickle
import magnum as mn
import numpy as np
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from habitat.articulated_agents.humanoids.kinematic_humanoid import KinematicHumanoid
from habitat.articulated_agent_controllers.humanoid_rearrange_controller import HumanoidRearrangeController
from habitat.core.utils import try_cv2_import
from habitat.config.default_structured_configs import HumanoidJointActionConfig, HumanoidPickActionConfig, OracleNavActionConfig
from habitat.config.default_structured_configs import TaskConfig, EnvironmentConfig, HabitatConfig, SimulatorConfig, DatasetConfig, AgentConfig
from habitat.config.default_structured_configs import ThirdRGBSensorConfig, HeadRGBSensorConfig, HumanoidDetectorSensorConfig, HeadDepthSensorConfig, ArmPanopticSensorConfig, HeadingSensorConfig, ArmRGBSensorConfig, ArmDepthSensorConfig
from habitat.utils.visualizations import maps
import habitat
from habitat.core.env import Env
from habitat_sim.utils import viz_utils as vut
from habitat_sim.nav import ShortestPath


cv2 = try_cv2_import()

DEFAULT_WALK_POSE_PATH = "data/humanoids/humanoid_data/female_2/female_2_motion_data_smplx.pkl"
HUMANOID_ACTION_DICT = {
    "humanoid_joint_action": HumanoidJointActionConfig(),
    "navigate_action": OracleNavActionConfig(type="OracleNavCoordinateAction", 
                                                      motion_control="human_joints",
                                                      spawn_max_dist_to_obj=1.0),
    "humanoid_pick_obj_id_action": HumanoidPickActionConfig(type="HumanoidPickObjIdAction"),
}

ROBOT_DEFAULT_DICT = {
    "navigate_action" : OracleNavActionConfig(type="OracleNavCoordinateAction",
                                                    motion_control="base_velocity",
                                                    )
}

def get_scenario_data(file_path):
    with open(file_path, "rb") as inp:
        observations = pickle.load(inp)
    print(observations)
    return observations


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

def display_map(topdown_map, key_points=None):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    # plot points on map

def make_sim_cfg(agent_dict):
    sim_cfg = SimulatorConfig(type="RearrangeSim-v0", seed=20)
    sim_cfg.habitat_sim_v0.enable_physics = True
    sim_cfg.habitat_sim_v0.enable_hbao = True

    sim_cfg.scene = "data/hab3_bench_assets/hab3-hssd/scenes/103997919_171031233.scene_instance.json"
    sim_cfg.scene_dataset = "data/hab3_bench_assets/hab3-hssd/hab3-hssd.scene_dataset_config.json"
    sim_cfg.additional_object_paths = ['data/objects/ycb/configs/']

    cfg = OmegaConf.create(sim_cfg)
    cfg.agents = agent_dict # sets up agents
    cfg.agents_order = list(agent_dict.keys())


    return cfg

def make_hab_cfg(agent_dict, action_dict):
    sim_cfg = make_sim_cfg(agent_dict)
    task_config = TaskConfig(type="RearrangeEmptyTask-v0")
    task_config.actions = action_dict # setups the actions that we can use, we can import these from structuredconfigs as well
    task_config.lab_sensors = {
        'humanoid_detector' : HumanoidDetectorSensorConfig(human_pixel_threshold=1500), # these get assigned to each agent, so remember to prepend with the agent_{id}
        'heading_sensor' : HeadingSensorConfig()
    }
    dataset_cfg = DatasetConfig(
        type="RearrangeDataset-v0",
        data_path="data/hab3_bench_assets/episode_datasets/small_medium.json.gz",
    )
    env_cfg = EnvironmentConfig()
    
    habitat_cfg = HabitatConfig()
    habitat_cfg.simulator = sim_cfg
    habitat_cfg.environment = env_cfg
    habitat_cfg.dataset = dataset_cfg
    habitat_cfg.task = task_config
    habitat_cfg.simulator.seed = habitat_cfg.seed

    return habitat_cfg

def env_setup(agent_dict, action_dict): # pass in config from yaml
    hab_cfg = make_hab_cfg(agent_dict, action_dict)
    res_cfg = OmegaConf.create(hab_cfg)
    return Env(res_cfg)

def agent_action_setup():
    agent_dict = {}
    main_agent_config = AgentConfig()
    main_agent_config.articulated_agent_type = 'KinematicHumanoid'
    main_agent_config.articulated_agent_urdf = 'data/hab3_bench_assets/humanoids/female_0/female_0.urdf'
    main_agent_config.motion_data_path = 'data/hab3_bench_assets/humanoids/female_0/female_0_motion_data_smplx.pkl'
    main_agent_config.sim_sensors = {
        'third_rgb' : ThirdRGBSensorConfig(position=[0.0, 0.25, 0.0]),
        'head_rgb' : HeadRGBSensorConfig(),
    }
    agent_dict["agent_0"] = main_agent_config
    action_dict = HUMANOID_ACTION_DICT
    new_action_dict = {}
    for key in action_dict:
        new_key = f"agent_0_{key}"
        new_action_dict[new_key] = action_dict[key]
    
    robot_agent_config = AgentConfig()
    robot_agent_config.articulated_agent_type = "SpotRobot"
    robot_agent_config.articulated_agent_urdf = 'data/robots/hab_spot_arm/urdf/hab_spot_arm.urdf'
    robot_agent_config.sim_sensors = {
        'third_rgb' : ThirdRGBSensorConfig(),
        'head_depth' : HeadDepthSensorConfig(),
        'arm_panoptic' : ArmPanopticSensorConfig(),
        'arm_rgb' : ArmRGBSensorConfig(height=400, width=400),
        'arm_depth' : ArmDepthSensorConfig(),
    }
    robot_agent_config.start_position = [3.0, 0, 0]
    agent_dict["agent_1"] = robot_agent_config
    robot_action_dict = ROBOT_DEFAULT_DICT
    for key in robot_action_dict:
        new_key = f"agent_1_{key}"
        new_action_dict[new_key] = robot_action_dict[key]
    
    return agent_dict, new_action_dict

def display_env_topdown_map(env : habitat.Env):
    hablab_topdown_map = maps.get_topdown_map(
        env.sim.pathfinder, env.sim.pathfinder.get_bounds()[0][1], 
    )
    recolor_map = np.array(
            [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
        )
    hablab_topdown_map = recolor_map[hablab_topdown_map]
    display_map(hablab_topdown_map)
    mp_filename = os.path.join(os.getcwd(), 'map.png')
    imageio.imsave(mp_filename, hablab_topdown_map)

def gesture_ahead(env : habitat.Env, humanoid_controller, observations):
    offset =  env.sim.agents_mgr[0].articulated_agent.base_transformation.transform_vector(mn.Vector3(0, 0.3, 0))
    hand_pose = env.sim.agents_mgr[0].articulated_agent.ee_transform(0).translation + offset # EE stands for end effector
    prev_heading = None
    hand_pose = hand_pose + mn.Vector3(0,0,0.5)
    humanoid_controller.calculate_reach_pose(hand_pose, index_hand=1)
    new_pose = humanoid_controller.get_pose()
    action_dict = {
        "action": "agent_0_humanoid_joint_action",
        "action_args": {"agent_0_human_joints_trans": new_pose}
    }
    obs = env.step(action_dict)
    for _ in range(90):
        # This computes a pose that moves the agent to relative_position
        hand_pose = hand_pose + mn.Vector3(0, 0, 0.01)
        
        humanoid_controller.calculate_reach_pose(hand_pose, index_hand=1)
        # # The get_pose function gives as a humanoid pose in the same format as HumanoidJointAction
        new_pose = humanoid_controller.get_pose()
        action_dict = {
            "action": "agent_0_humanoid_joint_action",
            "action_args": {"agent_0_human_joints_trans": new_pose}
        }
        obs = env.step(action_dict)
        observations.append(obs)

def get_next_closest_point(agent_pos, final_targ, pathfinder):
    path = ShortestPath()
    path.requested_start = agent_pos
    path.requested_end = final_targ
    found_path = pathfinder.find_path(path)
    if not found_path:
        return [agent_pos, final_targ]
    return path.points

def agent_first(agent_no, final_targ, env, observations):
    
    for i in range(45):
        action_dict = {
            'action' : f"agent_{agent_no}_navigate_action",
            "action_args" : {
                f"agent_{agent_no}_oracle_nav_lookat_action" : final_targ,
                f"agent_{agent_no}_mode" : 1
            }
        }
        obs = env.step(action_dict)
        observations.append(obs)



def play_env(env : habitat.Env) -> None: # we are in the environment at this point
    # at this point, the simulator should be set up and embedded into env
    observations = []
    # Set up humaonid here
    # humanoid = KinematicHumanoid(get_agent_config(sim_config=config.habitat.simulator), env.sim)
    # humanoid.reconfigure()
    # humanoid.update()

    # humanoid_controller = HumanoidRearrangeController(DEFAULT_WALK_POSE_PATH)
    # humanoid_controller.reset(env.sim.articulated_agent.base_transformation)
    # humanoid_controller.apply_base_transformation(env.sim.articulated_agent.base_transformation)

    rom = env.sim.get_rigid_object_manager() # manager for all current objects, we're going to choose one to go 
    obj_id = env.sim.scene_obj_ids[-1]
    first_object = rom.get_object_by_id(obj_id)

    
    humanoid_controller = HumanoidRearrangeController(walk_pose_path=DEFAULT_WALK_POSE_PATH)
    humanoid_controller.reset(env.sim.agents_mgr[0].articulated_agent.base_transformation)

    object_trans = first_object.translation
    print("Object translation ", object_trans)
    print(first_object.handle, "is in", object_trans)
    print(env.sim.pathfinder.get_bounds())

    agent_displ = np.inf
    agent_rot = np.inf
    prev_rot = env.sim.agents_mgr[0].articulated_agent.base_rot # when we use multiple agents we need to use agents_mgr
    prev_pos = env.sim.agents_mgr[0].articulated_agent.base_pos
    base_transformation = env.sim.agents_mgr[0].articulated_agent.base_transformation 
    print(base_transformation, prev_pos)

    # while True:
    #     # point is a np.array

    #     # Robot movement, mostly copied from the OracleNavAction
    #     forward = np.array([1.0, 0.0, 0.0])
    #     base_T = env.sim.agents_mgr[0].articulated_agent.base_transformation
    #     cur_pos = env.sim.agents_mgr[0].articulated_agent.base_pos
    #     next_points = get_next_closest_point(cur_pos, object_trans, env.sim.pathfinder)

    #     if len(next_points) == 1:
    #         next_points += next_points

    #     cur_targ = next_points[1]
        
    #     robot_forward = np.array(base_T.transform_vector(forward))
    #     rel_targ = cur_targ - cur_pos

    #     robot_forward = robot_forward[[0, 2]]
    #     rel_targ = rel_targ[[0, 2]]
    #     rel_pos = (object_trans - cur_pos)[[0, 2]]

    #     angle_to_target = get_angle(robot_forward, rel_targ)
    #     angle_to_obj = get_angle(robot_forward, rel_pos)

    #     dist_to_final_targ = np.linalg.norm(
    #         (object_trans - cur_pos)[[0, 2]]
    #     )

    #     # Human movement
    #     humanoid_controller.calculate_walk_pose(mn.Vector3(point))
    #     new_pose = humanoid_controller.get_pose()
    #     action_dict = {
    #         "action" : ("agent_0_humanoid_joint_action"),
    #         "action_args" : {
    #             "agent_0_human_joints_trans" : new_pose
    #         }
    #     }
    #     obs = env.step(action_dict)
    #     observations.append(obs)
    human_seen = False
    while agent_displ > 1e-9 or agent_rot > 1e-9:
        prev_rot = env.sim.agents_mgr[0].articulated_agent.base_rot # when we use multiple agents we need to use agents_mgr
        prev_pos = env.sim.agents_mgr[0].articulated_agent.base_pos
        action_dict = {
            "action": ("agent_0_navigate_action", "agent_1_navigate_action"), 
            "action_args": {
                "agent_0_oracle_nav_lookat_action": object_trans,
                "agent_0_mode": 1,
                "agent_1_oracle_nav_lookat_action" : object_trans,
                "agent_1_mode" : 1,
            }
        }
        obs = env.step(action_dict)
        observations.append(obs)
        if obs["agent_1_humanoid_detector_sensor"] == np.ones(1, dtype=np.float32) and not human_seen:

            # need to implement a slow down here
            # stopping/slowing down
            human_seen = True
            # humanoid_controller.calculate_walk_pose(mn.Vector3(prev_pos))
            # next_pose = humanoid_controller.get_pose()
            # action_dict = {
            #     "action": "agent_0_humanoid_joint_action",
            #     "action_args": {"agent_0_human_joints_trans": next_pose}
            # }
            # obs = env.step(action_dict)
            # observations.append(obs)
            gesture_ahead(env, humanoid_controller, observations)
            agent_first(1, object_trans, env, observations)
            # make a few methods here, human first, robot first, languages

            
        
        cur_rot = env.sim.agents_mgr[0].articulated_agent.base_rot
        cur_pos = env.sim.agents_mgr[0].articulated_agent.base_pos
        agent_displ = (cur_pos - prev_pos).length() # these two variables are tracking the distance
        agent_rot = np.abs(cur_rot - prev_rot)
        
    print(len(observations))
    # output_file = os.path.join(os.getcwd(), 'scenario_data.pkl')
    # with open(output_file, 'wb') as outp:
    #     pickle.dump(observations, outp, pickle.HIGHEST_PROTOCOL)

    vut.make_video(
        observations,
        "agent_1_articulated_agent_arm_rgb",
        "color",
        "robot_tutorial_video",
        open_vid=False,
    )
# some more notes:
# in order to step through the environment, we need to pass in an action dictionary with {"action", "action_args"}    
# actions go under the task config yaml

# look at actions for the 
# need to combine aspects of the examples of shortestpathfollower and the pygame set up from the environment

# we should extract the shortest path from shortestpathfollower and then use that to play the entire trajectory

# we want the humanoid to follow the shortestpath




if __name__ == '__main__':
    agent_dict, action_dict = agent_action_setup()
    env = env_setup(agent_dict, action_dict)
    env.reset()
    # display_env_topdown_map(env)

    play_env(env)