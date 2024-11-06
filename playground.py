
import math
import pickle
import magnum as mn
import numpy as np
from omegaconf import OmegaConf
from habitat.articulated_agent_controllers.humanoid_rearrange_controller import HumanoidRearrangeController
from agent_actions import CustomController
from habitat.config.default_structured_configs import HumanoidJointActionConfig, HumanoidPickActionConfig, OracleNavActionConfig
from habitat.config.default_structured_configs import TaskConfig, EnvironmentConfig, HabitatConfig, SimulatorConfig, DatasetConfig, AgentConfig
from habitat.config.default_structured_configs import ThirdRGBSensorConfig, HeadRGBSensorConfig, HumanoidDetectorSensorConfig, HeadDepthSensorConfig, ArmPanopticSensorConfig, HeadingSensorConfig, ArmRGBSensorConfig, ArmDepthSensorConfig, BaseVelocityActionConfig, HumanoidJointSensorConfig, ArmActionConfig
from habitat.utils.visualizations import maps
import habitat
from habitat.core.env import Env
from habitat_sim.utils import viz_utils as vut
from habitat_sim.nav import ShortestPath



DEFAULT_WALK_POSE_PATH = 'data/humanoids/humanoid_data/male_0/male_0_motion_data_smplx.pkl'
HUMANOID_ACTION_DICT = {
    "humanoid_joint_action": HumanoidJointActionConfig(),
    "navigate_action": OracleNavActionConfig(type="OracleNavCoordinateAction", 
                                                      motion_control="human_joints",
                                                      spawn_max_dist_to_obj=-1.0),
    "humanoid_pick_obj_id_action": HumanoidPickActionConfig(type="HumanoidPickObjIdAction"),
}

ROBOT_DEFAULT_DICT = {
    "navigate_action" : OracleNavActionConfig(type="OracleNavCoordinateAction",
                                              spawn_max_dist_to_obj=-1.0,
                                                    motion_control="base_velocity",
                                                    ),
    "base_vel_action" : BaseVelocityActionConfig()
}


def make_sim_cfg(agent_dict):
    sim_cfg = SimulatorConfig(type="RearrangeSim-v0")
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
        'heading_sensor' : HeadingSensorConfig(),
        'joint_sensor' : HumanoidJointSensorConfig(),
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
    main_agent_config.articulated_agent_urdf = 'data/humanoids/humanoid_data/male_0/male_0.urdf'
    main_agent_config.motion_data_path = 'data/humanoids/humanoid_data/male_0/male_0_motion_data_smplx.pkl'
    # main_agent_config.ik_arm_urdf = ('data/humanoids/humanoid_data/male_0/male_0.urdf')
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
        # 'third_rgb' : ThirdRGBSensorConfig(),
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




def gesture_ahead(env : habitat.Env, humanoid_controller, target):
    # offset =  env.sim.agents_mgr[0].articulated_agent.base_transformation.transform_vector(mn.Vector3(0, 0.3, 0))
    # argmin on which hand to use based on distance
    target = np.array(target)
    hands = [np.array(env.sim.agents_mgr[0].articulated_agent.ee_transform(0).translation), np.array(env.sim.agents_mgr[0].articulated_agent.ee_transform(1).translation)]
    distances = [np.linalg.norm(target - hand) for hand in hands]
    if distances[0] < distances[1]:
        hand_pose = env.sim.agents_mgr[0].articulated_agent.ee_transform(0).translation
        hand = 0
    else:
        hand_pose = env.sim.agents_mgr[0].articulated_agent.ee_transform(1).translation # EE stands for end effector
        hand = 1
    observations = []
    # we need to keep updating the base by using the articulated_agent.base_transformation
    humanoid_controller.obj_transform_base = env.sim.agents_mgr[0].articulated_agent.base_transformation
    rel_pos = np.array(env.sim.agents_mgr[1].articulated_agent.base_pos - env.sim.agents_mgr[0].articulated_agent.base_pos)[[0, 2]]
    # rel_pos = np.array(env.sim.agents_mgr[1].articulated_agent.base_pos) - np.array(env.sim.agents_mgr[0].articulated_agent.base_pos) 
    humanoid_controller.calculate_turn_pose(mn.Vector3(rel_pos[0], 0, rel_pos[1]))
    new_pose = humanoid_controller.get_pose()
    action_dict = {
        "action": "agent_0_humanoid_joint_action",
        "action_args": {"agent_0_human_joints_trans": new_pose}
    }
    observations.append(env.step(action_dict))
    for i in range(10): # 10 ms
        observations.append(env.step(action_dict))
    gesture_steps = 30
    hand_vec = np.array(target - hands[hand])[[0, 2]] / 2
    for i in range(gesture_steps//3):
        # This computes a pose that moves the agent to relative_position
        hand_pose = hand_pose + mn.Vector3(hand_vec[0] / gesture_steps, 0, hand_vec[1] / gesture_steps)
        human_transformation = env.sim.agents_mgr[0].articulated_agent.base_transformation
        humanoid_controller.obj_transform_base = human_transformation
        humanoid_controller.calculate_reach_pose(hand_pose, index_hand=hand)
        # # The get_pose function gives as a humanoid pose in the same format as HumanoidJointAction
        new_pose = humanoid_controller.get_pose()
        action_dict = {
            "action": "agent_0_humanoid_joint_action",
            "action_args": {"agent_0_human_joints_trans": new_pose}
        }
        obs = env.step(action_dict)
        observations.append(obs)
    for i in range(gesture_steps//3):
        # This computes a pose that moves the agent to relative_position
        hand_pose = hand_pose + mn.Vector3(-hand_vec[0] / gesture_steps, 0, -hand_vec[1] / gesture_steps)
        human_transformation = env.sim.agents_mgr[0].articulated_agent.base_transformation
        humanoid_controller.obj_transform_base = human_transformation
        humanoid_controller.calculate_reach_pose(hand_pose, index_hand=hand)
        # # The get_pose function gives as a humanoid pose in the same format as HumanoidJointAction
        new_pose = humanoid_controller.get_pose()
        action_dict = {
            "action": "agent_0_humanoid_joint_action",
            "action_args": {"agent_0_human_joints_trans": new_pose}
        }
        obs = env.step(action_dict)
        observations.append(obs)
    return observations

def get_next_closest_point(agent_pos, final_targ, pathfinder):
    path = ShortestPath()
    path.requested_start = agent_pos
    path.requested_end = final_targ
    found_path = pathfinder.find_path(path)
    if not found_path:
        return [agent_pos, final_targ]
    return path.points

def agent_first(agent_no, final_targ, env):
    observations = []
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
    return observations

def detect_humanoid(cur_panoptic_observation, ratio, pixel_threshold, human_id):
    # get current height and width of humanoid sensor
    height, width = cur_panoptic_observation.shape[:2]
    
    # take a bounding box ratio, we aren't going to consider this outer bow
    u_height, l_height = math.floor(height * ratio), math.ceil(height * (1 - ratio))
    u_width, l_width = math.floor(width * ratio), math.ceil(width * (1 - ratio))

    bounded_obs : np.array = cur_panoptic_observation[l_height:u_height, l_width:u_width]
    human_id_sum = np.sum(bounded_obs == human_id)
    # print(bounded_obs)
    return human_id_sum > pixel_threshold

def slow_down_agents(env : habitat.Env, target_pos, humanoid_controller : HumanoidRearrangeController, human):
    if human:
        cur_human_transformation = env.sim.agents_mgr[0].articulated_agent.base_transformation
        cur_pos = env.sim.agents_mgr[0].articulated_agent.base_pos
        points = get_next_closest_point(cur_pos, target_pos, env.sim.pathfinder)
        next_targ = points[1]
        vector = next_targ - cur_pos
        vector = vector[[0, 2]]
        humanoid_controller.obj_transform_base = cur_human_transformation
        humanoid_controller.calculate_walk_pose(mn.Vector3(vector[0], 0, vector[1]), 0.01)
        humanoid_action = humanoid_controller.get_pose()
        return humanoid_action

def gesture_stop(env, humanoid_controller):
    observations = []
    human_transformation = env.sim.agents_mgr[0].articulated_agent.base_transformation
    humanoid_controller.obj_transform_base = human_transformation
    rel_pos = env.sim.agents_mgr[1].articulated_agent.base_pos
    print(rel_pos)
    rel_pos = np.array(env.sim.agents_mgr[1].articulated_agent.base_pos) - np.array(env.sim.agents_mgr[0].articulated_agent.base_pos)
    humanoid_controller.calculate_turn_pose(mn.Vector3(rel_pos[0], 0, rel_pos[2]))
    new_pose = humanoid_controller.get_pose()
    action_dict = {
        "action": "agent_0_humanoid_joint_action",
        "action_args": {"agent_0_human_joints_trans": new_pose}
    }
    observations.append(env.step(action_dict))
    hand_pose = env.sim.agents_mgr[0].articulated_agent.ee_transform(1).translation
    gesture_steps = 50
    # maybe divide by gesture steps, with 3 above
    humanoid_controller.obj_transform_base = human_transformation
    humanoid_controller.calculate_reach_pose(env.sim.agents_mgr[1].articulated_agent.translation, index_hand=1)
    new_pose = humanoid_controller.get_pose()
    action_dict = {
        "action": "agent_0_humanoid_joint_action",
        "action_args": {"agent_0_human_joints_trans": new_pose}
    }
    obs = env.step(action_dict)
    observations.append(obs) 

    # for i in range(gesture_steps // 2):
    #     hand_pose = hand_pose + mn.Vector3(rel_pos[0] / gesture_steps, (rel_pos[1] + 0.6) / gesture_steps, rel_pos[2] / gesture_steps)
    #     human_transformation = env.sim.agents_mgr[0].articulated_agent.base_transformation
    #     humanoid_controller.obj_transform_base = human_transformation
    #     humanoid_controller.calculate_reach_pose(hand_pose, index_hand=1)
    #     new_pose = humanoid_controller.get_pose()
    #     action_dict = {
    #         "action": "agent_0_humanoid_joint_action",
    #         "action_args": {"agent_0_human_joints_trans": new_pose}
    #     }
    #     obs = env.step(action_dict)
    #     observations.append(obs)
    # for i in range(gesture_steps // 2):
    #     hand_pose = hand_pose + mn.Vector3(-rel_pos[0] / gesture_steps, -(rel_pos[1] + 0.6) / gesture_steps, rel_pos[2] / gesture_steps)
    #     human_transformation = env.sim.agents_mgr[0].articulated_agent.base_transformation
    #     humanoid_controller.obj_transform_base = human_transformation
    #     humanoid_controller.calculate_reach_pose(hand_pose, index_hand=1)
    #     new_pose = humanoid_controller.get_pose()
    #     action_dict = {
    #         "action": "agent_0_humanoid_joint_action",
    #         "action_args": {"agent_0_human_joints_trans": new_pose}
    #     }
    #     obs = env.step(action_dict)
    #     observations.append(obs)
    return observations
def play_env(env : habitat.Env) -> None: # we are in the environment at this point
    # at this point, the simulator should be set up and embedded into env
    observations = []
    nav_observations = []

    # humanoid_controller = HumanoidRearrangeController(DEFAULT_WALK_POSE_PATH)
    # humanoid_controller.reset(env.sim.articulated_agent.base_transformation)
    # humanoid_controller.apply_base_transformation(env.sim.articulated_agent.base_transformation)
    

    # for i in range(len(env.sim.scene_obj_ids)):)
    env.reset()
    obj_ids = env.sim.scene_obj_ids
    for i in range(10):
        env.reset()
        observations = []
        rom = env.sim.get_rigid_object_manager() # manager for all current objects, we're going to choose one to go 
        obj_id = env.sim.scene_obj_ids[i % len(env.sim.scene_obj_ids)]
        print(obj_id)
        first_object = rom.get_object_by_id(obj_id)
        object_trans = first_object.translation
        # Let's try to place the human and robot_pos at equidistant places from the object
        possible_human_pos = env.sim.agents_mgr[0].articulated_agent.base_pos
        possible_robot_pos = env.sim.agents_mgr[1].articulated_agent.base_pos
        possible_human_pos = env.sim.pathfinder.snap_point(env.sim.pathfinder.get_random_navigable_point_near(circle_center=np.array(object_trans), radius=8))
        possible_robot_pos = env.sim.pathfinder.snap_point(env.sim.pathfinder.get_random_navigable_point_near(circle_center=np.array(object_trans), radius=8))
        env.sim.agents_mgr[1].articulated_agent.base_pos = possible_robot_pos
        env.sim.agents_mgr[0].articulated_agent.base_pos = possible_human_pos
        humanoid_controller = HumanoidRearrangeController(walk_pose_path=DEFAULT_WALK_POSE_PATH)
        humanoid_controller.reset(env.sim.agents_mgr[0].articulated_agent.base_transformation)
        print("Object translation ", object_trans)
        print(first_object.handle, "is in", object_trans)
        targ = env.sim.pathfinder.snap_point(object_trans)
        agent_displ = np.inf
        agent_rot = np.inf
        prev_rot = env.sim.agents_mgr[0].articulated_agent.base_rot # when we use multiple agents we need to use agents_mgr
        prev_pos = env.sim.agents_mgr[0].articulated_agent.base_pos
        base_transformation = env.sim.agents_mgr[0].articulated_agent.base_transformation 
        print(base_transformation, prev_pos)
        human_seen = False
        if env.episode_over:
            env.reset()
            continue

        while agent_displ > 1e-9 or agent_rot > 1e-9:
            prev_rot = env.sim.agents_mgr[0].articulated_agent.base_rot # when we use multiple agents we need to use agents_mgr
            prev_pos = env.sim.agents_mgr[0].articulated_agent.base_pos
            action_dict = {
                "action": ("agent_0_navigate_action", "agent_1_navigate_action"), 
                "action_args": {
                    "agent_0_oracle_nav_lookat_action": targ,
                    "agent_0_mode": 1,
                    "agent_1_oracle_nav_lookat_action" : targ,
                    "agent_1_mode" : 1,
                }
            }
            if env.episode_over:
                break
            obs = env.step(action_dict)
            observations.append(obs)
            if detect_humanoid(obs['agent_1_articulated_agent_arm_panoptic'], 0.85, 1350, 100) and not human_seen:
                human_seen = True
                action = slow_down_agents(env, targ, humanoid_controller, True)
                action_dict = {
                    "action": "agent_0_humanoid_joint_action",
                    "action_args": {"agent_0_human_joints_trans" : action } 
                    }
                obs = env.step(action_dict)
                observations.append(obs)
                nav_observations.append(obs)

                gesture_obs = gesture_ahead(env, humanoid_controller, targ)
                observations += gesture_obs
                nav_observations += gesture_obs
                agent_first_obs = agent_first(1, targ, env)
                observations += agent_first_obs
                nav_observations += gesture_obs
            cur_rot = env.sim.agents_mgr[0].articulated_agent.base_rot
            cur_pos = env.sim.agents_mgr[0].articulated_agent.base_pos
            agent_displ = (cur_pos - prev_pos).length() # these two variables are tracking the distance
            agent_rot = np.abs(cur_rot - prev_rot)
        if human_seen:
            vut.make_video(
                observations,
                "agent_1_articulated_agent_arm_rgb",
                "color",
                f"test_{i}",
                open_vid=False,
            )
        # env.sim.seed = i

def testing(env : habitat.Env):
    observations = []
    # print(env.sim.agents_mgr[0].ik_helper.get_joint_limits())
    humanoid = env.sim.agents_mgr[0].articulated_agent
    humanoid.reconfigure()
    humanoid.update()
    obs = env.reset()
    observations.append(obs)
    hand_pos = env.sim.agents_mgr[0].articulated_agent.ee_transform(0).translation
    sim_obj = humanoid.sim_obj
    joints, base_transform = humanoid.get_joint_transform()
    custom = CustomController(DEFAULT_WALK_POSE_PATH)
    custom.reset(env.sim.agents_mgr[0].articulated_agent.base_transformation)
    human_transformation = env.sim.agents_mgr[0].articulated_agent.base_transformation
    joints = custom.hand_processed_data['left_hand'][0][0] # this is the joints for the number of poses
    print(len(joints), len(joints[0]))
    print(custom.hand_processed_data['left_hand'][0][0].shape) # 3 * n_poses(257) * 1 * 54
    print(custom.vpose_info)
    # custom.obj_transform_base = human_transformation
    # custom.update_hand_rotation(np.array(hand_pos), np.array(hand_pos + mn.Vector3(0, 0, 0.5)))
    # new_pose = custom.get_pose()
    # print("action")

    # print(new_pose)
    # hand_location = env.sim.agents_mgr[0].articulated_agent.ee_transform(0).translation
    # for i in range(50):
    #     hand_location = hand_location + mn.Vector3(0, 0, 0.05)
    #     action_dict = {
    #         "action": "agent_0_arm_action",
    #         "action_args": {"agent_0_arm_action": hand_location}
    #     }
    #     observations.append(env.step(action_dict))
    # # obs = env.step(action_dict)
    # # observations.append(obs)
    # vut.make_video(
    #         observations,
    #         "agent_0_third_rgb",
    #         "color",
    #         f"videos/hand_test",
    #         open_vid=False,
    #     )
    # ee_links = humanoid._get_humanoid_params().ee_links
    # print(obs["agent_0_humanoid_joint_sensor"] == joints)
    # print(ee_links)
    # print(sim_obj.get_link_scene_node(ee_links[0]).transformation)
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
    # testing(env)
    play_env(env)