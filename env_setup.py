
from omegaconf import OmegaConf
from habitat.config.default_structured_configs import HumanoidJointActionConfig, HumanoidPickActionConfig, OracleNavActionConfig
from habitat.config.default_structured_configs import TaskConfig, EnvironmentConfig, HabitatConfig, SimulatorConfig, DatasetConfig, AgentConfig
from habitat.config.default_structured_configs import ThirdRGBSensorConfig, HeadRGBSensorConfig, HumanoidDetectorSensorConfig, HeadDepthSensorConfig, ArmPanopticSensorConfig, HeadingSensorConfig, ArmRGBSensorConfig, ArmDepthSensorConfig, BaseVelocityActionConfig, HumanoidJointSensorConfig
from habitat.core.env import Env

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