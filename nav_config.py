# Notes:

# # to use replica cad:
# sim_cfg.scene_dataset = "data/versioned_data/replica_cad_dataset/replicaCAD.scene_dataset_config.json"
#     sim_cfg.additional_object_paths = ['data/objects/ycb/configs']

# dataset_cfg = DatasetConfig(
#         type="RearrangeDataset-v0",
#         data_path="data/datasets/rearrange_pick/replica_cad/v0/rearrange_pick_replica_cad_v0/empty.json.gz",
#     )

# seed that worked for hab3: 20
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

sim_cfg:
    type: "RearrangeSim-v0"
    seed: 20
    scene: data/hab3_bench_assets/hab3-hssd/scenes/103997919_171031233.scene_instance.json
    scene_dataset: data/hab3_bench_assets/hab3-hssd/hab3-hssd.scene_dataset_config.json
    additional_object_paths:
    habitat_sim_v0: ['data/objects/ycb/configs/']
        enable_physics: True
        enable_hbao: True
agent_cfg:
    - agent_1:
        articulated_agent_type: KinematicHumanoid
        articulated_agent_urdf: data/humanoids/humanoid_data/male_0/male_0.urdf
        motion_data_path: data/humanoids/humanoid_data/male_0/male_0_motion_data_smplx.pkl
        sim_sensors:
            - third_rgb
            - head_rgb
            
        