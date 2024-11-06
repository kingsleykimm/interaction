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
