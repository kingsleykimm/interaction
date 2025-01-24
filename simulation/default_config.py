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
                                                    ),
    "base_vel_action" : BaseVelocityActionConfig()
}
