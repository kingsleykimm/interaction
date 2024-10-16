import numpy as np
import magnum as mn
import habitat
import math

"""
This class takes in a config file / yaml and runs the amount of scenarios and saves them to a file (for data collection purposes).
"""
class SocialNavScenario:

    def __init__(self, human_first, gesture, language, num_goals, iterations):
        self.human_first = human_first
        self.gesture = gesture
        self.language = language
        self.num_goals = num_goals
        self.iterations = iterations
    def setup_config(self, file_name):
        # file_name is a yaml file
        pass

    def setup_env(self, env, observations, config, 
                  ):
        # config will contain humanoidrearrangecontroller, the goal object, everything starting from the while loop
        self.config = config
        self.target = config["goal_object"]
        self.controller = config["humanoid_controller"]
        self.env = env
        self.observations = observations
    def run_scenario_once(self):
        obs = self.env.reset()
        final_targ = self.final_targ
        human_base_transformation = self.env.sim.agents_mgr[0].articulated_agent.base_transformation
        human_seen = False
        while obs["agent_0_has_finished_oracle_nav"] == 0:
            action_dict = {
                "action": ("agent_0_navigate_action", "agent_1_navigate_action"), 
                "action_args": {
                    "agent_0_oracle_nav_lookat_action": self.target,
                    "agent_0_mode": 1,
                    "agent_1_oracle_nav_lookat_action" : self.target,
                    "agent_1_mode" : 1,
                }
            }
            obs = self.env.step(action_dict)
            self.observations.append(obs)
            # TODO: should change hardcoded human_id to config.human_id
            if self.detect_humanoid(obs['agent_1_articulated_agent_arm_panoptic'], self.config.ratio, self.config.pixel_threshold, 100) and not human_seen:
                human_seen = True                                           
                # issue: need to get more of human in frame
                if not self.human_first:
                    # human slows down

                    if self.gesture:
                        # implement  gesture here
                        self.gesture_ahead(self.env, self.controller, self.observations) # human gestures
                    self.agent_first(1, final_targ)
                else:
                    # human speeds up
                    self.human_speed_up()
                        



                
                # fix: rotate the robot more to the left, or get the headings of both agents / use the robot sensor to find the human's location
    def gesture_ahead(self, env : habitat.Env):
        hand_pose = env.sim.agents_mgr[0].articulated_agent.ee_transform(1).translation # EE stands for end effector
        # we need to keep updating the base by using the articulated_agent.base_transformation
        human_transformation = env.sim.agents_mgr[0].articulated_agent.base_transformation
        self.controller.obj_transform_base = human_transformation
        rel_pos = np.array(env.sim.agents_mgr[1].articulated_agent.base_pos) - np.array(env.sim.agents_mgr[0].articulated_agent.base_pos)
        self.controller.calculate_turn_pose(mn.Vector3(rel_pos[0], 0, rel_pos[2]))
        new_pose = self.controller.get_pose()
        action_dict = {
            "action": "agent_0_humanoid_joint_action",
            "action_args": {"agent_0_human_joints_trans": new_pose}
        }
        self.observations.append(env.step(action_dict))
        gesture_steps = 15
        for i in range(gesture_steps):
            # This computes a pose that moves the agent to relative_position
            hand_pose = hand_pose + mn.Vector3(0, 0, 0.05)
            human_transformation = env.sim.agents_mgr[0].articulated_agent.base_transformation
            self.controller.obj_transform_base = human_transformation
            self.controller.calculate_reach_pose(hand_pose, index_hand=1)
            # # The get_pose function gives as a humanoid pose in the same format as HumanoidJointAction
            new_pose = self.controller.get_pose()
            action_dict = {
                "action": "agent_0_humanoid_joint_action",
                "action_args": {"agent_0_human_joints_trans": new_pose}
            }
            obs = env.step(action_dict)
            self.observations.append(obs)
    def add_language(self):
        # adds language to observations, otherwise an empty stringp
        pass

    def detect_humanoid(self, env : habitat.Env, cur_panoptic_observation, ratio, pixel_threshold, human_id):
    # get current height and width of humanoid sensor
        height, width = cur_panoptic_observation.shape[:2]
        # take a bounding box ratio, we aren't going to consider this outer bow
        u_height, l_height = math.floor(height * ratio), math.ceil(height * (1 - ratio))
        u_width, l_width = math.floor(width * ratio), math.ceil(width * (1 - ratio))
        bounded_obs : np.array = cur_panoptic_observation[l_height:u_height, l_width:u_width]
        human_id_sum = np.sum(bounded_obs == human_id)
        return human_id_sum > pixel_threshold
    def agent_first(self, agent_no, final_targ):
    
        for i in range(30):
            action_dict = {
                'action' : f"agent_{agent_no}_navigate_action",
                "action_args" : {
                    f"agent_{agent_no}_oracle_nav_lookat_action" : final_targ,
                    f"agent_{agent_no}_mode" : 1
                }
            }
            obs = self.step(action_dict)
            self.observations.append(obs)