import os
import pickle
import numpy as np
import magnum as mn
import habitat
import math
from datetime import datetime
from habitat.articulated_agent_controllers import HumanoidRearrangeController
from utils import *
from env_setup import *

DEFAULT_WALK_POSE_PATH = "data/humanoids/humanoid_data/male_0/male_0_motion_data_smplx.pkl"

"""
This class takes in a config file / yaml and runs the amount of scenarios and saves them to a file (for data collection purposes).
"""
class SocialNavScenario:

    def __init__(self, config):
        self.config = config
        self.human_first = self.config.human_first
        self.gesture = self.config.gesture
        self.language = self.config.language
        self.num_goals = self.config.num_goals
        self.iterations = self.config.iterations
        self.save_path = self.config.save_path
        self.dataset = []
        agent_dict, action_dict = agent_action_setup()
        self.env = env_setup(agent_dict, action_dict)
        self.controller : HumanoidRearrangeController = HumanoidRearrangeController(DEFAULT_WALK_POSE_PATH)
    def record_data(self):
        
        num_targets = len(self.env.sim.scene_obj_ids)
        self.num_goals = min(self.num_goals, num_targets)
        obj_ids = self.env.sim.scene_obj_ids
        # decide target here
        for goal in range(self.num_goals):
            rom = self.env.sim.get_rigid_object_manager()
            obj_id = obj_ids[goal]
            first_object = rom.get_object_by_id(obj_id)
            object_trans = first_object.translation
            for iteration in self.iterations:
                # run the random spawning method
                # also a scenario is only interesting if we ever see the human
                #TODO :target
                if self.env.episode_over:
                    self.env.reset()
                success = self.run_scenario_once(object_trans.translation, iteration)
                if success:
                    cur_date = datetime.now().strftime("%Y-%m-%d")
                    save_fname = os.path.join(self.save_path, f"target_{first_object.handle}_iteration_{iteration}_date_{cur_date}.pkl")
                    with open(save_fname, "wb") as f:
                        pickle.dump(self.dataset, f)
                    vut.make_video(
                    self.observations,
                    "agent_1_articulated_agent_arm_rgb",
                    "color",
                    save_fname.replace(".pkl", ".mp4"),
                    open_vid=False,
                    )
            self.dataset = []

    def run_scenario_once(self, target, iteration_num) -> bool:
        """Scenario run, returns true if human was seen and successful data log"""
        human_base_transformation = self.env.sim.agents_mgr[0].articulated_agent.base_transformation
        self.humanoid_controller.reset(human_base_transformation)
        # random location instantation
        possible_human_pos = self.env.sim.pathfinder.snap_point(self.env.sim.pathfinder.get_random_navigable_point_near(circle_center=np.array(target), radius=8))
        possible_robot_pos = self.env.sim.pathfinder.snap_point(self.env.sim.pathfinder.get_random_navigable_point_near(circle_center=np.array(target), radius=8))
        self.env.sim.agents_mgr[1].articulated_agent.base_pos = possible_robot_pos
        self.env.sim.agents_mgr[0].articulated_agent.base_pos = possible_human_pos
        human_seen = False
        obs = self.env.reset()
        self.observations = [obs]
        while obs["agent_0_has_finished_oracle_nav"] == 0 or obs["agent_1_has_finished_oracle_nav"] == 0:
            action_dict = {
                "action": ("agent_0_navigate_action", "agent_1_navigate_action"), 
                "action_args": {
                    "agent_0_oracle_nav_lookat_action": target,
                    "agent_0_mode": 1,
                    "agent_1_oracle_nav_lookat_action" : target,
                    "agent_1_mode" : 1,
                }
            }
            obs = self.env.step(action_dict)
            self.observations.append(obs)
            if self.env.episode_over:
                return False
            # TODO: should change hardcoded human_id to config.human_id
            if self.detect_humanoid(obs['agent_1_articulated_agent_arm_panoptic'], self.config.ratio, self.pixel_threshold, 100) and not human_seen:
                human_seen = True
                action = self.slow_down_agents(True)                                
                # issue: need to get more of human in frame
                if not self.human_first:
                    # human slows down
                    if self.gesture:
                        # implement  gesture here
                        self.gesture_ahead(self.env, self.controller, target) # human gestures
                    self.agent_first(1, target) # robot goes first
                else:
                    if self.gesture:
                        self.gesture_stop(self.env, self.controller, target) # TODO
                    # human speeds up, robot keeps stopping then we continue
                    self.agent_first(0, target) # human goes first
        
        # self.observations now contains one episode of observations, we need to figure how to record all of these
        # i think just having a dataset dictionary, with a 2d array?
        if human_seen:
            self.dataset.append(self.observations) # don't do video format
            return True
        return False
                
                # fix: rotate the robot more to the left, or get the headings of both agents / use the robot sensor to find the human's location
    def gesture_ahead(self, env : habitat.Env, humanoid_controller, target):
        target = np.array(target)
        hands = [np.array(env.sim.agents_mgr[0].articulated_agent.ee_transform(0).translation), np.array(env.sim.agents_mgr[0].articulated_agent.ee_transform(1).translation)]
        distances = [np.linalg.norm(target - hand) for hand in hands]
        if distances[0] < distances[1]:
            hand_pose = env.sim.agents_mgr[0].articulated_agent.ee_transform(0).translation
            hand = 0
        else:
            hand_pose = env.sim.agents_mgr[0].articulated_agent.ee_transform(1).translation # EE stands for end effector
            hand = 1
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
        self.observations.append(env.step(action_dict))
        for i in range(10): # 10 ms
            self.observations.append(env.step(action_dict))
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
            self.observations.append(obs)
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
            self.observations.append(obs)
    def gesture_stop(self):
        pass
    def add_language(self):
        # adds language to observations, otherwise an empty stringp
        pass    
    def slow_down_agents(self, human):
        if human:
            cur_human_transformation = self.env.sim.agents_mgr[0].articulated_agent.base_transformation
            cur_pos = self.env.sim.agents_mgr[0].articulated_agent.base_pos
            points = get_next_closest_point(cur_pos, self.target, self.env.sim.pathfinder)
            next_targ = points[1]
            vector = next_targ - cur_pos
            vector = vector[[0, 2]]
            self.humanoid_controller.obj_transform_base = cur_human_transformation
            self.humanoid_controller.calculate_walk_pose(mn.Vector3(vector[0], 0, vector[1]), 0.01)
            humanoid_action = self.humanoid_controller.get_pose()
            action_dict = {
                "action": "agent_0_humanoid_joint_action",
                "action_args": {"agent_0_human_joints_trans" : humanoid_action 
                                }
                }
            self.observations.append(self.env.step(action_dict))
            return humanoid_action
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

