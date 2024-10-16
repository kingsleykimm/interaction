from habitat.articulated_agents.humanoids import KinematicHumanoid
from habitat.articulated_agent_controllers import HumanoidRearrangeController
import magnum as mn
import numpy as np
class Human(KinematicHumanoid):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
    

class CustomController(HumanoidRearrangeController):
    def __init__(self, walk_pose_path):
        super().__init__(walk_pose_path)
    def manipulate_hand_joint(self, new_quaternion: np.ndarray):
    # Ensure the new quaternion is a valid shape
        assert new_quaternion.shape == (4,), "Quaternion must have shape (4,)"

        # Extract current joint poses (list of quaternions for all joints)
        current_joint_poses = self.joint_pose  # This is a list or array of joint quaternions

        # Select joint 20 (hand joint) and replace its quaternion
        current_joint_poses[20 * 4 : 20 * 4 + 4] = new_quaternion

        # Optionally, normalize the quaternion for joint 20 to ensure it's a valid rotation quaternion
        current_joint_poses[20 * 4 : 20 * 4 + 4] /= np.linalg.norm(current_joint_poses[20 * 4 : 20 * 4 + 4])

        # Update the joint_pose with the modified hand joint
        self.joint_pose = current_joint_poses
    import numpy as np


    def update_hand_rotation(self, current_position: np.ndarray, target_position: np.ndarray):
        """
        Update the hand joint (joint 20) quaternion to reach the target position.
        
        :param current_position: Current 3D position of the hand (joint 20).
        :param target_position: Target 3D position for the hand.
        """
        # Calculate the direction vector
        target_direction = calculate_direction_vector(current_position, target_position)
        
        # Define the hand's current forward direction (this can be along an axis like z-axis)
        current_hand_forward_direction = np.array([1, 0, 0])  # Assuming forward is along the z-axis
        
        # Calculate the rotation quaternion
        hand_quaternion = calculate_rotation_quaternion(current_hand_forward_direction, target_direction)
        
        # Update the hand's quaternion in the joint_pose list (joint 20)
        self.manipulate_hand_joint(hand_quaternion)


def calculate_direction_vector(current_position: np.ndarray, target_position: np.ndarray) -> np.ndarray:
    """
    Calculate the normalized direction vector from the current position to the target position.
    
    :param current_position: The current 3D position of the hand.
    :param target_position: The target 3D position for the hand.
    :return: A normalized direction vector pointing from current to target.
    """
    direction_vector = target_position - current_position
    return direction_vector / np.linalg.norm(direction_vector)  # Normalize the vector

from scipy.spatial.transform import Rotation as R

def calculate_rotation_quaternion(current_direction: np.ndarray, target_direction: np.ndarray) -> np.ndarray:
    """
    Calculate the quaternion that rotates current_direction to align with target_direction.
    
    :param current_direction: The current direction the hand is facing (normalized vector).
    :param target_direction: The target direction the hand should face (normalized vector).
    :return: A quaternion (4D array) that represents the rotation.
    """
    # Ensure both vectors are normalized
    current_direction = current_direction / np.linalg.norm(current_direction)
    target_direction = target_direction / np.linalg.norm(target_direction)

    # Calculate the axis of rotation (cross product) and angle (dot product)
    rotation_axis = np.cross(current_direction, target_direction)
    rotation_angle = np.arccos(np.clip(np.dot(current_direction, target_direction), -1.0, 1.0))

    # Create quaternion from axis and angle
    if np.linalg.norm(rotation_axis) < 1e-6:
        # No rotation needed if the vectors are already aligned
        return np.array([0, 0, 0, 1])  # Identity quaternion

    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)  # Normalize axis

    # Quaternion from axis-angle representation
    rotation_quat = R.from_rotvec(rotation_angle * rotation_axis).as_quat()  # Convert to quaternion
    return rotation_quat
