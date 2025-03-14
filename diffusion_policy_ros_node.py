#!/usr/bin/env python
import time

import argparse
from pathlib import Path
import pybullet as p
import copy
import numpy as np
import rospy
import torch
from enum import Enum

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from std_msgs.msg import Header

from scipy.spatial.transform import Rotation as R

# Import your policy class
from bimanual_imitation.algorithms.imitate_diffusion_state import ImitateDiffusionState
from bimanual_imitation.algorithms.configs import DiffusionStateParamConfig


class StatsType(Enum):
    NO_NORMALIZE = 0
    NORMALIZE = 1

def pose_msg_to_T(msg: Pose) -> np.ndarray:
    """
    Convert a geometry_msgs/Pose into a 4x4 homogeneous transform.
    """
    T = np.eye(4)
    T[:3, 3] = np.array([msg.position.x, msg.position.y, msg.position.z])
    rot_mat = R.from_quat([
        msg.orientation.x,
        msg.orientation.y,
        msg.orientation.z,
        msg.orientation.w
    ]).as_matrix()
    T[:3, :3] = rot_mat
    return T

def get_link_name_to_idx(robot: int) -> dict:
    link_name_to_idx = {}
    for i in range(p.getNumJoints(robot)):
        joint_info = p.getJointInfo(robot, i)
        link_name_to_idx[joint_info[12].decode("utf-8")] = i
    return link_name_to_idx

def set_robot_state(robot, q: np.ndarray) -> None:
    num_total_joints = p.getNumJoints(robot)
    actuatable_joint_idxs = [ i for i in range(num_total_joints) if p.getJointInfo(robot, i)[2] != p.JOINT_FIXED ]
    num_actuatable_joints = len(actuatable_joint_idxs)

    assert len(q.shape) == 1, f"q.shape: {q.shape}"
    assert (
        q.shape[0] <= num_actuatable_joints
    ), f"q.shape: {q.shape}, num_actuatable_joints: {num_actuatable_joints}"

    for i, joint_idx in enumerate(actuatable_joint_idxs):
        # q may not contain all the actuatable joints, so we assume that the joints not in q are all 0
        if i < len(q):
            p.resetJointState(robot, joint_idx, q[i])
        else:
            p.resetJointState(robot, joint_idx, 0)


def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats["min"]) / (stats["max"] - stats["min"] + 1e-8)
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1

    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats["max"] - stats["min"] + 1e-8) + stats["min"]
    return data

class PolicyInferenceNode:
    def __init__(self, task_name, absolute):
        """
        - Initializes ROS node.
        - Loads the diffusion policy.
        - Subscribes/publishes to relevant topics.
        - Stores the latest messages in buffers.
        - Defines joint limits (first 7 are KUKA, last 16 are Allegro).
        """
        rospy.init_node("policy_inference_node")

        # ------------------------------
        # Set up inference rate
        # ------------------------------
        self.rate_hz = 1  # HARDCODED
        self.rate = rospy.Rate(self.rate_hz)

        # ------------------------------
        # Joint limits: first 7 are for iiwa, last 16 for allegro
        # ------------------------------
        self.joint_lower_limits = np.array([
            -2.96705972839, -2.09439510239, -2.96705972839,
            -2.09439510239, -2.96705972839, -2.09439510239,
            -3.05432619099,
            -0.47, -0.196, -0.174, -0.227,
            -0.47, -0.196, -0.174, -0.227,
            -0.47, -0.196, -0.174, -0.227,
             0.263, -0.105, -0.189, -0.162
        ])
        self.joint_upper_limits = np.array([
             2.96705972839,  2.09439510239,  2.96705972839,
             2.09439510239,  2.96705972839,  2.09439510239,
             3.05432619099,
             0.47, 1.61, 1.709, 1.618,
             0.47, 1.61, 1.709, 1.618,
             0.47, 1.61, 1.709, 1.618,
             1.396, 1.163, 1.644, 1.719
        ])
        assert len(self.joint_lower_limits) == 23, "Expected 23 total joints (7 + 16)."
        assert len(self.joint_upper_limits) == 23, "Expected 23 total joints (7 + 16)."

        # ------------------------------
        # Load the pybullet
        # ------------------------------
        p.connect(p.DIRECT)
        ROBOT_URDF = "/juno/u/oliviayl/repos/cross_embodiment/interactive_robot_visualizer/curobo/src/curobo/content/assets/robot/iiwa_allegro_description/kuka_allegro.urdf"
        self.robot_id = p.loadURDF(
                    ROBOT_URDF,
                    basePosition=[0, 0, 0],
                    baseOrientation=[0, 0, 0, 1],
                    useFixedBase=True,
                    flags=p.URDF_USE_INERTIA_FROM_FILE
                )
        self.robot_link_name_to_id = get_link_name_to_idx(self.robot_id)

        # ------------------------------
        # Load the diffusion policy
        # ------------------------------
        self.task_name = task_name
        self.absolute = absolute
        if self.absolute: self.task_name = self.task_name + "_absolute"
        model_path = f"/juno/u/oliviayl/repos/cross_embodiment/state-imitation/test_results/diffusion_state/{self.task_name}/checkpoints/iteration_050.ckpt"

        # Config parameters that match your training configuration
        self.PRED_HORIZON = 8
        self.OBS_HORIZON = 4
        self.ACTION_HORIZON = 4

        # Initialize policy
        config = DiffusionStateParamConfig(
            pred_horizon=self.PRED_HORIZON,
            obs_horizon=self.OBS_HORIZON,
            action_horizon=self.ACTION_HORIZON,
            batch_size=128,
            num_diffusion_iters=50,
            opt_learning_rate=1e-4,
            opt_weight_decay=1e-6,
            lr_warmup_steps=500,
            lr_scheduler="constant"
        )

        # Create policy and initialize
        self.policy = ImitateDiffusionState(task_name=self.task_name, absolute=True)
        self.policy.run = lambda: None  # Override run method
        self.policy.env_name = "quad_insert_a0o0"
        self.policy.init_params(config)

        # Load checkpoint
        checkpoint_state = torch.load(model_path, map_location=self.policy.device)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint_state:
            self.policy._noise_pred_net.load_state_dict(checkpoint_state['model_state_dict'])
        elif isinstance(checkpoint_state, dict) and not 'model_state_dict' in checkpoint_state:
            self.policy._noise_pred_net.load_state_dict(checkpoint_state)
        else:
            self.policy._noise_pred_net.load_state_dict(checkpoint_state['model_state'])

        # Set to evaluation mode
        self.policy._noise_pred_net.eval()
        self.policy._ema_noise_pred_net = self.policy._noise_pred_net
        rospy.loginfo(f"Loaded policy from {model_path}")

        # Load obs/action normalization stats
        stats = np.load(Path(model_path).parent.parent / "stats.npz", allow_pickle=True)
        self.obs_norm_stats, self.act_norm_stats = stats['obs'].item(), stats['action'].item()

        # ------------------------------
        # Publishers
        # ------------------------------
        self.iiwa_cmd_pub = rospy.Publisher("/iiwa/joint_cmd", JointState, queue_size=10)
        self.allegro_cmd_pub = rospy.Publisher("/allegroHand_0/joint_cmd", JointState, queue_size=10)

        # ------------------------------
        # Subscribers (storing messages in buffers)
        # ------------------------------
        self.iiwa_joint_state_msg = None
        self.allegro_joint_state_msg = None
        self.object_pose_msg = None

        self.iiwa_sub = rospy.Subscriber("/iiwa/joint_states", JointState, self.iiwa_joint_state_callback)
        self.allegro_sub = rospy.Subscriber("/allegroHand_0/joint_states", JointState, self.allegro_joint_state_callback)
        self.object_pose_sub = rospy.Subscriber("/object_pose", Pose, self.object_pose_callback)


        self.obs_buffer = []

        rospy.loginfo("PolicyInferenceNode initialized.")

    # ------------------------------
    # ROS Callbacks: store latest messages
    # ------------------------------
    def iiwa_joint_state_callback(self, msg: JointState):
        self.iiwa_joint_state_msg = msg

    def allegro_joint_state_callback(self, msg: JointState):
        self.allegro_joint_state_msg = msg

    def object_pose_callback(self, msg: Pose):
        self.object_pose_msg = msg
    
    def update_observation_buffer(self) -> np.ndarray:
        """
        Updates the observation buffer with a new observation.
        If buffer isn't fully populated yet, repeats the first observation to pad it.
        
        Returns:
            A numpy array containing the observation history (shape: [obs_history_length, obs_dim])
        """
        assert self.iiwa_joint_state_msg is not None
        assert self.allegro_joint_state_msg is not None
        assert self.object_pose_msg is not None

        # Copy to avoid race conditions
        iiwa_joint_state_msg = copy.copy(self.iiwa_joint_state_msg)
        allegro_joint_state_msg = copy.copy(self.allegro_joint_state_msg)
        object_pose_msg = copy.copy(self.object_pose_msg)

        # ------------------------------
        # Build observation
        #  - 7 DoF for iiwa
        #  - 16 DoF for allegro
        #  - 16 for object pose (flatten 4x4)
        # => total 39 (example)
        # ------------------------------
        current_iiwa_q = np.array(iiwa_joint_state_msg.position)
        current_allegro_q = np.array(allegro_joint_state_msg.position)

        assert current_iiwa_q.shape == (7,), f"Expected 7 joints for iiwa, got {current_iiwa_q.shape}"
        assert current_allegro_q.shape == (16,), f"Expected 16 joints for allegro, got {current_allegro_q.shape}"
        current_q = np.concatenate([current_iiwa_q, current_allegro_q], axis=0)

        # Convert object_pose to 4x4
        T_C_O = pose_msg_to_T(object_pose_msg)

        # Hard-coded transform from camera to robot frame (example):
        T_R_C = np.eye(4)
        T_R_C[:3, :3] = np.array([
            [0.9543812680846684,  0.08746057618774912, -0.2854943830305726],
            [0.29537672607257903, -0.41644924520026877,  0.8598387150313551],
            [-0.043691930876822334, -0.904942359371598, -0.42328517738189414]
        ])
        T_R_C[:3, 3] = np.array([0.5947949577333569, -0.9635715691360609, 0.6851893282998003])

        # Transform object pose from camera frame to robot frame
        T_R_O = T_R_C @ T_C_O
        assert T_R_O.shape == (4, 4), f"T_R_O shape mismatch: {T_R_O.shape}"

        # Flatten the 4x4
        flat_object_pose = T_R_O.reshape(16)

        set_robot_state(self.robot_id, current_q)
        robot_palm_com, robot_palm_quat, *_ = p.getLinkState(
            self.robot_id,
            self.robot_link_name_to_id["palm_link"],
            computeForwardKinematics=1,
        )
        # Combine (23 + 16 + 7 = 46) for obs
        curr_obs = np.concatenate((current_q, robot_palm_com, robot_palm_quat, flat_object_pose), axis=0)
        assert curr_obs.shape[0] == 46, f"curr_obs.shape: {curr_obs.shape}"

        curr_obs_norm = normalize_data(curr_obs, self.obs_norm_stats)

        num_weird_obs = np.sum(np.absolute(curr_obs_norm) > 1)
        print(f"num_weird_obs = {num_weird_obs}")
        curr_obs_norm_clipped = np.clip(curr_obs_norm, -1, 1)

        # If buffer is empty and we get our first observation, repeat it to fill the buffer
        if len(self.obs_buffer) == 0:
            for _ in range(self.OBS_HORIZON):
                self.obs_buffer.append(curr_obs_norm_clipped.copy())
        else:
            # Add the new observation to the buffer
            self.obs_buffer.append(curr_obs_norm_clipped.copy())
            # Remove the oldest observation if buffer exceeds the desired length
            if len(self.obs_buffer) > self.OBS_HORIZON:
                self.obs_buffer.pop(0)
        return current_q

    # ------------------------------
    # Main loop
    # ------------------------------
    def run(self):
        while not rospy.is_shutdown():
            start_time = rospy.Time.now()

            # Check that we have necessary messages
            if (self.iiwa_joint_state_msg is None or
                self.allegro_joint_state_msg is None or
                self.object_pose_msg is None):
                rospy.logwarn_throttle(5.0, f"Waiting for all required messages... self.iiwa_joint_state_msg: {self.iiwa_joint_state_msg} self.allegro_joint_state_msg: {self.allegro_joint_state_msg} self.object_pose_msg: {self.object_pose_msg}")
                self.rate.sleep()
                continue

            current_q = self.update_observation_buffer()

            obs_np = np.stack(self.obs_buffer, axis=0)  # Shape: [obs_history_length, obs_dim]
            obs_torch = torch.from_numpy(obs_np).float().unsqueeze(0).to(self.policy.device)  # (1, 46)
            assert obs_torch.shape == (1, self.OBS_HORIZON, 46), f"Expected (1, {self.OBS_HORIZON}, 46), got {obs_torch.shape}"
            obs_torch = obs_torch.reshape(1, self.OBS_HORIZON * 46)

            # ------------------------------
            # Policy inference
            # ------------------------------
            with torch.no_grad():
                raw_action = self.policy.bc_policy_fn(obs_torch)
            assert raw_action.shape == (1, self.PRED_HORIZON, 23), f"Expected (1, {self.PRED_HORIZON}, 23)-dim action, got {raw_action.shape}"
            raw_action_np = raw_action[0, :, :].cpu().numpy()
            action_np = unnormalize_data(raw_action_np, self.act_norm_stats)
            assert action_np.shape == (self.PRED_HORIZON, 23), f"Expected ({self.PRED_HORIZON}, 23)-dim action, got {action_np.shape}"

            # Clip to joint limits
            if self.absolute:
                new_q = action_np
            else:
                new_q = action_np + current_q[None]

            new_q = np.clip(new_q, self.joint_lower_limits[None], self.joint_upper_limits[None])
            print(f"obs_torch = {obs_torch}")
            print(f"action_np = {action_np}")
            print(f"new_q = {new_q}")

            # Split again
            new_iiwa_q = new_q[:, :7]
            new_allegro_q = new_q[:, 7:]

            # ------------------------------
            # Publish commands
            # ------------------------------
            for i in range(self.PRED_HORIZON):
                current_time = rospy.Time.now()

                # KUKA
                iiwa_cmd_msg = JointState()
                iiwa_cmd_msg.header = Header(stamp=current_time)
                iiwa_cmd_msg.name = [f"iiwa_joint_{i+1}" for i in range(7)]
                iiwa_cmd_msg.position = new_iiwa_q[i].tolist()
                self.iiwa_cmd_pub.publish(iiwa_cmd_msg)

                # Allegro
                allegro_cmd_msg = JointState()
                allegro_cmd_msg.header = Header(stamp=current_time)
                allegro_cmd_msg.name = [f"allegro_joint_{i}" for i in range(16)]
                allegro_cmd_msg.position = new_allegro_q[i].tolist()
                self.allegro_cmd_pub.publish(allegro_cmd_msg)

                self.update_observation_buffer()
                time.sleep(1/self.rate_hz)

            # ------------------------------
            # Sleep to maintain rate
            # ------------------------------
            before_sleep_time = rospy.Time.now()
            self.rate.sleep()
            after_sleep_time = rospy.Time.now()

            total_loop_time = (after_sleep_time - start_time).to_sec()
            rospy.loginfo_throttle(
                2.0,
                f"[{rospy.get_name()}] Loop took {total_loop_time:.4f}s "
                f"(~{1.0/total_loop_time:.2f} Hz actual)."
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="snackbox_push")
    parser.add_argument("--absolute", type=bool, default=False)
    args = parser.parse_args()

    node = PolicyInferenceNode(args.task_name, args.absolute)
    node.run()

if __name__ == '__main__':
    main()
