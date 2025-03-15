import os
import glob
import argparse
import random
import numpy as np
from pathlib import Path

import pybullet as p

from irl_data.trajectory import Trajectory, TrajBatch
from irl_data.proto_logger import export_trajs
from irl_data.constants import IRL_DATA_BASE_DIR

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

def load_npzs_as_trajs(npz_files, absolute=False, obs_dim=46, action_dim=23):
    """
    Load data from npz file, ensuring correct tensor shapes
    
    Expected npz file should have keys 'obs' and 'action'
    'obs' = [joint_angles, ee_pos, ee_quat, obj_pose]. joint_angles.shape=(23,), ee_pos.shape=(3,), ee_quat.shape=(4,), obj_pose=(16,)
    """    
    p.connect(p.DIRECT)
    ROBOT_URDF = "/juno/u/oliviayl/repos/cross_embodiment/interactive_robot_visualizer/curobo/src/curobo/content/assets/robot/iiwa_allegro_description/kuka_allegro.urdf"
    robot_id = p.loadURDF(
                ROBOT_URDF,
                basePosition=[0, 0, 0],
                baseOrientation=[0, 0, 0, 1],
                useFixedBase=True,
                flags=p.URDF_USE_INERTIA_FROM_FILE
            )
    robot_link_name_to_id = get_link_name_to_idx(robot_id)

    all_trajectories = []
    for file_path in npz_files:
        data = np.load(file_path)
        tsteps = data['ts'] # (T,)
        T = tsteps.shape[0]
        joint_angles = data['qs'] # (T, 23)
        obj_poses = data['object_poses'].reshape(T, 16) # (T, 16)

        obs_np, actions_np = [], []
        for i in range(0, T-1):
            set_robot_state(robot_id, joint_angles[i])
            robot_palm_com, robot_palm_quat, *_ = p.getLinkState(
                robot_id,
                robot_link_name_to_id["palm_link"],
                computeForwardKinematics=1,
            )
            curr_obs = np.concatenate((joint_angles[i], robot_palm_com, robot_palm_quat, obj_poses[i]), axis=0)
            assert curr_obs.shape[0] == obs_dim, f"curr_obs.shape: {curr_obs.shape}"
            obs_np.append(curr_obs)

            D = 15 # We want one waypoint every 0.5 seconds
            waypoint = min(((i // D) + 1) * D, T-1)
            curr_action = joint_angles[waypoint] - joint_angles[i]
            if absolute:
                curr_action = joint_angles[waypoint]
            assert curr_action.shape[0] == action_dim, f"curr_action.shape: {curr_action.shape}"
            actions_np.append(np.array(curr_action))
    
        obs_np, actions_np, rew_np = np.array(obs_np), np.array(actions_np), np.zeros(len(obs_np))
        
        traj = Trajectory(
            obs_T_Do=obs_np,
            obsfeat_T_Df=np.zeros(obs_np.shape),  # Placeholder for obsfeat
            adist_T_Pa=np.zeros((obs_np.shape[0], 2)),  # Placeholder for adist
            a_T_Da=actions_np,
            r_T=rew_np
        )
        all_trajectories.append(traj)
        print(f"Added trajectory with {len(obs_np)} timesteps, obs_shape={obs_np.shape}, action_shape={actions_np.shape}")
    
    p.disconnect()
    return all_trajectories


def export_trajectories(trajectories, output_file):
    """
    Export a list of Trajectory objects to a proto file
    
    Parameters:
    trajectories (list): List of Trajectory objects
    output_file (str): Path to the output proto file
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a TrajBatch from the list of trajectories
    traj_batch = TrajBatch.FromTrajs(trajectories)
    
    # Export to proto format
    print(f"Exporting to proto file: {output_file}")
    export_trajs(traj_batch, output_file)
    print(f"Exported {len(trajectories)} trajectories to {output_file}")

def setup_expert_trajectories_dir():
    """Ensure the expert_trajectories directory exists in IRL_DATA_BASE_DIR"""
    expert_dir = IRL_DATA_BASE_DIR / "expert_trajectories"
    if not os.path.exists(expert_dir):
        os.makedirs(expert_dir)
    return expert_dir

def main():
    parser = argparse.ArgumentParser(description="Convert NPZ files to proto format for imitation learning")
    parser.add_argument("--task_name", type=str, required=True, default="snackbox_push", help="Path(s) to the NPZ file(s)")
    parser.add_argument("--absolute", type=bool, default=False, help="Compute absolute instead of delta actions")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the proto file(s). Defaults to IRL_DATA_BASE_DIR/expert_trajectories/")
    parser.add_argument("--validation_split", type=float, default=0.0, help="Fraction of trajectories to use for validation (0.0 to 1.0)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for shuffling")
    
    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)

    # List npz files
    npz_files = glob.glob(f"/juno/u/tylerlum/github_repos/interactive_robot_visualizer/2025-03-11_BC_30demos/{args.task_name}_*.npz")
    if not npz_files:
        raise ValueError(f"No NPZ files found for task {args.task_name}")
    print(f"Found {len(npz_files)} NPZ files for task {args.task_name}")
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = setup_expert_trajectories_dir()
    
    # Determine output file paths
    train_proto_file = output_dir / f"{args.task_name}.proto"
    if args.absolute:
        train_proto_file = output_dir / f"{args.task_name}_absolute.proto"
    val_proto_file = None
    
    if args.validation_split > 0:
        val_proto_file = output_dir.parent / "expert_validation_trajectories" / f"{args.env_name}.proto"
        if args.absolute:
            val_proto_file = output_dir.parent / "expert_validation_trajectories" / f"{args.env_name}_absolute.proto"
        # Create validation directory if needed
        if not os.path.exists(os.path.dirname(val_proto_file)):
            os.makedirs(os.path.dirname(val_proto_file))
    
    # Process all NPZ files
    all_trajectories = load_npzs_as_trajs(npz_files, args.absolute)
    print(f"Loaded {len(all_trajectories)} trajectories")
    
    # If we have multiple NPZ files or need a validation split, we need to recreate the proto file
    if (args.validation_split > 0 and len(all_trajectories) > 1):
        # Shuffle the list of trajectories
        random.shuffle(all_trajectories)
        
        # Split into train and validation
        split_idx = max(1, int(len(all_trajectories) * (1 - args.validation_split)))
        train_trajectories = all_trajectories[:split_idx]
        val_trajectories = all_trajectories[split_idx:]
        
        # Export train and validation sets
        export_trajectories(train_trajectories, train_proto_file)
        export_trajectories(val_trajectories, val_proto_file)
    else:
        # Export all trajectories to one file
        export_trajectories(all_trajectories, train_proto_file)

if __name__ == "__main__":
    main()