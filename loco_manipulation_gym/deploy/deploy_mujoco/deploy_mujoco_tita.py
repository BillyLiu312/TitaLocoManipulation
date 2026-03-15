import time
import mujoco.viewer
import mujoco
import numpy as np
from loco_manipulation_gym import LOCO_MANI_GYM_ROOT_DIR
import torch
import yaml
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

def cart_to_sphere(cart):
    """Convert Cartesian coordinates to spherical coordinates.
    
    Args:
        cart (array-like): Cartesian coordinates [x, y, z].
    
    Returns:
        array-like: Spherical coordinates [r, phi, theta], where:
            - r: radius (distance from origin)
            - phi: azimuthal angle (atan2(z, x))
            - theta: polar angle (asin(y / r))
    """
    
    sphere = np.zeros_like(cart)
    sphere[0] = np.linalg.norm(cart)
    sphere[1] = np.arctan2(cart[2], cart[0])
    sphere[2] = np.arcsin(cart[1] / sphere[0]) if abs(sphere[0]) > 1e-8 else 0
    return sphere

def sphere_to_cart(sphere):
    """Convert spherical coordinates to Cartesian coordinates.
    
    Args:
        sphere (array-like): Spherical coordinates [r, phi, theta].
    
    Returns:
        array-like: Cartesian coordinates [x, y, z].
    """
    cart = np.zeros_like(sphere)
    cart[0] = sphere[0] * np.cos(sphere[2]) * np.cos(sphere[1])
    cart[1] = sphere[0] * np.sin(sphere[2])
    cart[2] = sphere[0] * np.cos(sphere[2]) * np.sin(sphere[1])
    return cart

def normalize_wheel_pos(wheel_pos):
    """Normalize wheel position to the range [-π, π]."""
    return (wheel_pos + np.pi) % (2 * np.pi) - 1 * np.pi

def print_joint_order(m):
    """Print the joint order in the model."""
    print("\n=== Joint Order Information ===")
    for i in range(m.njnt):
        jnt_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f"Joint {i}: {jnt_name}")

def get_gravity_orientation(quat_wxyz):
    gravity_vec = [0, 0, -1]
    return quat_rotate_inv(quat_wxyz, gravity_vec)

def quat_rotate_pos(quat_wxyz, vec):
    qw = quat_wxyz[0]
    qx = quat_wxyz[1]
    qy = quat_wxyz[2]
    qz = quat_wxyz[3]
    r = R.from_quat([qx, qy, qz, qw])
    return r.apply(vec)

def quat_rotate_inv(quat_wxyz, vec):
    quat_conj_wxyz = np.array([quat_wxyz[0], -quat_wxyz[1], -quat_wxyz[2], -quat_wxyz[3]])
    return quat_rotate_pos(quat_conj_wxyz, vec)

def quat_to_euler(quat_wxyz):
    """Convert quaternion to Euler angles (roll, pitch, yaw).
    
    Args:
        quat_wxyz (array-like): Quaternion in [w, x, y, z] format.
    
    Returns:
        array-like: Euler angles [roll, pitch, yaw] in degrees.
    """
    qw = quat_wxyz[0]
    qx = quat_wxyz[1]
    qy = quat_wxyz[2]
    qz = quat_wxyz[3]
    r = R.from_quat([qx, qy, qz, qw])
    return r.as_euler('xyz', degrees=True)

def euler_to_quat_wxyz(euler, degrees=True):
    """
    Convert Euler angles to quaternion.
    
    Args:
        euler (array-like): Euler angles [roll, pitch, yaw] (rotations around x, y, z axes).
        degrees (bool): Whether input angles are in degrees. Default is True.
    
    Returns:
        array-like: Quaternion in [w, x, y, z] format.
    """
    # Create rotation object
    r = R.from_euler('xyz', euler, degrees=degrees)
    # Get quaternion (scipy returns xyzw format by default)
    quat_xyzw = r.as_quat()
    # Convert to wxyz format
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return quat_wxyz

def wxyz_to_xyzw(wxyz_quat):
    """Convert quaternion from wxyz format to xyzw format."""
    # wxyz: [w, x, y, z] -> xyzw: [x, y, z, w]
    return np.array([wxyz_quat[1], wxyz_quat[2], wxyz_quat[3], wxyz_quat[0]])

def xyzw_to_wxyz(xyzw_quat):
    """Convert quaternion from xyzw format to wxyz format."""
    # xyzw: [x, y, z, w] -> wxyz: [w, x, y, z]
    return np.array([xyzw_quat[3], xyzw_quat[0], xyzw_quat[1], xyzw_quat[2]])

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    parser.add_argument("--control_type", type=str, default="P", choices=["P", "V", "T"], help="Control type")
    
    args = parser.parse_args()
    config_file = args.config_file
    
    with open(f"{LOCO_MANI_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LOCO_MANI_GYM_ROOT_DIR}", LOCO_MANI_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LOCO_MANI_GYM_ROOT_DIR}", LOCO_MANI_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)
        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        lin_vel_scale = config["lin_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        action_scale_vel = config["action_scale_vel"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
        gripper_track_scale = config.get("gripper_track_scale", 1.0)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    print(f"=== Configuration Debug Information ===")
    print(f"kps: {kps}, shape: {kps.shape}, length: {len(kps)}")
    print(f"kds: {kds}, shape: {kds.shape}, length: {len(kds)}")
    print(f"num_actions: {num_actions}")
    print(f"default_angles: {default_angles}, length: {len(default_angles)}")
    print(f"Control type: {args.control_type}")

    # Define variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0
    last_dof_vel = np.zeros(len(default_angles), dtype=np.float32)
    
    # Target definition in local coordinate system (relative to base)
    local_start_pos = np.array([0.45, 0.0, 0.05])  # Start position (local coordinate)
    local_end_pos = np.array([0.3, 0.0, 0.2])    # End position (local coordinate)

    # Convert to spherical coordinates
    local_start_sphere = cart_to_sphere(local_start_pos)
    local_end_sphere = cart_to_sphere(local_end_pos)

    local_curr_ee_goal_sphere = local_start_sphere.copy()
    local_curr_ee_goal_cart = sphere_to_cart(local_curr_ee_goal_sphere).copy()
    
    curr_goal_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
    
    # Interpolation parameters
    trajectory_duration = 15.0  # Trajectory period (seconds)
    trajectory_start_time = 10.0
    interpolation_alpha = 0.0
    
    # Add data recording lists
    time_stamps = []
    ee_positions = []
    ee_eulers = []
    target_positions = []
    target_eulers = []
    
    # Define joint indices
    wheel_indices = [2, 5]
    arm_indices = [6, 7, 8, 9, 10, 11]

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # Print joint order
    print_joint_order(m)

    print(f"\n=== Model Debug Information ===")
    print(f"Total joints (m.njnt): {m.njnt}")
    print(f"Position vector length (m.nq): {m.nq}")
    print(f"Velocity vector length (m.nv): {m.nv}")
    print(f"Control vector length (m.nu): {m.nu}")

    gripper_body_name = "gripper_link"
    base_body_name = "base_link"
    
    gripper_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, gripper_body_name)
    base_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, base_body_name)

    # Load policy
    policy = torch.jit.load(policy_path)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            
            # Get current joint states
            dof_pos = d.qpos[7:7+len(default_angles)].copy()
            dof_vel = d.qvel[6:6+len(default_angles)].copy()
            
            # Calculate position error, but set wheel joint errors to 0
            dof_err = dof_pos - default_angles
            dof_err[wheel_indices] = 0
            dof_pos[wheel_indices] = 0
            
            actions_scaled = action * action_scale
            
            # Modify velocity vector, set arm joint velocities to 0
            modify_dof_vel = dof_vel.copy()
            modify_dof_vel[arm_indices] = 0
            
            # Calculate torques based on control type
            if args.control_type == "P":
                torques = kps * (actions_scaled - dof_err) - kds * modify_dof_vel
            elif args.control_type == "V":
                torques = kps * (actions_scaled - modify_dof_vel) - kds * (modify_dof_vel - last_dof_vel) / simulation_dt
            elif args.control_type == "T":
                torques = actions_scaled
            else:
                raise NameError(f"Unknown controller type: {args.control_type}")
            
            if action_scale_vel:
                torques[wheel_indices] = kds[wheel_indices] * (action_scale_vel * action[wheel_indices] - dof_vel[wheel_indices])
            
            # Update last velocity
            last_dof_vel = dof_vel.copy()

            # Apply control torques
            d.ctrl[:len(torques)] = torques

            mujoco.mj_step(m, d)

            counter += 1

            if counter % control_decimation == 0:
                base_quat_wxyz = d.xquat[base_body_id]
                base_pos = d.xpos[base_body_id]  
                base_lin_vel = quat_rotate_inv(base_quat_wxyz, d.qvel[0:3])
                base_ang_vel = quat_rotate_inv(base_quat_wxyz, d.qvel[3:6])
                
                gripper_pos = d.xpos[gripper_body_id]
                gripper_quat_wxyz = d.xquat[gripper_body_id]
                gripper_quat_xyzw = wxyz_to_xyzw(gripper_quat_wxyz)
                curr_goal_quat_xyzw = wxyz_to_xyzw(curr_goal_quat_wxyz)
                
                # Calculate simulation time
                current_time = time.time() - start
                
                interpolation_alpha = (current_time - trajectory_start_time) / trajectory_duration
                interpolation_alpha = np.clip(interpolation_alpha, 0.0, 1.0)
                
                # Spherical interpolation in local coordinate system
                local_curr_ee_goal_sphere = local_start_sphere + interpolation_alpha * (local_end_sphere - local_start_sphere)
                local_curr_ee_goal_cart = sphere_to_cart(local_curr_ee_goal_sphere)

                gripper_pos_bias = gripper_pos + quat_rotate_pos(gripper_quat_wxyz, np.array([0.1, 0.0, 0.0]))
                base_align_z_axis = np.array([base_pos[0], base_pos[1], 0.5])
                base_yaw = quat_to_euler(base_quat_wxyz)[2]  # Extract base yaw angle
                base_yaw_quat_wxyz = euler_to_quat_wxyz(np.array([0, 0, base_yaw]), degrees=True)

                local_gripper_pos = quat_rotate_inv(base_yaw_quat_wxyz, gripper_pos_bias - base_align_z_axis)
                
                abs_ee_quat_xyzw = gripper_quat_xyzw
                
                gravity_orientation = get_gravity_orientation(base_quat_wxyz)

                obs[0:3] = base_lin_vel * lin_vel_scale
                obs[3:6] = base_ang_vel * ang_vel_scale
                obs[6:9] = gravity_orientation
                obs[9:12] = cmd * cmd_scale
                obs[12:24] = dof_err * dof_pos_scale
                obs[24:36] = dof_vel * dof_vel_scale
                obs[36:39] = local_gripper_pos * gripper_track_scale
                obs[39:42] = local_curr_ee_goal_cart * gripper_track_scale
                obs[42:46] = abs_ee_quat_xyzw * gripper_track_scale
                obs[46:50] = curr_goal_quat_xyzw * gripper_track_scale
                obs[50:62] = action
                
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action = policy(obs_tensor).detach().numpy().squeeze()

                # logs
                current_time = time.time() - start
                time_stamps.append(current_time)
                ee_pos = local_gripper_pos.copy()
                ee_positions.append(ee_pos)
                ee_eulers.append(quat_to_euler(gripper_quat_wxyz)) 
                target_pos_with_bias = local_curr_ee_goal_cart.copy()
                target_positions.append(target_pos_with_bias)
                target_eulers.append(quat_to_euler(curr_goal_quat_wxyz))

            viewer.sync()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    # plot
    if len(time_stamps) > 0:
        time_stamps = np.array(time_stamps)
        ee_positions = np.array(ee_positions) + [0, 0, 0.5]
        ee_eulers = np.array(ee_eulers)
        target_positions = np.array(target_positions) + [0, 0, 0.5]
        target_eulers = np.array(target_eulers)
        
        # ee postion
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 1, 1)
        plt.plot(time_stamps, ee_positions[:, 0], 'r-', label='Current X')
        plt.plot(time_stamps, target_positions[:, 0], 'r--', label='Target X')
        plt.plot(time_stamps, ee_positions[:, 1], 'g-', label='Current Y')
        plt.plot(time_stamps, target_positions[:, 1], 'g--', label='Target Y')
        plt.plot(time_stamps, ee_positions[:, 2], 'b-', label='Current Z')
        plt.plot(time_stamps, target_positions[:, 2], 'b--', label='Target Z')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.title('End Effector Position Comparison')
        plt.legend()
        plt.grid(True)
        
        # ee orientation
        plt.subplot(2, 1, 2)
        plt.plot(time_stamps, ee_eulers[:, 0], 'r-', label='Current Roll')
        plt.plot(time_stamps, target_eulers[:, 0], 'r--', label='Target Roll')
        plt.plot(time_stamps, ee_eulers[:, 1], 'g-', label='Current Pitch')
        plt.plot(time_stamps, target_eulers[:, 1], 'g--', label='Target Pitch')
        plt.plot(time_stamps, ee_eulers[:, 2], 'b-', label='Current Yaw')
        plt.plot(time_stamps, target_eulers[:, 2], 'b--', label='Target Yaw')
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (deg)')
        plt.title('End Effector Orientation Comparison (Euler Angles)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()