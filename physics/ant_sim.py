import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import math

xml_path = 'simple_quadraped.xml'  # Ensure this path points to your MuJoCo model

def init_simulation(xml_file_path):
    """
    Initialize the simulation environment.
    """
    model = mj.MjModel.from_xml_path(xml_file_path)
    data = mj.MjData(model)
    return model, data

def simple_limb_movement_controller(model, data, amplitude=0.2, frequency=1.0):
    """
    A simple controller to move the limbs of the quadruped.
    """
    # Apply the desired angle to the joints
    for name in ["front_right_shoulder", "front_left_shoulder", "rear_right_shoulder", "rear_left_shoulder",
                 "front_right_elbow", "front_left_elbow", "rear_right_elbow", "rear_left_elbow"]:
        joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        control_input = np.random.uniform(-0.2, 0.2)
        data.ctrl[joint_id-1] = control_input

def observe_state(model, data):
    # Joint angles
    joint_angles = data.qpos[:model.nq].copy()

    print('JOINT ANGLES')
    print(joint_angles)

    limb_joint_angles = []

    # Extract joint angles for each limb
    for limb in ["front_right", "front_left", "rear_right", "rear_left"]:
        shoulder_joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, limb + "_shoulder")
        elbow_joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, limb + "_elbow")
        limb_joint_angles.append(data.qpos[shoulder_joint_id-1])
        limb_joint_angles.append(data.qpos[elbow_joint_id-1])

    print("limb joint angles", limb_joint_angles)

    # Body orientation
    if model.nq > 3:  # Assuming the orientation is represented by a quaternion
        root_quat = data.qpos[3:7]  # Assuming the root's orientation is stored here
        root_orientation_euler = quat_to_euler(root_quat)
    else:
        root_orientation_euler = np.zeros(3)
    
    print('ORIENTATION')
    print(root_orientation_euler)

    # body position
    body_position = data.qpos[:3].copy()

    print('BODY POSITION')
    print(body_position)
    
    return np.concatenate([limb_joint_angles, root_orientation_euler, body_position])

def quat_to_euler(quat):
    """
    Convert a quaternion into Euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    w, x, y, z = quat
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, a_min=-1.0, a_max=1.0)
    pitch_y = np.arcsin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z

def run_simulation(model, data):
    # Initialize GLFW
    glfw.init()
    window = glfw.create_window(1200, 900, "Quadruped Simulation", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # Initialize visualization structures
    maxgeom = 1000
    scn = mj.MjvScene(model, maxgeom)
    ctx = mj.MjrContext(model, 0)

    # Initialize a default camera
    # Initialize a camera with specified settings
    cam = mj.MjvCamera()
    cam.azimuth = 90
    cam.elevation = -30
    cam.distance = 5

    count = 0
    while count < 4:
        count += 1
    # while not glfw.window_should_close(window):
        # Apply simple limb movement
        simple_limb_movement_controller(model, data)
        print(observe_state(model, data))
        # Step simulation
        mj.mj_step(model, data)

        # Update scene for rendering
        mj.mjv_updateScene(model, data, mj.MjvOption(), None, cam, -1, scn)

        # Render
        viewport_width, viewport_height = glfw.get_framebuffer_size(window)
        viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
        mj.mjr_render(viewport, scn, ctx)

        # Swap buffers
        glfw.swap_buffers(window)
        glfw.poll_events()

    # glfw.terminate()

if __name__ == "__main__":
    model, data = init_simulation(xml_path)
    run_simulation(model, data)