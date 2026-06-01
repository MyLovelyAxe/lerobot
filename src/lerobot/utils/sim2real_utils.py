import numpy as np
import copy
import time
from typing import Dict, List
import logging
logging.basicConfig(level=logging.INFO)

from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.utils.robot_utils import precise_sleep

from lerobot.utils.sim2real_constant import (
    HOME_MOVE_HZ,
    HOME_SPEED,
    HOME_TOL,
    JOINT_ORDER,
    SIMULATION_RANGE,
    SO101_FOLLOWER_NEW_CALIB,
    SO101_FOLLOWER_OLD_CALIB,
)

def move_robot_to_target_pose(
    robot: SO101Follower, 
    target_pose: Dict[str, float],
    reverse_order: bool = False,
):
    """TODO: docstring"""
    action = {
        k: v
        for k, v in robot.get_observation().items()
        if k.endswith(".pos")
    }

    # TODO: set a lower acceleration, this code might not be correct
    # for motor in robot.bus.motors:
    #     robot.bus.write("Acceleration", motor, 254)

    joint_order = JOINT_ORDER.copy()
    if reverse_order:
        joint_order.reverse()

    dt = 1.0 / HOME_MOVE_HZ
    max_step = HOME_SPEED * dt

    for joint_name in joint_order:
        logging.info(f"Moving {joint_name} to home")

        while True:
            loop_start = time.perf_counter()

            diff = target_pose[joint_name] - action[joint_name]
            if abs(diff) <= HOME_TOL:
                action[joint_name] = target_pose[joint_name]
                robot.send_action(action)
                break

            step = max(-max_step, min(max_step, diff))
            action[joint_name] += step
            robot.send_action(action)

            precise_sleep(dt - (time.perf_counter() - loop_start))

        logging.info(f"joint {joint_name} returns to home")


def log_joint_state(
    joint_state: Dict[str, float],
    logging_label: str = "Current joint state",
):
    """Log the joint state of robot."""
    joint_state_list = list(f"{value:4f}" for key, value in joint_state.items() if key.endswith(".pos"))
    logging.info(f"{logging_label}: {', '.join(joint_state_list)}")


def rad2pos(
    rad: float, 
    joint_name: str, 
    calibration: str = SO101_FOLLOWER_NEW_CALIB,
):
    """Convert radians into motor pos.
    
    Isaac Sim joint state has real radian values, while SO100 robot hardware
    use -100 to 100 as "pos". Original script for SO100 follower, i.e. using_smolvla_example.py,
    use the default value of RobotConfig.use_degrees = False, so it use MotorNormMode.RANGE_M100_100,
    check norm_mode_body of class so100_foller.SO100Follower
    """

    sim_min = SIMULATION_RANGE[calibration][joint_name]['sim_min']
    sim_max = SIMULATION_RANGE[calibration][joint_name]['sim_max']

    if joint_name == 'gripper.pos':
        # norm = ((bounded_val - min_) / (max_ - min_)) * 100
        pos = ((rad - sim_min) / (sim_max - sim_min)) * 100
    else:
        # norm = (((bounded_val - min_) / (max_ - min_)) * 200) - 100
        pos = ((rad - sim_min) / (sim_max - sim_min)) * 200 - 100

    # # TODO: this gives a larger gripper joint value, even though it should use MotorNormMode.RANGE_0_100
    # pos = ((rad - sim_min) / (sim_max - sim_min)) * 200 - 100
    return pos


def pos2rad(
    pos: float, 
    joint_name: str, 
    calibration: str = SO101_FOLLOWER_NEW_CALIB,
):
    """Convert the noramlized motor pos into real radian."""

    sim_min = SIMULATION_RANGE[calibration][joint_name]['sim_min']
    sim_max = SIMULATION_RANGE[calibration][joint_name]['sim_max']

    if joint_name == 'gripper.pos':
        # unnormalized_values[id_] = int((bounded_val / 100) * (max_ - min_) + min_)
        rad = (pos / 100) * (sim_max - sim_min) + sim_min
    else:
        # unnormalized_values[id_] = int(((bounded_val + 100) / 200) * (max_ - min_) + min_)
        rad = (pos + 100) / 200 * (sim_max - sim_min) + sim_min
    
    # # TODO: use correct version for gripper later
    # rad = (pos + 100) / 200 * (sim_max - sim_min) + sim_min
    return rad


def joint_state_pos2rad(
    pos_joint_state: Dict[str, float],
    calibration: str = SO101_FOLLOWER_NEW_CALIB,
) -> np.ndarray:
    """ Convert calibrated normalized joint state into radian for all joints.
    
    :param pos_joint_state: calibrated normalized joint state
    :param calibration: calibration method
    """
    rad_joint_state = np.array([
        pos2rad(pos=pos_joint_state[joint], joint_name=joint, calibration=calibration) for joint in JOINT_ORDER
    ],dtype=np.float32)

    return rad_joint_state

def joint_state_rad2pos(
    rad_joint_state: np.ndarray,
    calibration: str = SO101_FOLLOWER_NEW_CALIB,
) -> Dict[str, float]:
    """ Convert calibrated normalized joint state into radian for all joints.
    
    :param rad_joint_state: joint state values in unit of radian
    :param calibration: calibration method
    """
    rad_joint_state = {
        joint: rad2pos(rad=rad_joint_state[idx], joint_name=joint, calibration=calibration) for idx, joint in enumerate(JOINT_ORDER)
    }

    return rad_joint_state


def generate_trajectory(
    q_start: np.ndarray, 
    q_target: np.ndarray, 
    T: float = 3.0, 
    dt: float = 0.02,
) -> List[Dict[str, float]]:
    """Generate a simple and smooth trajectory between 2 joint states.
    
    :param q_start: starting joint state
    :param q_target: target joint state
    :param T: the complete duration to execute the trajectory
    :param dt: temporal interval in the trajectory
    """
    
    q_start = np.array(q_start, dtype=float)
    q_target = np.array(q_target, dtype=float)

    trajectory: List[Dict[str, float]] = list()
    times = np.arange(0.0, T + dt, dt)

    # compute one smooth step for one timestamp
    smooth_step = lambda tau: 3 * tau**2 - 2 * tau**3

    for t in times:
        s = smooth_step(tau=t/T)
        q = q_start + s * (q_target - q_start)
        step = {
            "timestamp": t,
            "joint_state": q.tolist(),
        }
        trajectory.append(step)

    return trajectory

def generate_robot_actions_trajectory(
    start_state: Dict[str, float],
    target_state: Dict[str, float],
    T: float = 3.0, 
    dt: float = 0.02,
) -> List[Dict[str, float]]:
    """Generate trajectory in form of robot actions."""

    q_start = np.array([start_state[name] for name in JOINT_ORDER], dtype=np.float32)
    q_target = np.array([target_state[name] for name in JOINT_ORDER], dtype=np.float32)
    trajectory = generate_trajectory(
        q_start=q_start, 
        q_target=q_target, 
        T=T, 
        dt=dt,
    )
    action_trajectory: List[Dict[str, float]] = list()
    for step in trajectory:
        action_trajectory.append(
            dict(zip(JOINT_ORDER, step["joint_state"])),
        )
    return action_trajectory