import argparse
import zmq
import json
import os
import numpy as np
import io
import copy
import cv2
import torch
import time
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple
import logging
logging.basicConfig(level=logging.INFO)

from pprint import pformat
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.utils.robot_utils import precise_sleep

from lerobot.utils.sim2real_constant import (
    HOME_MOVE_HZ,
    HOME_SPEED,
    HOME_TOL,
    INPUT_OBSERVATION_SOCKET,
    OUTPUT_ACTION_SOCKET,
    EMPTY_SIGNAL_SOCKET,
    HOME_JOINT_STATE,
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


def remap_action_between_calibrations(
    action: Dict[str, float],
    source_calib: str = SO101_FOLLOWER_OLD_CALIB,
    target_calib: str = SO101_FOLLOWER_NEW_CALIB,
    clip: bool = True,
) -> Dict[str, float]:
    """Map normalized joint targets from one calibration range to another.

    The base model appears to output normalized positions for the original
    SO101 range. This converts each value to the equivalent simulated radian
    position in that source range, then converts that radian position into the
    target normalized range.
    """

    remapped_action = copy.deepcopy(action)
    for joint_name in JOINT_ORDER:
        source_rad = pos2rad(
            pos=action[joint_name],
            joint_name=joint_name,
            calibration=source_calib,
        )
        target_pos = rad2pos(
            rad=source_rad,
            joint_name=joint_name,
            calibration=target_calib,
        )
        if clip:
            target_pos = float(np.clip(target_pos, -100.0, 100.0))
        remapped_action[joint_name] = target_pos

    return remapped_action