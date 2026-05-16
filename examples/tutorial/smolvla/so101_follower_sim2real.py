"""
only show current joint state:
$ python so101_follower_sim2real.py

send action to robot to execute:
$ python so101_follower_sim2real.py --send-action

"""

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


HOME_MOVE_HZ = 50
HOME_SPEED = 8.0  # normalized position units per second; tune lower/slower
HOME_TOL = 0.5

def move_home_sequential(
    robot: SO101Follower, 
    calib_json: str = "old_calib_home",
    reverse_order: bool = False,
):
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

            diff = HOME_JOINT_STATE[calib_json][joint_name] - action[joint_name]
            if abs(diff) <= HOME_TOL:
                action[joint_name] = HOME_JOINT_STATE[calib_json][joint_name]
                robot.send_action(action)
                break

            step = max(-max_step, min(max_step, diff))
            action[joint_name] += step
            robot.send_action(action)

            precise_sleep(dt - (time.perf_counter() - loop_start))

        logging.info(f"joint {joint_name} returns to home")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--send-action",
        action="store_true",
        help="Send the returned action from policy to real robot. " \
        "Without this flag, the script only logs the returned actions " \
        "without sending to real robot.",
    )
    return parser.parse_args()


def log_joint_state(
    joint_state: Dict[str, float],
    logging_label: str = "Current joint state",
):
    """Log the joint state of robot."""
    joint_state_list = list(f"{value:4f}" for key, value in joint_state.items() if key.endswith(".pos"))
    logging.info(f"{logging_label}: {', '.join(joint_state_list)}")


DEVICE = torch.device("cuda")
TASK = "pick the red block"
ROBOT_TYPE = "so100_follower"
LOG_SECONDS = 5.0
LOG_HZ = 1
MINIMUM_STEP = 1 # step of calibrated position
ROBOT = "so101_old_calib"
FOLLOWER_PORT_ID = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AAF218449-if00"
MODEL_ID = "lerobot/smolvla_base"
MAX_EPISODES = 5
MAX_STEPS_PER_EPISODE = 20
STORE_IMAGES = False # whether store preprocessed images from socket to /logs
RECORD_ACTIONS = False # store the actions into json for later reply
IMAGES_STORE_PATH = Path("logs")

INPUT_OBSERVATION_SOCKET = "tcp://localhost:5556"
OUTPUT_ACTION_SOCKET = "tcp://127.0.0.1:5555"
EMPTY_SIGNAL_SOCKET = "tcp://localhost:5558" # signal to empty action chunk queue

JOINT_ORDER = [
    'shoulder_pan.pos',
    'shoulder_lift.pos',
    'elbow_flex.pos',
    'wrist_flex.pos',
    'wrist_roll.pos',
    'gripper.pos',
]

# define feature structure of output action
# i.e. target joint states
ACTION_FEATURES = {
    'action': {
        'dtype': 'float32',
        'shape': (6,),
        'names': [
            'shoulder_pan.pos', 
            'shoulder_lift.pos', 
            'elbow_flex.pos', 
            'wrist_flex.pos', 
            'wrist_roll.pos', 
            'gripper.pos',
        ],
    },
}

# define feature structure of intput observation
# i.e. images and current joint states
OBS_FEATURES = {
    'observation.state': {
        'dtype': 'float32',
        'shape': (6,),
        'names': [
            'shoulder_pan.pos', 
            'shoulder_lift.pos', 
            'elbow_flex.pos', 
            'wrist_flex.pos', 
            'wrist_roll.pos', 
            'gripper.pos',
        ],
    },
    'observation.images.camera1': {
        'dtype': 'video',
        'shape': (480, 640, 3),
        'names': [
            'height', 
            'width', 
            'channels',
        ],
    },
    'observation.images.camera2': {
        'dtype': 'video',
        'shape': (480, 640, 3),
        'names': [
            'height', 
            'width', 
            'channels',
        ],
    },
}

DATASET_FEATURES = {**ACTION_FEATURES, **OBS_FEATURES}

# radian range of so100 joints in Isaac Sim
SIMULATION_RANGE = {
    # TODO: for now, always use so101 range
    "so101_old_calib": {
        'shoulder_pan.pos': {'sim_min': -2.0, 'sim_max': 2.0},
        'shoulder_lift.pos': {'sim_min': 0.0, 'sim_max': 3.5},
        'elbow_flex.pos': {'sim_min': -3.142, 'sim_max': 0.0},
        'wrist_flex.pos': {'sim_min': -2.5, 'sim_max': 1.2},
        'wrist_roll.pos': {'sim_min': -3.142, 'sim_max': 3.142},
        'gripper.pos': {'sim_min': -0.2, 'sim_max': 2.0},
    },
    # TODO: but the real robot behavior looks really like so101_new_calib, i.e. moves aggresively
    # TODO: maybe because the homing position of real robot used similar middle positions with new calib
    # e.g. in so101, the homing position
    "so101_new_calib": {
        'shoulder_pan.pos': {'sim_min': -1.92, 'sim_max': 1.92},
        'shoulder_lift.pos': {'sim_min': -1.75, 'sim_max': 1.75},
        'elbow_flex.pos': {'sim_min': -1.69, 'sim_max': 1.69},
        'wrist_flex.pos': {'sim_min': -1.66, 'sim_max': 1.66},
        'wrist_roll.pos': {'sim_min': -2.74, 'sim_max': 2.84},
        'gripper.pos': {'sim_min': -0.17, 'sim_max': 1.74},
    },
}

HOME_JOINT_STATE = {
    "old_calib_home": {
        'shoulder_pan.pos': 0.0,
        'shoulder_lift.pos': 0.0,
        'elbow_flex.pos': 0.0,
        'wrist_flex.pos': 40.0,
        'wrist_roll.pos': 0.0,
        'gripper.pos': 0.0,
    },
    # "new_calib_home": {
    #     'shoulder_pan.pos': -4.0,
    #     'shoulder_lift.pos': -99.0,
    #     'elbow_flex.pos': 98.0,
    #     'wrist_flex.pos': -80.0,
    #     'wrist_roll.pos': -55.0,
    #     'gripper.pos': 0.0,
    # },
    "new_calib_home": {
        'shoulder_pan.pos': 0.0,
        'shoulder_lift.pos': 0.0,
        'elbow_flex.pos': 0.0,
        'wrist_flex.pos': 0.0,
        'wrist_roll.pos': 0.0,
        'gripper.pos': 0.0,
    },
}


# TODO: update rad2pos and pos2rad according to norm mode, since gripper uses a different norm range

def rad2pos(rad: float, joint_name: str, robot: str):
    """Convert radians into motor pos.
    
    Isaac Sim joint state has real radian values, while SO100 robot hardware
    use -100 to 100 as "pos". Original script for SO100 follower, i.e. using_smolvla_example.py,
    use the default value of RobotConfig.use_degrees = False, so it use MotorNormMode.RANGE_M100_100,
    check norm_mode_body of class so100_foller.SO100Follower
    """

    sim_min = SIMULATION_RANGE[robot][joint_name]['sim_min']
    sim_max = SIMULATION_RANGE[robot][joint_name]['sim_max']

    # TODO: use this after fine-tuning
    # if joint_name == 'gripper.pos':
    #     # norm = ((bounded_val - min_) / (max_ - min_)) * 100
    #     pos = ((rad - sim_min) / (sim_max - sim_min)) * 100
    # else:
    #     # norm = (((bounded_val - min_) / (max_ - min_)) * 200) - 100
    #     pos = ((rad - sim_min) / (sim_max - sim_min)) * 200 - 100

    # TODO: this gives a larger gripper joint value, even though it should use MotorNormMode.RANGE_0_100
    pos = ((rad - sim_min) / (sim_max - sim_min)) * 200 - 100
    return pos


def pos2rad(pos: float, joint_name: str, robot: str):
    """Convert the noramlized motor pos into real radian."""

    sim_min = SIMULATION_RANGE[robot][joint_name]['sim_min']
    sim_max = SIMULATION_RANGE[robot][joint_name]['sim_max']

    # TODO: use this after fine-tuning
    # if joint_name == 'gripper.pos':
    #     # unnormalized_values[id_] = int((bounded_val / 100) * (max_ - min_) + min_)
    #     rad = (pos / 100) * (sim_max - sim_min) + sim_min
    # else:
    #     # unnormalized_values[id_] = int(((bounded_val + 100) / 200) * (max_ - min_) + min_)
    #     rad = (pos + 100) / 200 * (sim_max - sim_min) + sim_min
    
    # TODO: use correct version for gripper later
    rad = (pos + 100) / 200 * (sim_max - sim_min) + sim_min
    return rad


def remap_action_between_calibrations(
    action: Dict[str, float],
    source_robot: str = "so101_old_calib",
    target_robot: str = "so101_new_calib",
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
            robot=source_robot,
        )
        target_pos = rad2pos(
            rad=source_rad,
            joint_name=joint_name,
            robot=target_robot,
        )
        if clip:
            target_pos = float(np.clip(target_pos, -100.0, 100.0))
        remapped_action[joint_name] = target_pos

    return remapped_action


if __name__ == "__main__":

    current_time = time.strftime("%Y%m%d_%H%M%S")
    record_action_folder = Path(__file__).resolve().parent / "recorded_actions"
    os.makedirs(record_action_folder, exist_ok=True)

    # robot
    args = parse_args()

    # to receive observation as input
    obs_context = zmq.Context()
    obs_socket = obs_context.socket(zmq.SUB)
    obs_socket.connect(INPUT_OBSERVATION_SOCKET)
    obs_socket.setsockopt(zmq.SUBSCRIBE, b"")

    # to receive a signal to decide if empty current action chunk
    signal_context = zmq.Context()
    signal_socket = signal_context.socket(zmq.SUB)
    signal_socket.connect(EMPTY_SIGNAL_SOCKET)
    signal_socket.setsockopt(zmq.SUBSCRIBE, b"")

    # to send out the proposed action chunk
    act_context = zmq.Context()
    act_socket = act_context.socket(zmq.PUB)
    act_socket.bind(OUTPUT_ACTION_SOCKET)
    time.sleep(0.5) # Give subscribers a short time to connect

    os.makedirs(IMAGES_STORE_PATH, exist_ok=True)

    recorded_actions: Dict[int, Dict[str, float]] = dict()

    try:

        # this id seems to be the name of calibration json
        # /home/hardli/.cache/huggingface/lerobot/calibration/robots/so101_follower/so101_follower_arm.json
        follower_calibration_json_filename = "so101_follower_new_calib"
        # follower_calibration_json_filename = "so101_follower_old_calib"

        robot_cfg = SO101FollowerConfig(
            port=FOLLOWER_PORT_ID,
            id=follower_calibration_json_filename,
            cameras={},
            # Set to a positive value to clip large jumps for safety, or None to send the exact target.
            max_relative_target=None,
        )
        robot = SO101Follower(config=robot_cfg)
        robot.connect()
        log_joint_state(
            joint_state=robot.get_observation(),
            logging_label="Initial joint state",
        )
        log_joint_state(
            joint_state=robot.get_current_goal_action(),
            logging_label="Current stored goal position",
        )

        # return to initial joint state
        curr_joint_state = robot.get_observation()
        log_joint_state(
            joint_state=curr_joint_state,
            logging_label="Moving to home pose, current joint state",
        )
        present_velocity, goal_velocity = robot.get_velocity()
        log_joint_state(
            joint_state=present_velocity,
            logging_label="Present velocity",
        )
        log_joint_state(
            joint_state=goal_velocity,
            logging_label="Goal velocity",
        )
        move_home_sequential(
            robot=robot, 
            calib_json="new_calib_home",
            reverse_order=True,
        )
        logging.info(f"Robot returns to home, wait for 3 seconds......")
        time.sleep(3)
        logging.info(f"Robot ready to go")

        model = SmolVLAPolicy.from_pretrained(MODEL_ID)

        logging.info(f"Model device: {next(model.parameters()).device}")
        logging.info(f"Robot model: {ROBOT}")
        logging.info(f"Current task: {TASK}")

        camera_feature_keys = list(model.config.image_features)
        max_supported_cameras = len(camera_feature_keys)
        if max_supported_cameras == 0:
            raise ValueError(f"Policy {MODEL_ID} exposes no camera inputs.")

        preprocess, postprocess = make_pre_post_processors(
            policy_cfg=model.config,
            pretrained_path=MODEL_ID,
            preprocessor_overrides={"device_processor": {"device": str(DEVICE)}},
        )

        count = 0

        while True:

            payload = obs_socket.recv()   # one npz blob
            buf = io.BytesIO(payload)
            data = np.load(buf)

            ts = float(data["ts"][0])
            joints = data["joints"].astype(np.float32)

            # image 1: from topic /camera1_rgb
            img1_vec = data["img1"]
            encoded_flag = int(data.get("img1_encoded", np.array([1]))[0])
            # convert JPEG bytes to numpy image
            if encoded_flag == 1:
                buf1 = np.frombuffer(img1_vec.tobytes(), dtype=np.uint8)
                img1 = cv2.imdecode(buf1, cv2.IMREAD_COLOR)
            else:
                img1 = img1_vec.reshape((480, 640, 3))
            # convert img into RGB
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

            # image 2: from topic /camera2_rgb
            img2_vec = data["img2"]
            encoded_flag = int(data.get("img2_encoded", np.array([1]))[0])
            # convert JPEG bytes to numpy image
            if encoded_flag == 1:
                buf2 = np.frombuffer(img2_vec.tobytes(), dtype=np.uint8)
                img2 = cv2.imdecode(buf2, cv2.IMREAD_COLOR)
            else:
                img2 = img2_vec.reshape((480, 640, 3))
            # convert img into RGB
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

            # The expected obs should have this structure:
            # {
            #   "shoulder_pan.pos": joint_value,
            #   "shoulder_lift.pos": joint_value,
            #   "elbow_flex.pos": joint_value,
            #   "wrist_flex.pos": joint_value,
            #   "wrist_roll.pos": joint_value,
            #   "gripper.pos": joint_value,
            #   "camera1": cv2.RGB image with shape h, w, c, no rotation
            #   "camera2": cv2.RGB image with shape h, w, c, no rotation
            # }

            # the joint states in obs should be normalized as relative position,
            # i.e. in range [-100, 100] for body joints or [0, 100] for gripper

            obs = {
                "shoulder_pan.pos": rad2pos(rad=joints[0], joint_name="shoulder_pan.pos", robot=ROBOT),
                "shoulder_lift.pos": rad2pos(rad=joints[1], joint_name="shoulder_lift.pos", robot=ROBOT),
                "elbow_flex.pos": rad2pos(rad=joints[2], joint_name="elbow_flex.pos", robot=ROBOT),
                "wrist_flex.pos": rad2pos(rad=joints[3], joint_name="wrist_flex.pos", robot=ROBOT),
                "wrist_roll.pos": rad2pos(rad=joints[4], joint_name="wrist_roll.pos", robot=ROBOT),
                "gripper.pos": rad2pos(rad=joints[5], joint_name="gripper.pos", robot=ROBOT),
                "camera1": img1,
                "camera2": img2,
            }

            # TODO: check what viewpoint angle and distance to robot (i.e. pose) should the camera have

            # the built obs_frame would have such structure:
            # {
            #   "observation.state": torch.Tensor with shape [1, 6]
            #   "observation.images.camera1": torch.Tensor of image batch with shape [1, c, h, w]
            #   "task": task string same as input,
            #   "robot_type": robot_type string same as input,
            # }
            obs_frame = build_inference_frame(
                observation=obs, 
                ds_features=DATASET_FEATURES, 
                device=DEVICE, 
                task=TASK, 
                robot_type=ROBOT_TYPE,
            )

            obs = preprocess(obs_frame)

            # save images in obs
            if STORE_IMAGES and count % LOG_HZ == 0:
                # cam1
                cam1_img = obs['observation.images.camera1'].cpu().detach().numpy().squeeze().reshape(480, 640, 3)
                cam1_img_uint8 = (cam1_img * 255).astype(np.uint8)
                cam1_img_uint8_save = Image.fromarray(cam1_img_uint8)  # expects RGB order
                cam1_img_uint8_save.save(IMAGES_STORE_PATH / f"obs_cam1_{count}.png")
                # cam2
                cam2_img = obs['observation.images.camera2'].cpu().detach().numpy().squeeze().reshape(480, 640, 3)
                cam2_img_uint8 = (cam2_img * 255).astype(np.uint8)
                cam2_img_uint8_save = Image.fromarray(cam2_img_uint8)  # expects RGB order
                cam2_img_uint8_save.save(IMAGES_STORE_PATH / f"obs_cam2_{count}.png")

            # if receive a stop signal from safety estimator,
            # empty the current action chunk
            # get empty signal
            try:
                empty_msg_bytes = signal_socket.recv(flags=zmq.NOBLOCK)
                empty = np.frombuffer(empty_msg_bytes, dtype=np.float32)[0]
            except zmq.Again:
                empty = 0.0
            if empty:
                model._queues["action"].clear()
                print(f"Too many unsafe actions, clear the current chunk......")

            if count % LOG_HZ == 0:
                log_joint_state(
                    joint_state=robot.get_observation(),
                    logging_label="Current joint state (calibrated position)",
                )

            ### send action

            action = model.select_action(obs)
            action = postprocess(action)

            # the returned action would have such structure:
            # {
            #     'shoulder_pan.pos': abs_target_value,
            #     'shoulder_lift.pos': abs_target_value,
            #     'elbow_flex.pos': abs_target_value,
            #     'wrist_flex.pos': abs_target_value, 
            #     'wrist_roll.pos': abs_target_value, 
            #     'gripper.pos': abs_target_value, 
            # }
            raw_action = make_robot_action(action, DATASET_FEATURES)
            if count % LOG_HZ == 0:
                log_joint_state(
                    joint_state=raw_action,
                    logging_label="Raw actions from model (calibrated position)",
                )

            if RECORD_ACTIONS:
                recorded_actions[count] = raw_action

            # send the actions to real robot
            # TODO: it seems the pre-trained smolvla base model outputs action in so101 norm range, instead of so101_new_calib range
            if args.send_action:
                # remapped_action = remap_action_between_calibrations(
                #     action=raw_action,
                #     source_robot="so101_old_calib",
                #     target_robot="so101_new_calib",
                #     clip=True,
                # )
                # real_action = robot.send_action(remapped_action)
                real_action = robot.send_action(raw_action)
                if count % LOG_HZ == 0:
                    log_joint_state(
                        joint_state=real_action,
                        logging_label="Actions sent to real robot (calibrated position)",
                    )
            else:
                logging.info("No actions are sent to real robot.")


            # send the actions to simulated robot
            sim_action = np.array([
                pos2rad(pos=raw_action["shoulder_pan.pos"], joint_name="shoulder_pan.pos", robot=ROBOT),
                pos2rad(pos=raw_action["shoulder_lift.pos"], joint_name="shoulder_lift.pos", robot=ROBOT),
                pos2rad(pos=raw_action["elbow_flex.pos"], joint_name="elbow_flex.pos", robot=ROBOT),
                pos2rad(pos=raw_action["wrist_flex.pos"], joint_name="wrist_flex.pos", robot=ROBOT),
                pos2rad(pos=raw_action["wrist_roll.pos"], joint_name="wrist_roll.pos", robot=ROBOT),
                pos2rad(pos=raw_action["gripper.pos"], joint_name="gripper.pos", robot=ROBOT),
            ],dtype=np.float32)

            if count % LOG_HZ == 0:
                print_action = ', '.join(f"{value:2f}" for value in sim_action.tolist())
                logging.info(f"Actions sent to simulated robot (radian): [{print_action}]")

            print()
            # send actions to zmq socket
            act_socket.send(sim_action.tobytes())

            count += 1

    except KeyboardInterrupt:
        logging.info("Ctrl-C received")

    finally:
        
        if RECORD_ACTIONS:
            recroded_actions_output_json = Path(f"{current_time}_recorded_actions.json")
            logging.info(f"Record the current action series into {record_action_folder}")
            with open(record_action_folder / recroded_actions_output_json, "w") as f:
                json.dump(recorded_actions, f, indent=2)

        logging.info("Releasing robot torque and disconnecting")
        if robot.is_connected:
            try:
                robot.bus.disable_torque(num_retry=5)
            except Exception as e:
                logging.warning(f"Could not disable torque: {e}")

            try:

                robot.disconnect()
            except Exception as e:
                logging.warning(f"Could not disconnect robot: {e}")