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
import cv2
import torch
import time
from PIL import Image
from pathlib import Path
from typing import Dict
import logging
logging.basicConfig(level=logging.INFO)

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.sim2real.constant import (
    INPUT_OBSERVATION_SOCKET,
    OUTPUT_ACTION_SOCKET,
    EMPTY_SIGNAL_SOCKET,
    DATASET_FEATURES,
    SO101_FOLLOWER_PORT_ID,
    MODEL_ID,
    RETURN_JOINT_STATE,
    SO101_FOLLOWER_NEW_CALIB,
    SO101_NEW_CALIB,
    HOME_JOINT_STATE,
)
from lerobot.sim2real.utils import (
    move_robot_to_target_pose,
    log_joint_state,
    joint_state_rad2pos,
    joint_state_pos2rad,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # basic config
    parser.add_argument(
        "--robot",
        type=str,
        default="so101_follower",
        help="The robot type",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="pick the red block",
        help="The language instruction to tell the robot what to do",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="The device to run the VLA model",
    )
    parser.add_argument(
        "--send-action",
        default=False,
        action="store_true",
        help="Send the returned action from policy to real robot. " \
        "Without this flag, the script only logs the returned actions " \
        "without sending to real robot.",
    )
    # log
    parser.add_argument(
        "--log_hz",
        type=int,
        default=1,
        help="The frequency to print out log info",
    )
    # record
    parser.add_argument(
        "--record_root",
        type=Path,
        default=Path(__file__).resolve().parent / "logs",
        help="The root folder to record results for analysis",
    )
    parser.add_argument(
        "--record_images",
        type=bool,
        default=False,
        help="Whether record the preprocessed images received from zmq socket",
    )
    parser.add_argument(
        "--record_actions",
        type=bool,
        default=False,
        help="Whether record the calibrated normalized actions returned by the VLA model into json",
    )

    return parser.parse_args()


if __name__ == "__main__":

    current_time = time.strftime("%Y%m%d_%H%M%S")
    args = parse_args()
    device = torch.device(args.device)
    if args.record_images or args.record_actions:
        record_folder = args.record_root / current_time
        os.makedirs(record_folder, exist_ok=True)

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

    recorded_actions: Dict[int, Dict[str, float]] = dict()

    try:
        if args.send_action:
            robot_cfg = SO101FollowerConfig(
                port=SO101_FOLLOWER_PORT_ID,
                id=SO101_FOLLOWER_NEW_CALIB, # calibration.json name
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
            move_robot_to_target_pose(
                robot=robot, 
                target_pose=HOME_JOINT_STATE[SO101_NEW_CALIB],
                reverse_order=True,
            )
            logging.info(f"Robot returns to home, wait for 3 seconds......")
            time.sleep(3)
            logging.info(f"Robot ready to go")

        model = SmolVLAPolicy.from_pretrained(MODEL_ID)

        logging.info(f"Model device: {next(model.parameters()).device}")
        logging.info(f"Robot model: {args.robot}")
        logging.info(f"Current task: {args.task}")

        camera_feature_keys = list(model.config.image_features)
        max_supported_cameras = len(camera_feature_keys)
        if max_supported_cameras == 0:
            raise ValueError(f"Policy {MODEL_ID} exposes no camera inputs.")

        preprocess, postprocess = make_pre_post_processors(
            policy_cfg=model.config,
            pretrained_path=MODEL_ID,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
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

            # joint state
            obs = joint_state_rad2pos(
                rad_joint_state=joints,
                calibration=SO101_NEW_CALIB,
            )
            obs["camera1"] = img1
            obs["camera2"] = img2

            # build frame
            obs_frame = build_inference_frame(
                observation=obs, 
                ds_features=DATASET_FEATURES, 
                device=device, 
                task=args.task, 
                robot_type=args.robot,
            )
            obs = preprocess(obs_frame)

            # save images in obs
            if args.record_images and count % args.log_hz == 0:
                # cam1
                cam1_img = obs['observation.images.camera1'].cpu().detach().numpy().squeeze().reshape(480, 640, 3)
                cam1_img_uint8 = (cam1_img * 255).astype(np.uint8)
                cam1_img_uint8_save = Image.fromarray(cam1_img_uint8)  # expects RGB order
                cam1_img_uint8_save.save(record_folder / f"obs_cam1_{count}.png")
                # cam2
                cam2_img = obs['observation.images.camera2'].cpu().detach().numpy().squeeze().reshape(480, 640, 3)
                cam2_img_uint8 = (cam2_img * 255).astype(np.uint8)
                cam2_img_uint8_save = Image.fromarray(cam2_img_uint8)  # expects RGB order
                cam2_img_uint8_save.save(record_folder / f"obs_cam2_{count}.png")

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

            if args.send_action and count % args.log_hz == 0:
                log_joint_state(
                    joint_state=robot.get_observation(),
                    logging_label="Current joint state (calibrated position)",
                )

            ### send action
            action = model.select_action(obs)
            action = postprocess(action)
            raw_action = make_robot_action(action, DATASET_FEATURES)
            if count % args.log_hz == 0:
                log_joint_state(
                    joint_state=raw_action,
                    logging_label="Raw actions from model (calibrated position)",
                )
            if args.record_actions:
                recorded_actions[count] = raw_action

            # send the actions to real robot
            if args.send_action:
                real_action = robot.send_action(raw_action)
                if count % args.log_hz == 0:
                    log_joint_state(
                        joint_state=real_action,
                        logging_label="Actions sent to real robot (calibrated position)",
                    )
            else:
                logging.info("No actions are sent to real robot.")


            # send the actions to simulated robot
            sim_action = joint_state_pos2rad(
                pos_joint_state=raw_action,
                calibration=SO101_NEW_CALIB,
            )
            if count % args.log_hz == 0:
                print_action = ', '.join(f"{value:2f}" for value in sim_action.tolist())
                logging.info(f"Actions sent to simulated robot (radian): [{print_action}]")
            print()

            # send actions to zmq socket
            act_socket.send(sim_action.tobytes())

            count += 1

    except KeyboardInterrupt:
        logging.info("Ctrl-C received")

    finally:

        if args.send_action:
        
            if args.record_actions:
                recroded_actions_json = record_folder / "recorded_actions.json"
                logging.info(f"Record the current action series into {recroded_actions_json}")
                with open(recroded_actions_json, "w") as f:
                    json.dump(recorded_actions, f, indent=2)

            logging.info("Releasing robot torque and disconnecting")
            if robot.is_connected:
                try:
                    robot.bus.disable_torque(num_retry=5)
                except Exception as e:
                    logging.warning(f"Could not disable torque: {e}")

                try:
                    # move the robot to return pose
                    move_robot_to_target_pose(
                        robot=robot, 
                        target_pose=RETURN_JOINT_STATE[SO101_NEW_CALIB],
                        reverse_order=False,
                    )
                    robot.disconnect()
                except Exception as e:
                    logging.warning(f"Could not disconnect robot: {e}")

        else:
            logging.info(f"No real robot is involved, exit safely.")