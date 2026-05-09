"""
only show current joint state:
$ python test_follower.py

send action to robot to execute:
$ python test_follower.py --send-action

"""

import argparse
import time
from pprint import pformat

from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower


LOG_SECONDS = 5.0
LOG_HZ = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--send-action",
        action="store_true",
        help="Send the target pose before logging. Without this flag, the script only logs joint positions.",
    )
    return parser.parse_args()


def log_joint_state(robot: SO101Follower, label: str) -> None:
    observation = robot.get_observation()
    joint_state = {key: value for key, value in observation.items() if key.endswith(".pos")}
    print(f"{label}:\n{pformat(joint_state, sort_dicts=False)}")


def main():
    args = parse_args()

    # Find ports using: lerobot-find-port
    follower_port = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AAF218449-if00"

    # This id is used to load the matching calibration file.
    # follower_id = "follower_so101"
    follower_id = "so101_follower_arm"
    # TODO: try "so101_follower_arm" if the default one raise error, since this id seems to be the name of calibration json
    # /home/hardli/.cache/huggingface/lerobot/calibration/robots/so101_follower/so101_follower_arm.json

    robot_cfg = SO101FollowerConfig(
        port=follower_port,
        id=follower_id,
        cameras={},
        # Set to a positive value to clip large jumps for safety, or None to send the exact target.
        max_relative_target=None,
    )
    robot = SO101Follower(config=robot_cfg)

    initial_pose = {
        'shoulder_pan.pos': -4.466592838685855,
        'shoulder_lift.pos': -99.33026370866472,
        'elbow_flex.pos': 98.45665002269632,
        'wrist_flex.pos': 79.51176983435047,
        'wrist_roll.pos': -52.22544113774032,
        'gripper.pos': 0.3434065934065934,
    }

    target_pose = {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": -100,
        "elbow_flex.pos": 100,
        "wrist_flex.pos": 80,
        "wrist_roll.pos": -52,
        "gripper.pos": 3,
    }

    try:
        robot.connect()

        log_joint_state(robot, "Initial joint state")
        if args.send_action:
            sent_action = robot.send_action(target_pose)
            print(f"Sent target pose:\n{pformat(sent_action, sort_dicts=False)}")
        else:
            print("Logging only. No action sent. Use --send-action to command the target pose.")

        dt = 1.0 / LOG_HZ
        end_time = time.perf_counter() + LOG_SECONDS
        while time.perf_counter() < end_time:
            log_joint_state(robot, "Current joint state")
            time.sleep(dt)
    finally:
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
