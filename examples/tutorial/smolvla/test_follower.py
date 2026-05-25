"""
only show current joint state:
$ python test_follower.py

send action to robot to execute:
$ python test_follower.py --send-action

"""

import argparse
import time
import numpy as np

from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.sim2real_utils import (
    generate_robot_actions_trajectory,
    log_joint_state,
)
from lerobot.utils.sim2real_constant import (
    JOINT_ORDER,
)

LOG_SECONDS = 5.0
LOG_HZ = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--send-action",
        action="store_true",
        help="Send the target pose before logging. Without this flag, the script only logs joint positions.",
    )
    parser.add_argument(
        "--exec_duration",
        type=float,
        default=5.0,
        help="The duration of executing the trajectory from initial pose to target pose, in seconds.",
    )
    parser.add_argument(
        "--exec_steps",
        type=int,
        default=80*5, # 50 times per second
        help="How many steps will the trajectory be executed.",
    )
    return parser.parse_args()




def main():
    
    args = parse_args()
    follower_port = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AAF218449-if00"

    # this id seems to be the name of calibration json
    # /home/hardli/.cache/huggingface/lerobot/calibration/robots/so101_follower/so101_follower_arm.json
    
    # follower_id = "so101_follower_old_calib" 

    # # NOTE: can't directly load new_calib for testing, since after lerobot-calibrate, 
    # the calibration values are registered into the motors and stay there (confirm this),
    # and the code examine if the stored register values are the same with the ones trying to load, 
    # if not, then "mismatch" and will ask for recalibration
    follower_id = "so101_follower_new_calib"

    robot_cfg = SO101FollowerConfig(
        port=follower_port,
        id=follower_id,
        cameras={},
        # Set to a positive value to clip large jumps for safety, or None to send the exact target.
        max_relative_target=None,
    )
    robot = SO101Follower(config=robot_cfg)



    target_pose = {
        'shoulder_pan.pos': -50.0,
        'shoulder_lift.pos': -50.0,
        'elbow_flex.pos': 50.0,
        'wrist_flex.pos': 50.0,
        'wrist_roll.pos': -40.0,
        'gripper.pos': 30.0,
    }


    try:
        robot.connect()

        initial_pose = robot.get_observation()
        dt = args.exec_duration / args.exec_steps
        action_trajectory = generate_robot_actions_trajectory(
            start_state=initial_pose,
            target_state=target_pose,
            T=args.exec_duration, 
            dt=dt,
        )

        log_joint_state(
            joint_state=robot.get_observation(),
            logging_label="Initial joint state",
        )
        if args.send_action:
            for curr_action in action_trajectory:
                loop_start = time.perf_counter()
                sent_action = robot.send_action(curr_action)
                log_joint_state(
                    joint_state=sent_action,
                    logging_label="Sent action",
                )
                precise_sleep(dt - (time.perf_counter() - loop_start))
        else:
            print("Logging only. No action sent. Use --send-action to command the target pose.")

    finally:
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
