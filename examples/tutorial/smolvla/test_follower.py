"""
only show trajectory joint state:
$ python test_follower.py

only send action to simulated robot:
$ python test_follower.py --sim

only send action to real robot:
$ python test_follower.py --real

synchronize simulated and real robots:
$ python test_follower.py --sim --real
"""

import argparse
import time
import zmq
import logging
logging.basicConfig(level=logging.INFO)

from typing import Dict
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.sim2real_utils import (
    generate_robot_actions_trajectory,
    log_joint_state,
)
from lerobot.utils.sim2real_constant import (
    JOINT_ORDER,
    OUTPUT_ACTION_SOCKET,
    SO101_FOLLOWER_PORT_ID,
    RETURN_JOINT_STATE,
    SO101_FOLLOWER_NEW_CALIB,
)
from lerobot.utils.sim2real_utils import (
    move_robot_to_target_pose,
    log_joint_state,
    joint_state_pos2rad,
)


LOG_SECONDS = 5.0
LOG_HZ = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sim",
        action="store_true",
        help="Send target pose to simulation. Without this flag, the script only logs joint positions.",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Send target pose to real robot. Without this flag, the script only logs joint positions.",
    )
    parser.add_argument(
        "--exec_duration",
        type=float,
        default=5.0,
        help="The duration of executing the trajectory from initial pose to target pose, in seconds.",
    )
    parser.add_argument(
        "--exec_steps_per_sec",
        type=int,
        default=80, # 50 times per second
        help="How many steps will the trajectory be executed per second.",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=True,
        help="Whether log the process or not.",
    )
    return parser.parse_args()


def main():
    
    args = parse_args()
    robot_cfg = SO101FollowerConfig(
        port=SO101_FOLLOWER_PORT_ID,
        id=SO101_FOLLOWER_NEW_CALIB,
        cameras={},
        # Set to a positive value to clip large jumps for safety, or None to send the exact target.
        max_relative_target=None,
    )
    robot = SO101Follower(config=robot_cfg)

    # to send out the proposed action chunk
    act_context = zmq.Context()
    act_socket = act_context.socket(zmq.PUB)
    act_socket.bind(OUTPUT_ACTION_SOCKET)
    time.sleep(0.5) # Give subscribers a short time to connect

    initial_pose = RETURN_JOINT_STATE[SO101_FOLLOWER_NEW_CALIB]

    target_pose = {
        'shoulder_pan.pos': -50.0,
        'shoulder_lift.pos': -50.0,
        'elbow_flex.pos': 50.0,
        'wrist_flex.pos': 50.0,
        'wrist_roll.pos': -40.0,
        'gripper.pos': 30.0,
    }

    dt = 1 / args.exec_steps_per_sec
    action_trajectory = generate_robot_actions_trajectory(
        start_state=initial_pose,
        target_state=target_pose,
        T=args.exec_duration, 
        dt=dt,
    )

    try:
        # move to initial pose firstly
        # TODO: make sim and real reseting into separate thread
        if args.sim:
            sim_action = joint_state_pos2rad(
                pos_joint_state=initial_pose,
                calibration=SO101_FOLLOWER_NEW_CALIB,
            )
            act_socket.send(sim_action.tobytes())
            logging.info(f"Move simulated robot to initial pose......")
            # TODO: how to check if simulated robot finishes reseting?
            # the move_robot_to_target_pose for real robot has precise control inside
            # but for isaac sim, just sending target pose to controller
            # temporarily just set a higher waiting time
            time.sleep(5)
            logging.info(f"Simulated robot is ready to go")
        if args.real:
            robot.connect()
            move_robot_to_target_pose(
                robot=robot, 
                target_pose=initial_pose,
                reverse_order=True,
            )
            logging.info(f"Robot returns to home, wait for 3 seconds......")
            time.sleep(3)
            logging.info(f"Hardware robot is ready to go")

        # exeucte the trajectory
        for curr_action in action_trajectory:
            loop_start = time.perf_counter()
            # TODO: make sim and real into 2 threads
            if args.real:
                sent_real_action = robot.send_action(curr_action)
                if args.verbose:
                    log_joint_state(
                        joint_state=sent_real_action,
                        logging_label="Sent action to real robot",
                    )
            if args.sim:
                sent_sim_action = joint_state_pos2rad(
                    pos_joint_state=curr_action,
                    calibration=SO101_FOLLOWER_NEW_CALIB,
                )
                act_socket.send(sent_sim_action.tobytes())
                if args.verbose:
                    log_joint_state(
                        joint_state=dict(zip(JOINT_ORDER, sent_sim_action)),
                        logging_label="Sent action to simulated robot",
                    )
            precise_sleep(dt - (time.perf_counter() - loop_start))
        

    finally:
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
