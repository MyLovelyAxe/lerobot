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
import threading
import queue
import time
import zmq
import logging
logging.basicConfig(level=logging.INFO)

from typing import Dict
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.sim2real.utils import (
    generate_robot_actions_trajectory,
    compute_latency,
)
from lerobot.sim2real.constant import (
    JOINT_ORDER,
    INPUT_OBSERVATION_SOCKET,
    OUTPUT_ACTION_SOCKET,
    SO101_FOLLOWER_PORT_ID,
    RETURN_JOINT_STATE,
    SO101_FOLLOWER_NEW_CALIB,
    SO101_NEW_CALIB,
)
from lerobot.sim2real.thread import (
    reset_sim_robot_worker,
    reset_real_robot_worker,
    send_action_worker,
    record_sim_joint_state_worker,
    record_real_joint_state_worker,
)
from lerobot.sim2real.debug import (
    print_record,
    plot_joint_lines_in_record,
)


LOG_SECONDS = 5.0
LOG_HZ = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sim",
        action="store_true",
        default=False,
        help="Send target pose to simulation. Without this flag, the script only logs joint positions.",
    )
    parser.add_argument(
        "--real",
        default=False,
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
        default=100,
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

    # to receive simulated joint state as input
    obs_context = zmq.Context()
    obs_socket = obs_context.socket(zmq.SUB)
    obs_socket.setsockopt(zmq.CONFLATE, 1) # queue size is 1, new msg overwrites old msg
    obs_socket.setsockopt(zmq.SUBSCRIBE, b"")
    obs_socket.connect(INPUT_OBSERVATION_SOCKET)

    # to send out the proposed action chunk
    act_context = zmq.Context()
    act_socket = act_context.socket(zmq.PUB)
    act_socket.bind(OUTPUT_ACTION_SOCKET)
    time.sleep(0.5) # Give subscribers a short time to connect

    initial_pose = RETURN_JOINT_STATE[SO101_NEW_CALIB]

    target_pose = {
        'shoulder_pan.pos': -50.0,
        'shoulder_lift.pos': -50.0,
        'elbow_flex.pos': 50.0,
        'wrist_flex.pos': 50.0,
        'wrist_roll.pos': -40.0,
        'gripper.pos': 30.0,
    }

    dt = 1 / args.exec_steps_per_sec
    all_trajectories = list()
    action_trajectory_go = generate_robot_actions_trajectory(
        start_state=initial_pose,
        target_state=target_pose,
        T=args.exec_duration, 
        dt=dt,
    )
    all_trajectories.append(action_trajectory_go)
    action_trajectory_back = generate_robot_actions_trajectory(
        start_state=target_pose,
        target_state=initial_pose,
        T=args.exec_duration, 
        dt=dt,
    )
    all_trajectories.append(action_trajectory_back)

    try:
        # create threads and shared record buffer
        
        latency_record = queue.Queue()
        send_action_finish = threading.Event()
        robot_lock = threading.Lock() # to avoid multiple threads to read or write to motor bus at the same time

        reset_sim_thread = threading.Thread(
            target=reset_sim_robot_worker,
            args=(initial_pose, obs_socket, act_socket, JOINT_ORDER),
        )
        reset_real_thread = threading.Thread(
            target=reset_real_robot_worker,
            args=(initial_pose, robot),
        )
        send_action_thread = threading.Thread(
            target=send_action_worker, 
            args=(send_action_finish, robot_lock, latency_record, dt, all_trajectories, act_socket, robot, args.sim, args.real, False),
        )
        record_sim_action_thread = threading.Thread(
            target=record_sim_joint_state_worker, 
            args=(send_action_finish, latency_record, 1.0/200, obs_socket),
        )
        record_real_action_thread = threading.Thread(
            target=record_real_joint_state_worker, 
            args=(send_action_finish, robot_lock, latency_record, 1.0/200, robot),
        )

        # move to initial pose firstly

        if args.sim:
            reset_sim_thread.start()
        if args.real:
            robot.connect()
            reset_real_thread.start()

        if args.sim:
            reset_sim_thread.join()
        if args.real:
            reset_real_thread.join()

        # execute trajectories

        send_action_thread.start()
        if args.sim:
            record_sim_action_thread.start()
        if args.real:
            record_real_action_thread.start()

        send_action_thread.join()
        if args.sim:
            record_sim_action_thread.join()
        if args.real:
            record_real_action_thread.join()

        # compute latency

        record: Dict[str, Dict[str, Dict[str, float]]] = dict()
        while not latency_record.empty():
            record.update(latency_record.get())

        sim_latency = None
        if args.sim:
            sim_latency = compute_latency(
                reference_actions=record["send_sim"],
                target_actions=record["exec_sim"],
                percentage=(0.1,0.9),
            )
            logging.info(f"sim_latency: {sim_latency * 1000:.1f} ms")

        real_latency = None
        if args.real:
            real_latency = compute_latency(
                reference_actions=record["send_real"],
                target_actions=record["exec_real"],
                percentage=(0.1,0.9),
            )
            logging.info(f"real_latency: {real_latency * 1000:.1f} ms")

        sim2real_latency = None
        if args.sim and args.real:
            sim2real_latency = compute_latency(
                reference_actions=record["exec_sim"],
                target_actions=record["exec_real"],
                percentage=(0.1,0.9),
            )
            logging.info(f"sim2real_latency: {sim2real_latency * 1000:.1f} ms")

        
    finally:
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
