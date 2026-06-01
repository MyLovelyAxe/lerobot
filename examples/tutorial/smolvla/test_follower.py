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

import io
import argparse
import threading
import queue
import time
import zmq
import logging
import numpy as np
logging.basicConfig(level=logging.INFO)

from typing import Dict, List
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.sim2real_utils import (
    generate_robot_actions_trajectory,
    log_joint_state,
)
from lerobot.utils.sim2real_constant import (
    JOINT_ORDER,
    INPUT_OBSERVATION_SOCKET,
    OUTPUT_ACTION_SOCKET,
    SO101_FOLLOWER_PORT_ID,
    RETURN_JOINT_STATE,
    SO101_FOLLOWER_NEW_CALIB,
)
from lerobot.utils.sim2real_utils import (
    move_robot_to_target_pose,
    log_joint_state,
    joint_state_pos2rad,
    joint_state_rad2pos,
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


def send_action_worker(
    send_action_finish: threading.Event,
    robot_lock: threading.Lock,
    record: queue.Queue,
    dt: float,
    all_action_trajectories: List[List[Dict[str, float]]],
    action_socket: zmq.SyncSocket,
    robot: SO101Follower,
    sim: bool = True,
    real: bool = True,
    verbose: bool = False,
):
    """The thread to send fixed trajectory to either simulated or real robot.
    
    :param send_action_finish: indicate if the actions are all sent
    :param robot_lock: a lock to avoid other threads to operate on motor bus when sending actions to real robot
    :param record: the shared queue to keep the latency info
    :param dt: the time interval to send an action
    :param all_action_trajectories: multiple fixed trajectories with actions
    :param action_socket: the socket to send actions to simulated robot
    :param robot: the SO101Follower object to connect with real robot
    :param sim: whether send to simulated robot
    :param real: whether sent to real robot
    :param verbose: whether log the process
    """
    logging.info("Thread send_action_worker begins.")

    # TODO: add a label to indicate when sending action is finished, let recording workers to stop
    # NOTE: only record calibrated normalized actions
    latency_info = {
        "send_real": dict(),
        "send_sim" : dict(),
    }
    # exeucte the trajectories
    for action_trajectory in all_action_trajectories:
        for curr_action in action_trajectory:
            loop_start = time.perf_counter()
            if real:
                with robot_lock:
                    send_real_time = time.perf_counter()
                    sent_real_action = robot.send_action(curr_action)
                latency_info["send_real"][send_real_time] = curr_action
                if verbose:
                    log_joint_state(
                        joint_state=sent_real_action,
                        logging_label="Sent action to real robot",
                    )
            if sim:
                sent_sim_action = joint_state_pos2rad(
                    pos_joint_state=curr_action,
                    calibration=SO101_FOLLOWER_NEW_CALIB,
                )
                send_sim_time = time.perf_counter()
                action_socket.send(sent_sim_action.tobytes())
                latency_info["send_sim"][send_sim_time] = curr_action
                if verbose:
                    log_joint_state(
                        joint_state=dict(zip(JOINT_ORDER, sent_sim_action)),
                        logging_label="Sent action to simulated robot",
                    )
            precise_sleep(dt - (time.perf_counter() - loop_start))

    record.put(latency_info)
    send_action_finish.set()

    logging.info("Thread send_action_worker ends.")


def record_sim_joint_state_worker(
    send_action_finish: threading.Event,
    record: queue.Queue,
    dt: float,
    obs_socket: zmq.SyncSocket,
):
    """The thread to real current joint states of simulated robot.

    :param send_action_finish: indicate if the actions are all sent
    :param record: the shared queue to keep the latency info
    :param dt: the time interval to read joint state from simulated robot
    :param obs_socket: the socket to read the current joint states of simulated robot
    """
    logging.info("Thread record_sim_joint_state_worker begins.")

    latency_info = {
        "exec_sim": dict(),
    }

    while not send_action_finish.is_set():
        record_sim_time = time.perf_counter()
        payload = obs_socket.recv()   # one npz blob
        buf = io.BytesIO(payload)
        data = np.load(buf)
        # NOTE: this is a list of joint state values, needs to convert
        sim_joint_state_rad = data["joints"].astype(np.float32)
        sim_joint_state = joint_state_rad2pos(
            rad_joint_state=sim_joint_state_rad,
            calibration=SO101_FOLLOWER_NEW_CALIB,
        )
        latency_info["exec_sim"][record_sim_time] = sim_joint_state
        precise_sleep(dt - (time.perf_counter() - record_sim_time))

    record.put(latency_info)

    logging.info("Thread record_sim_joint_state_worker ends.")


def record_real_joint_state_worker(
    send_action_finish: threading.Event,
    robot_lock: threading.Lock,
    record: queue.Queue,
    dt: float,
    robot: SO101Follower,
):
    """The thread to real current joint states of real robot.

    :param send_action_finish: indicate if the actions are all sent
    :param robot_lock: a lock to avoid other threads to operate on motor bus when read present position registers
    :param record: the shared queue to keep the latency info
    :param dt: the time interval to read joint state from real robot
    :param robot: the SO101Follower object to connect with real robot
    """
    logging.info("Thread record_real_joint_state_worker begins.")

    latency_info = {
        "exec_real": dict(),
    }
    while not send_action_finish.is_set():
        with robot_lock:
            record_real_time = time.perf_counter()
            real_joint_state = robot.get_observation()
        latency_info["exec_real"][record_real_time] = real_joint_state
        precise_sleep(dt - (time.perf_counter() - record_real_time))

    record.put(latency_info)

    logging.info("Thread record_real_joint_state_worker ends.")


def compute_latency(
    sent_actions: Dict[str, Dict[str, float]],
    exec_actions: Dict[str, Dict[str, float]],
) -> float:
    """Compute the latency [ms] between sending and executing actions.
    
    Both of sent_actions and exec_actions should have this structure:
    - timestamp: 
        - joint name: calibrated normalized state

    :param sent_actions: the time - action pairs when sent to robot
    :param exec_actions: the time - action pairs when truly executed on robot
    """
    return -1


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
    obs_socket.connect(INPUT_OBSERVATION_SOCKET)
    obs_socket.setsockopt(zmq.SUBSCRIBE, b"")

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

        # execute trajectories

        send_action_thread.start()
        if args.sim:
            time.sleep(0.5)
            record_sim_action_thread.start()
        if args.real:
            time.sleep(0.5)
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
                sent_actions=record["send_sim"],
                exec_actions=record["exec_sim"],
            )

        real_latency = None
        if args.real:
            real_latency = compute_latency(
                sent_actions=record["send_real"],
                exec_actions=record["exec_real"],
            )

        sim2real_latency = None
        if args.sim and args.real:
            sim2real_latency = compute_latency(
                # TODO: maybe update the arg names with action1 and action2?
                sent_actions=record["exec_sim"],
                exec_actions=record["exec_real"],
            )
        logging.info(
            f"real_latency: {real_latency} \n"
            f"sim_latency: {sim_latency} \n"
            f"sim2real_latency: {sim2real_latency} \n"
        )
        

    finally:
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
