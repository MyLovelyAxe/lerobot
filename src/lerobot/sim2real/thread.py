import threading
import queue
import time
import zmq
import logging
logging.basicConfig(level=logging.INFO)

from typing import Dict, List, Union
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader
from lerobot.utils.robot_utils import precise_sleep
from lerobot.sim2real.utils import log_joint_state
from lerobot.sim2real.constant import (
    JOINT_ORDER,
    SO101_NEW_CALIB,
    HOME_TOL,
)
from lerobot.sim2real.utils import (
    move_robot_to_target_pose,
    # move_leader_to_target_pose,
    add_send_action_leader,
    log_joint_state,
    joint_state_pos2rad,
    get_rad_joint_state_from_socket,
)


def reset_sim_robot_worker(
    initial_pose: Dict[str, float],
    obs_socket: zmq.SyncSocket,
    action_socket: zmq.SyncSocket,
    joint_order: List[str] = JOINT_ORDER,
):
    """The thread to move the simulated robot to initial pose.

    :param initial_pose: starting pose of simulated robot
    :param obs_socket: the socket to read the current joint states of simulated robot
    :param action_socket: the socket to send actions to simulated robot
    :param joint_order: joint names in order
    """

    # send action for simulated robot to execute
    sim_action = joint_state_pos2rad(
        pos_joint_state=initial_pose,
        calibration=SO101_NEW_CALIB,
    )
    action_socket.send(sim_action.tobytes())
    logging.info(f"Move simulated robot to initial pose......")

    # check if the reset is done
    finished = False
    while not finished:
        sim_joint_state = get_rad_joint_state_from_socket(
            obs_socket=obs_socket,
        )
        difference = list()
        for joint_name in joint_order:
            difference.append(initial_pose[joint_name] - sim_joint_state[joint_name])
        if all(abs(d) <= HOME_TOL for d in difference):
            print(f"all finished")
            finished = True
    
    logging.info(f"Simulated robot is ready to go")


def reset_real_robot_worker(
    initial_pose: Dict[str, float],
    robot: Union[SO101Follower, SO101Leader],
):
    """The thread to move the real robot to initial pose.

    :param initial_pose: starting pose of simulated robot
    :param robot: the SO101Follower object to connect with real robot
    """
    move_robot_to_target_pose(
        robot=robot, 
        target_pose=initial_pose,
        reverse_order=True,
    )
    logging.info(f"Real robot is ready to go")


def send_action_worker(
    send_action_finish: threading.Event,
    robot_lock: threading.Lock,
    record: queue.Queue,
    dt: float,
    all_action_trajectories: List[List[Dict[str, float]]],
    action_socket: zmq.SyncSocket,
    robot: Union[SO101Follower, SO101Leader],
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
    # if the robot is a leader, temporarily add a .send_action() to it
    if isinstance(robot, SO101Leader) and not hasattr(robot, 'send_action'):
        robot = add_send_action_leader(robot=robot)

    logging.info("Thread send_action_worker begins.")

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
                    calibration=SO101_NEW_CALIB,
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
        sim_joint_state = get_rad_joint_state_from_socket(
            obs_socket=obs_socket,
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
    robot: Union[SO101Follower, SO101Leader],
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
            if isinstance(robot, SO101Follower):
                real_joint_state = robot.get_observation()
            if isinstance(robot, SO101Leader):
                real_joint_state = robot.get_action()
        latency_info["exec_real"][record_real_time] = real_joint_state
        precise_sleep(dt - (time.perf_counter() - record_real_time))

    record.put(latency_info)

    logging.info("Thread record_real_joint_state_worker ends.")