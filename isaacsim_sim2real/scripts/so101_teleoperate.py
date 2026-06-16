"""
Use leader arm to teleoperate real follower arm:
$ python lerobot_teleoperate.py --real

Use leader arm to teleoperate simulated arm:
$ python lerobot_teleoperate.py --sim

Use leader arm to teleoperate real and sim follower arm:
$ python lerobot_teleoperate.py --sim --real
"""
import argparse
import threading
import queue
import time
import zmq
import logging
from typing import Dict
logging.basicConfig(level=logging.INFO)

from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader
from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
from lerobot.sim2real.constant import (
    SO101_FOLLOWER_PORT_ID,
    SO101_LEADER_PORT_ID,
    SO101_FOLLOWER_NEW_CALIB,
    SO101_LEADER_NEW_CALIB,
    OUTPUT_ACTION_SOCKET,
    INPUT_OBSERVATION_SOCKET,
    JOINT_ORDER,
)
from lerobot.sim2real.thread import (
    reset_sim_robot_worker,
    reset_real_robot_worker,
    teleoperate_worker,
    record_sim_joint_state_worker,
    record_real_joint_state_worker,
)
from lerobot.sim2real.utils import (
    compute_latency,
)
from lerobot.sim2real.debug import (
    plot_joint_lines_in_record,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sim",
        action="store_true",
        default=True,
        help="Send target pose to simulation. Without this flag, the script only logs joint positions.",
    )
    parser.add_argument(
        "--real",
        default=True,
        action="store_true",
        help="Send target pose to real robot. Without this flag, the script only logs joint positions.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=100,
        help="The frequency of read action from leader and send to target arm, in unit of Hz.",
    )
    parser.add_argument(
        "--max_record_duration",
        type=float,
        default=30.0,
        help="The maximum duration to record the trajectories for computing latency, in unit of seconds.",
    )
    parser.add_argument(
        "--plot_record",
        type=bool,
        default=True,
        help="Whether save a plot of recorded trajectories.",
    )
    return parser.parse_args()


def main():

    args = parse_args()

    # action source
    leader = SO101Leader(
        config=SO101LeaderConfig(
            port=SO101_LEADER_PORT_ID, 
            id=SO101_LEADER_NEW_CALIB,
        ),
    )

    # real follower arm
    follower = SO101Follower(
        config=SO101FollowerConfig(
            port=SO101_FOLLOWER_PORT_ID, 
            id=SO101_FOLLOWER_NEW_CALIB,
        ),
    )
    leader.connect()

    # simulated follower arm
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


    try:

        # take the leader arm's current pose as initial pose
        
        initial_pose = leader.get_action()

        # create threads and shared record buffer
        
        latency_record = queue.Queue()
        stop_event = threading.Event()
        robot_lock = threading.Lock() # to avoid multiple threads to read or write to motor bus at the same time
        
        reset_sim_thread = threading.Thread(
            target=reset_sim_robot_worker,
            args=(initial_pose, obs_socket, act_socket, JOINT_ORDER),
        )
        reset_real_thread = threading.Thread(
            target=reset_real_robot_worker,
            args=(initial_pose, follower),
        )
        teleoperate_thread = threading.Thread(
            target=teleoperate_worker,
            args=(stop_event, robot_lock, latency_record, 1 / args.fps, leader, follower, act_socket, args.sim, args.real, False),
        )
        record_sim_action_thread = threading.Thread(
            target=record_sim_joint_state_worker, 
            args=(stop_event, latency_record, 1.0/200, obs_socket),
        )
        record_real_action_thread = threading.Thread(
            target=record_real_joint_state_worker, 
            args=(stop_event, robot_lock, latency_record, 1.0/200, follower),
        )

        # move to initial pose firstly

        if args.sim:
            reset_sim_thread.start()
        if args.real:
            follower.connect()
            reset_real_thread.start()

        if args.sim:
            reset_sim_thread.join()
        if args.real:
            reset_real_thread.join()

        # execute trajectories

        teleoperate_thread.start()
        if args.sim:
            record_sim_action_thread.start()
        if args.real:
            record_real_action_thread.start()

        # keep main thread alive
        while not stop_event.is_set():
            time.sleep(0.1)

    except KeyboardInterrupt:

        logging.info("Ctrl + C received")
        stop_event.set()

    finally:

        teleoperate_thread.join()
        if args.sim:
            record_sim_action_thread.join()
        if args.real:
            record_real_action_thread.join()

        # compute latency

        record: Dict[str, Dict[str, Dict[str, float]]] = dict()
        plot_title = ""
        while not latency_record.empty():
            record.update(latency_record.get())

        # Only keep the limited duration for computing latency, in case the record is too long
        for latency_label, recored_trajectory in record.items():
            timestamps = sorted(float(ts) for ts in recored_trajectory)
            if not timestamps:
                continue
            cutoff = timestamps[0] + args.max_record_duration
            record[latency_label] = {
                ts: v for ts, v in recored_trajectory.items()
                if float(ts) <= cutoff
            }

        if args.sim:
            sim_latency = compute_latency(
                reference_actions=record["send_sim"],
                target_actions=record["exec_sim"],
                percentage=(0.1,0.9),
            )
            sim_latenfy_info = f"sim_latency: {sim_latency * 1000:.1f} ms "
            logging.info(sim_latenfy_info)
            plot_title += sim_latenfy_info

        if args.real:
            real_latency = compute_latency(
                reference_actions=record["send_real"],
                target_actions=record["exec_real"],
                percentage=(0.1,0.9),
            )
            real_latency_info = f"real_latency: {real_latency * 1000:.1f} ms "
            logging.info(real_latency_info)
            plot_title += real_latency_info

        if args.sim and args.real:
            sim2real_latency = compute_latency(
                reference_actions=record["exec_sim"],
                target_actions=record["exec_real"],
                percentage=(0.1,0.9),
            )
            sim2real_latency_info = f"sim2real_latency: {sim2real_latency * 1000:.1f} ms "
            logging.info(sim2real_latency_info)
            plot_title += sim2real_latency_info

        if args.plot_record:

            plot_joint_lines_in_record(
                record=record,
                which_record=["send_sim", "exec_sim", "exec_real"],
                title=plot_title,
                store=True,
            )


        if leader.is_connected:
            leader.disconnect()
        if follower.is_connected:
            follower.disconnect()


if __name__ == "__main__":

    main()