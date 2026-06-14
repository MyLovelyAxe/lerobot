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
logging.basicConfig(level=logging.INFO)

from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader
from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.sim2real.constant import (
    SO101_FOLLOWER_PORT_ID,
    SO101_LEADER_PORT_ID,
    SO101_FOLLOWER_NEW_CALIB,
    SO101_LEADER_NEW_CALIB,
    OUTPUT_ACTION_SOCKET,
    INPUT_OBSERVATION_SOCKET,
    JOINT_ORDER,
    SO101_NEW_CALIB,
)
from lerobot.sim2real.thread import (
    reset_sim_robot_worker,
    reset_real_robot_worker,
)
from lerobot.sim2real.utils import (
    joint_state_pos2rad,
)

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
        "--fps",
        type=int,
        default=100,
        help="The frequency of read action from leader and send to target arm, in unit of Hz.",
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

        # create threads
        reset_sim_thread = threading.Thread(
            target=reset_sim_robot_worker,
            args=(initial_pose, obs_socket, act_socket, JOINT_ORDER),
        )
        reset_real_thread = threading.Thread(
            target=reset_real_robot_worker,
            args=(initial_pose, follower),
        )

        if args.sim:
            reset_sim_thread.start()
        if args.real:
            follower.connect()
            reset_real_thread.start()

        if args.sim:
            reset_sim_thread.join()
        if args.real:
            reset_real_thread.join()

        while True:

            loop_start = time.perf_counter()
            curr_action = leader.get_action()
            if args.real:
                follower.send_action(curr_action)
            if args.sim:
                sent_sim_action = joint_state_pos2rad(
                    pos_joint_state=curr_action,
                    calibration=SO101_NEW_CALIB,
                )
                act_socket.send(sent_sim_action.tobytes())
            dt_s = time.perf_counter() - loop_start
            precise_sleep(1 / args.fps - dt_s)
            loop_s = time.perf_counter() - loop_start
            print(f"\rTeleop loop time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)", end="", flush=True)
    
    except KeyboardInterrupt:
        pass

    finally:
        if leader.is_connected:
            leader.disconnect()
        if follower.is_connected:
            follower.disconnect()


if __name__ == "__main__":

    main()