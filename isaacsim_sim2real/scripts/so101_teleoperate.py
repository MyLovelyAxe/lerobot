"""
Run with:

$ python lerobot_teleoperate.py
"""

import time
import argparse
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
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="The frequency of read action from leader and send to follower, in unit of Hz.",
    )
    return parser.parse_args()


def main():

    args = parse_args()
    follower = SO101Follower(
        config=SO101FollowerConfig(
            port=SO101_FOLLOWER_PORT_ID, 
            id=SO101_FOLLOWER_NEW_CALIB,
        ),
    )
    leader = SO101Leader(
        config=SO101LeaderConfig(
            port=SO101_LEADER_PORT_ID, 
            id=SO101_LEADER_NEW_CALIB,
        ),
    )
    leader.connect()
    follower.connect()

    # TODO: add a warmup, firstly move the follower to the same pose with leader, smoothly

    try:
        while True:
            loop_start = time.perf_counter()
            action = leader.get_action()
            follower.send_action(action)
            dt_s = time.perf_counter() - loop_start
            precise_sleep(1 / args.fps - dt_s)
            loop_s = time.perf_counter() - loop_start
            print(f"\rTeleop loop time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        leader.disconnect()
        follower.disconnect()


if __name__ == "__main__":

    main()