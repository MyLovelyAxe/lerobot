import io
import time
import zmq
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Union, Any
from types import MethodType
import logging
logging.basicConfig(level=logging.INFO)

from scipy.interpolate import interp1d
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from lerobot.sim2real.constant import (
    HOME_MOVE_HZ,
    HOME_SPEED,
    HOME_TOL,
    JOINT_ORDER,
    SIMULATION_RANGE,
    SO101_NEW_CALIB,
)


def add_send_action_leader(
    robot: SO101Leader,
) -> SO101Leader:
    """Temporarily add a send_action() method to a so101 leader object."""

    def send_action(
        self: SO101Leader, 
        action: dict[str, Any],
    ) -> dict[str, Any]:
        """Send action to a leader arm."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}
        # Send goal position to the arm
        self.bus.sync_write("Goal_Position", goal_pos)
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    robot.send_action = MethodType(send_action, robot)
    return robot


def move_robot_to_target_pose(
    robot: Union[SO101Follower, SO101Leader], 
    target_pose: Dict[str, float],
    reverse_order: bool = False,
):
    """Move the real robot to given target pose by moving joints one by one.
    
    :param robot: the robot class of the hardware robot
    :param target_pose: the final pose the robot will reach
    :param reverse_order: False, move from 1st joint to the last joint, True, inverse
    """
    # if the robot is a leader, temporarily add a .send_action() to it
    if isinstance(robot, SO101Leader) and not hasattr(robot, 'send_action'):
        robot = add_send_action_leader(robot=robot)

    # decide the method to get the current state of robot
    if isinstance(robot, SO101Leader):
        curr_pose = {
            k: v
            for k, v in robot.get_action().items()
            if k.endswith(".pos")
        }
    elif isinstance(robot, SO101Follower):
        curr_pose = {
            k: v
            for k, v in robot.get_observation().items()
            if k.endswith(".pos")
        }

    joint_order = JOINT_ORDER.copy()
    if reverse_order:
        joint_order.reverse()

    dt = 1.0 / HOME_MOVE_HZ
    max_step = HOME_SPEED * dt

    for joint_name in joint_order:
        logging.info(f"Moving {joint_name} to home")

        while True:
            loop_start = time.perf_counter()

            diff = target_pose[joint_name] - curr_pose[joint_name]
            if abs(diff) <= HOME_TOL:
                curr_pose[joint_name] = target_pose[joint_name]
                robot.send_action(curr_pose)
                break

            step = max(-max_step, min(max_step, diff))
            curr_pose[joint_name] += step
            robot.send_action(curr_pose)

            precise_sleep(dt - (time.perf_counter() - loop_start))

        logging.info(f"joint {joint_name} returns to home")


def log_joint_state(
    joint_state: Dict[str, float],
    logging_label: str = "Current joint state",
):
    """Log the joint state of robot."""
    joint_state_list = list(f"{value:4f}" for key, value in joint_state.items() if key.endswith(".pos"))
    logging.info(f"{logging_label}: {', '.join(joint_state_list)}")


def rad2pos(
    rad: float, 
    joint_name: str, 
    calibration: str = SO101_NEW_CALIB,
):
    """Convert radians into motor pos.
    
    Isaac Sim joint state has real radian values, while SO100 robot hardware
    use -100 to 100 as "pos". Original script for SO100 follower, i.e. using_smolvla_example.py,
    use the default value of RobotConfig.use_degrees = False, so it use MotorNormMode.RANGE_M100_100,
    check norm_mode_body of class so100_foller.SO100Follower
    """

    sim_min = SIMULATION_RANGE[calibration][joint_name]['sim_min']
    sim_max = SIMULATION_RANGE[calibration][joint_name]['sim_max']

    if joint_name == 'gripper.pos':
        # norm = ((bounded_val - min_) / (max_ - min_)) * 100
        pos = ((rad - sim_min) / (sim_max - sim_min)) * 100
    else:
        # norm = (((bounded_val - min_) / (max_ - min_)) * 200) - 100
        pos = ((rad - sim_min) / (sim_max - sim_min)) * 200 - 100

    return pos


def pos2rad(
    pos: float, 
    joint_name: str, 
    calibration: str = SO101_NEW_CALIB,
):
    """Convert the noramlized motor pos into real radian."""

    sim_min = SIMULATION_RANGE[calibration][joint_name]['sim_min']
    sim_max = SIMULATION_RANGE[calibration][joint_name]['sim_max']

    if joint_name == 'gripper.pos':
        # unnormalized_values[id_] = int((bounded_val / 100) * (max_ - min_) + min_)
        rad = (pos / 100) * (sim_max - sim_min) + sim_min
    else:
        # unnormalized_values[id_] = int(((bounded_val + 100) / 200) * (max_ - min_) + min_)
        rad = (pos + 100) / 200 * (sim_max - sim_min) + sim_min
    
    return rad


def joint_state_pos2rad(
    pos_joint_state: Dict[str, float],
    calibration: str = SO101_NEW_CALIB,
) -> np.ndarray:
    """ Convert calibrated normalized joint state into radian for all joints.
    
    :param pos_joint_state: calibrated normalized joint state
    :param calibration: calibration method
    """
    rad_joint_state = np.array([
        pos2rad(pos=pos_joint_state[joint], joint_name=joint, calibration=calibration) for joint in JOINT_ORDER
    ],dtype=np.float32)

    return rad_joint_state

def joint_state_rad2pos(
    rad_joint_state: np.ndarray,
    calibration: str = SO101_NEW_CALIB,
) -> Dict[str, float]:
    """ Convert calibrated normalized joint state into radian for all joints.
    
    :param rad_joint_state: joint state values in unit of radian
    :param calibration: calibration method
    """
    rad_joint_state = {
        joint: float(rad2pos(rad=rad_joint_state[idx], joint_name=joint, calibration=calibration)) for idx, joint in enumerate(JOINT_ORDER)
    }

    return rad_joint_state


def generate_trajectory(
    q_start: np.ndarray, 
    q_target: np.ndarray, 
    T: float = 3.0, 
    dt: float = 0.02,
) -> List[Dict[str, float]]:
    """Generate a simple and smooth trajectory between 2 joint states.
    
    :param q_start: starting joint state
    :param q_target: target joint state
    :param T: the complete duration to execute the trajectory
    :param dt: temporal interval in the trajectory
    """
    
    q_start = np.array(q_start, dtype=float)
    q_target = np.array(q_target, dtype=float)

    trajectory: List[Dict[str, float]] = list()
    times = np.arange(0.0, T + dt, dt)

    # compute one smooth step for one timestamp
    smooth_step = lambda tau: 3 * tau**2 - 2 * tau**3

    for t in times:
        s = smooth_step(tau=t/T)
        q = q_start + s * (q_target - q_start)
        step = {
            "timestamp": t,
            "joint_state": q.tolist(),
        }
        trajectory.append(step)

    return trajectory

def generate_robot_actions_trajectory(
    start_state: Dict[str, float],
    target_state: Dict[str, float],
    T: float = 3.0, 
    dt: float = 0.02,
) -> List[Dict[str, float]]:
    """Generate trajectory in form of robot actions."""

    q_start = np.array([start_state[name] for name in JOINT_ORDER], dtype=np.float32)
    q_target = np.array([target_state[name] for name in JOINT_ORDER], dtype=np.float32)
    trajectory = generate_trajectory(
        q_start=q_start, 
        q_target=q_target, 
        T=T, 
        dt=dt,
    )
    action_trajectory: List[Dict[str, float]] = list()
    for step in trajectory:
        action_trajectory.append(
            dict(zip(JOINT_ORDER, step["joint_state"])),
        )
    return action_trajectory


def crop_by_time_percentage(
    times: np.ndarray, 
    percentage: Tuple[float, float] = (0.0, 1.0),
):
    """Crop all the timestamps according to desired percentage.
    
    :param times: all the raw timestamps
    :param percentage: how many timestamps are kept
    """
    times = np.array(sorted(times))
    t0 = times[0]
    t1 = times[-1]
    start_t = t0 + percentage[0] * (t1 - t0)
    end_t = t0 + percentage[1] * (t1 - t0)
    return times[(times >= start_t) & (times <= end_t)]


def compute_latency(
    reference_actions: Dict[str, Dict[str, float]],
    target_actions: Dict[str, Dict[str, float]],
    lag_range: Tuple[int, int] = (-1, 1),
    lag_step: float = 0.001,
    percentage: Tuple[float, float] = (0.0, 1.0),
) -> float:
    """Compute the latency [ms] between sending and executing actions.

    Both of reference_actions and target_actions should have this structure:
    - timestamp:
        - joint name: calibrated normalized state

    :param reference_actions: the time - action pairs when sent to robot
    :param target_actions: the time - action pairs when truly executed on robot
    :param lag_range: the range to search for the most aligned latency lagging
    :param lag_step: the resolution of researching space of latency lagging
    :param percentage: to crop the timestamps for both reference and target actions
    """

    # Sort timestamps
    ref_times_all = np.array(sorted(reference_actions.keys()))
    target_times_all = np.array(sorted(target_actions.keys()))

    # Crop each trajectory by its own time duration, not by shared index.
    t_ref = crop_by_time_percentage(times=ref_times_all, percentage=percentage)
    t_target = crop_by_time_percentage(times=target_times_all, percentage=percentage)
    q_ref = np.array([
        [reference_actions[t][j] for j in JOINT_ORDER]
        for t in t_ref
    ])
    q_target = np.array([
        [target_actions[t][j] for j in JOINT_ORDER]
        for t in t_target
    ])

    # Build interpolation functions
    interpolators = []
    for joint_idx in range(len(JOINT_ORDER)):
        interpolators.append(
            interp1d(
                t_target,
                q_target[:, joint_idx],
                kind="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
        )

    # Search lag
    lags = np.arange(
        lag_range[0],
        lag_range[1] + lag_step,
        lag_step
    )
    errors = []
    for lag in lags:
        shifted_time = t_ref + lag
        q_interp = np.column_stack([
            f(shifted_time)
            for f in interpolators
        ])
        valid_mask = ~np.isnan(q_interp).any(axis=1)
        if valid_mask.sum() < 10:
            errors.append(np.inf)
            continue
        error = np.mean(
            (
                q_ref[valid_mask]
                - q_interp[valid_mask]
            ) ** 2
        )
        errors.append(error)

    errors = np.array(errors)
    best_idx = np.argmin(errors)

    return lags[best_idx]


def get_rad_joint_state_from_socket(
    obs_socket: zmq.SyncSocket,
    timeout_ms: int = 10,
) -> Dict[str, float]:
    """Extract the joint state in radian from socket and convert to calibrated normalized format.
    
    :param obs_socket: the socket to read the current joint states of simulated robot
    """

    poller = zmq.Poller()
    poller.register(obs_socket, zmq.POLLIN)

    socks = dict(poller.poll(timeout_ms))

    if obs_socket not in socks:
        return None

    payload = obs_socket.recv()   # one npz blob
    buf = io.BytesIO(payload)
    data = np.load(buf)
    # NOTE: this is a list of joint state values, needs to convert
    joint_state_rad = data["joints"].astype(np.float32)
    joint_state = joint_state_rad2pos(
        rad_joint_state=joint_state_rad,
        calibration=SO101_NEW_CALIB,
    )
    return joint_state