from typing import Dict, List
from pathlib import Path
import matplotlib.pyplot as plt
import time
from lerobot.sim2real.constant import (
    LOG_FOLDER,
    JOINT_ORDER,
)



def print_record(
    record: Dict[str, Dict[str, Dict[str, float]]],
    record_key: str = "send_sim",
    start_idx: int = 0,
    step: int = 10,
    times: int = 10,
):
    """Print out parts of recorded trajectory."""

    count = 0
    for idx, (timestamp, positions) in enumerate(record[record_key].items()):
        if idx < start_idx or idx % step != 0:
            continue
        if count > times:
            break
        count += 1
        print(f"{timestamp}: {positions}")


def plot_joint_lines_in_record(
    record: dict,
    which_record: List[str],
    title: str,
    store: bool = False,
    store_folder: Path = LOG_FOLDER,
):
    """
    Plot joint trajectories for selected records.

    Parameters
    ----------
    record : dict
        Dictionary containing:
        {
            record_name: {
                timestamp: {
                    joint_name: value
                }
            }
        }

    which_record : list[str]
        Names of records to plot, e.g.
        ["send_sim", "exec_sim", "exec_real"]
    """

    num_joints = len(JOINT_ORDER)
    ncols = 2
    nrows = (num_joints + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(20, 3 * nrows),
        sharex=True,
    )

    for joint_idx, joint_name in enumerate(JOINT_ORDER):

        ax = axes[joint_idx // ncols][joint_idx % ncols]

        for record_name in which_record:

            timestamps = sorted(record[record_name].keys())

            values = [
                record[record_name][t][joint_name]
                for t in timestamps
            ]

            ax.plot(
                timestamps,
                values,
                label=record_name,
            )

        ax.grid(True)
        ax.legend()

    for ax in axes[-1]:
        ax.set_xlabel("Timestamp")

    fig.suptitle(title)
    plt.tight_layout()
    if not store:
        plt.show()
    else:
        save_path = store_folder / f"{time.strftime('%Y%m%d_%H%M%S')}_{'_'.join(which_record)}.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Plot saved to {save_path}")
    plt.close(fig)


if __name__ == "__main__":

    record: Dict[str, Dict[str, Dict[str, float]]] = dict()
    print_record(
        record=record,
        record_key="send_sim",
        start_idx=100,
        step=100,
        times=5,
    )

    plot_joint_lines_in_record(
        record=record,
        which_record=["send_sim", "exec_sim", "exec_real"],
        store=True,
    )