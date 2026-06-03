from typing import Dict, List, Tuple

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


if __name__ == "__main__":

    record: Dict[str, Dict[str, Dict[str, float]]] = dict()
    print_record(
        record=record,
        record_key="send_sim",
        start_idx=100,
        step=100,
        times=5,
    )