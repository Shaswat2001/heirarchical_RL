#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import numpy as np

Json = Union[Dict[str, Any], List[Any]]

def decode_action(action_dict):
    # Mapping from one-hot index to value for each axis
    mapping = [
        {"noop": 0, "forward": 1, "backward": -1},  # X
        {"noop": 0, "left": 1, "right": -1},        # Y
        {"noop": 0, "rotate-left": 1, "rotate-right": -1}  # Yaw
    ]
    
    action_names = ["X", "Y", "Yaw"]
    action_order = [
        ["noop", "forward", "backward"],
        ["noop", "left", "right"],
        ["noop", "rotate-left", "rotate-right"]
    ]
    
    result = []
    for axis_idx, axis_name in enumerate(action_names):
        one_hot = action_dict[axis_name][0]  # First row (assuming shape [1,3])
        idx = one_hot.index(1)  # Find active action
        action_str = action_order[axis_idx][idx]
        result.append(mapping[axis_idx][action_str])
    return result

def load_json(p: Path) -> Json:
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to read {p}: {e}")

def iter_state_action(records: Json) -> List[Tuple[Any, Any]]:
    if isinstance(records, list):

        data = []
        for step in records:
            action = decode_action(step["action"])
            data.append((step["state"], action))
        return data
    if isinstance(records, dict):
        if "state" in records and "action" in records:
            s, a = records["state"], records["action"]
        elif "states" in records and "actions" in records:

            s, a = records["states"], records["actions"]
        else:
            raise ValueError("JSON dict missing state/action keys")
        if len(s) != len(a):
            raise ValueError("Length mismatch between states and actions")
        return list(zip(s, a))
    raise ValueError("Unsupported JSON format")

def find_episodes(root: Path, filename: str = "0.json") -> List[Path]:
    return sorted(set(root.rglob(filename)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/Users/shaswatgarg/Downloads/staging_go2_datasets_july28", help="Root folder containing episode folders with 0.json")
    parser.add_argument("--outfile", type=str, default="dataset.npy", help="Output .npy path")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    out = Path(args.outfile).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    episodes = find_episodes(root, "0.json")
    if not episodes:
        print(f"No 0.json files found under {root}", file=sys.stderr)
        sys.exit(1)

    all_states, all_actions, all_success = [], [], []

    for ep_path in episodes:
        try:
            data = load_json(ep_path)
            pairs = iter_state_action(data)
            last_idx = len(pairs) - 1
            for t, (state, action) in enumerate(pairs):
                all_states.append(state[0])
                all_actions.append(action)
                all_success.append(t == last_idx)
        except Exception as e:
            print(f"[skip] {ep_path}: {e}", file=sys.stderr)
            continue

    dataset_dict = {
        "observations": np.array(all_states, dtype=object),
        "actions": np.array(all_actions, dtype=object),
        "terminals": np.array(all_success, dtype=bool)
    }

    print(dataset_dict["terminals"])

    np.save(out, dataset_dict)
    print(f"Saved dataset with {len(all_states)} steps to {out}")


if __name__ == "__main__":
    main()
