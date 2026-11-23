"""Generate point-tracking data over many houses with any number of resumable workers (RoboTrack-15M or your own).

    # the 300 houses of the released dataset, with their original seeds (statistically equivalent, not identical)
    python datagen/generate_robotrack15m.py prepare --root out/robotrack15m --reproduce
    # or N new houses (the released houses are skipped), optionally with other generate_house.py options
    python datagen/generate_robotrack15m.py prepare --root out/more --target-houses 1000 --seed 1 \
        --house-args "--num-points 2000 --resolution 640 480"
    python datagen/generate_robotrack15m.py worker --root out/robotrack15m --worker 0   # one per GPU slot; resumable
    python datagen/generate_robotrack15m.py status --root out/robotrack15m

Workers claim houses from a locked ledger (progress.json), so several can share a GPU or run on different machines over
a shared filesystem. A finished house lands in <root>/houses/house_<id>/; failed attempts stay in <root>/attempts/.
Each house's seed is (run seed + 100003 * house) mod (2^31 - 1), as in the released set (run seed 20260928).
Needs the official MolmoSpaces (the third_party/molmospaces submodule) installed; see the README.
Assets go to MLSPACES_CACHE_DIR and MLSPACES_ASSETS_DIR (MolmoSpaces' defaults if unset).
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import re
import shlex
import signal
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELEASED = json.loads((HERE / "robotrack15m_houses.json").read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    tmp = Path(path).with_name(Path(path).name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


@contextmanager
def ledger(root):
    # POSIX record locks (fcntl.lockf) work across machines on network filesystems; flock() may not.
    with (root / "progress.lock").open("a+") as lock:
        fcntl.lockf(lock, fcntl.LOCK_EX)
        try:
            state = json.loads((root / "progress.json").read_text())
            yield state
            write_json(root / "progress.json", state)
        finally:
            fcntl.lockf(lock, fcntl.LOCK_UN)


def scene_dir():
    from molmo_spaces import molmo_spaces_constants as c
    return Path(c.DATA_CACHE_DIR) / "scenes/procthor-10k-train/20251122_with_occupancy"


def prepare(a):
    if a.root.exists():
        raise FileExistsError(a.root)
    if a.reproduce:
        seed, houses, target = RELEASED["run_seed"], list(RELEASED["houses"]), len(RELEASED["houses"])
    else:
        scenes = scene_dir()
        available = sorted(int(re.fullmatch(r"train_(\d+)_ceiling.xml", p.name)[1])
                           for p in scenes.glob("train_*_ceiling.xml"))
        if not available:
            raise SystemExit(f"no ProcTHOR train scenes in {scenes}; run one house first so MolmoSpaces downloads them")
        houses = [h for h in available if h not in set(RELEASED["houses"])]
        random.Random(a.seed).shuffle(houses)        # failed houses are replaced by the next ones
        seed, target = a.seed, a.target_houses
        if target > len(houses):
            raise SystemExit(f"only {len(houses)} candidate houses")
    a.root.mkdir(parents=True)
    for d in ("houses", "attempts", "logs"):
        (a.root / d).mkdir()
    write_json(a.root / "run_config.json", dict(created_at=now(), reproduce=a.reproduce, seed=seed,
                                               target_houses=target, house_timeout_seconds=a.house_timeout,
                                               house_args=shlex.split(a.house_args)))
    write_json(a.root / "candidate_houses.json", houses)
    write_json(a.root / "progress.json", dict(next_candidate=0, claims={}))
    print(f"prepared {a.root}: {target} houses from {len(houses)} candidates, run seed {seed}")


def generate(root, cfg, house, seed):
    attempt = root / "attempts" / f"house_{house}_{uuid.uuid4().hex[:8]}"
    attempt.mkdir()
    env = dict(os.environ, MLSPACES_LICENSE_POLICY="commercial_safe", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
               PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    cmd = [sys.executable, str(HERE / "generate_house.py"), "--output-dir", str(attempt / "output"),
           "--house-id", str(house), "--seed", str(seed), *cfg.get("house_args", [])]
    with (attempt / "generation.log").open("w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            code = proc.wait(timeout=cfg["house_timeout_seconds"])
        except BaseException:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            raise
    if code:
        raise RuntimeError(f"house {house} failed ({code}); see {attempt}/generation.log")
    return attempt / "output"


def worker(a):
    root = a.root
    cfg = json.loads((root / "run_config.json").read_text())
    candidates = json.loads((root / "candidate_houses.json").read_text())
    target, failures = cfg["target_houses"], 0
    with (root / "logs" / f"worker_{a.worker:03d}.lock").open("a+") as lock:
        fcntl.lockf(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)        # one process per worker index
        while True:
            with ledger(root) as state:
                claims = state["claims"]
                done = sum(v["status"] == "complete" for v in claims.values())
                running = sum(v["status"] == "running" for v in claims.values())
                mine = [int(h) for h, v in claims.items() if v["worker"] == a.worker and v["status"] == "running"]
                if mine:
                    house = mine[0]                              # resume our own unfinished house
                elif done >= target:
                    print(f"worker {a.worker}: target reached ({done} houses)", flush=True)
                    return
                elif done + running >= target:
                    house = None                                 # enough in flight; wait in case one fails
                else:
                    if state["next_candidate"] >= len(candidates):
                        raise RuntimeError("candidate houses exhausted before the target")
                    house = candidates[state["next_candidate"]]
                    state["next_candidate"] += 1
                    claims[str(house)] = dict(worker=a.worker, status="running", started_at=now(),
                                              seed=(cfg["seed"] + 100003 * house) % (2**31 - 1))
                seed = claims[str(house)]["seed"] if house is not None else None
            if house is None:
                time.sleep(30)
                continue
            dest = root / "houses" / f"house_{house}"
            print(f"worker {a.worker}: {done}/{target} done, house {house} (seed {seed})", flush=True)
            try:
                if not dest.exists():
                    generate(root, cfg, house, seed).rename(dest)
                with ledger(root) as state:
                    state["claims"][str(house)].update(status="complete", completed_at=now())
                failures = 0
            except Exception as e:  # noqa: BLE001
                print(f"house {house}: {e}", flush=True)
                with ledger(root) as state:
                    state["claims"][str(house)].update(status="failed", failed_at=now(), error=repr(e))
                failures += 1
                if failures >= 10:                               # likely a broken GPU or setup: stop and look
                    raise RuntimeError("ten consecutive failed houses; stopping") from e


def status(a):
    cfg = json.loads((a.root / "run_config.json").read_text())
    claims = json.loads((a.root / "progress.json").read_text())["claims"]
    print(json.dumps(dict(target=cfg["target_houses"], **{k: sum(v["status"] == k for v in claims.values())
                                                          for k in ("complete", "running", "failed")}), indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("prepare", "worker", "status"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--reproduce", action="store_true", help="prepare: the 300 released houses and their seeds")
    p.add_argument("--target-houses", type=int, default=300)
    p.add_argument("--seed", type=int, default=20260928, help="prepare: run seed (house order and per-house seeds)")
    p.add_argument("--house-timeout", type=int, default=1800, help="prepare: seconds allowed per house")
    p.add_argument("--house-args", default="", help="prepare: extra generate_house.py options for every house, "
                   "e.g. \"--episodes 3 --num-points 2000\"")
    p.add_argument("--worker", type=int, default=0)
    a = p.parse_args()
    {"prepare": prepare, "worker": worker, "status": status}[a.command](a)


if __name__ == "__main__":
    main()
