"""Sequential rollouts with cooldowns, bounded retries, and resumable summaries."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from thermal_guard import thermal_state

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--model', choices=['smolvla', 'xvla', 'all'], default='all')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    lock = (ROOT / '.batch.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another batch is running; do not launch concurrent model jobs')
    config = json.loads((ROOT / 'experiment_config.json').read_text())
    queue = []
    models = ['smolvla', 'xvla'] if args.model == 'all' else [args.model]
    for model in models:
        chunks = [None] if args.pilot or model == 'xvla' else [None] + config['ablation']['n_action_steps']
        for chunk in chunks:
            for task in config['tasks']:
                for seed in ([0] if args.pilot else config['seeds']):
                    queue.append((model, chunk, task, seed))
    if args.limit:
        queue = queue[:args.limit]
    log = ROOT / 'logs/batch.jsonl'
    log.parent.mkdir(exist_ok=True)
    interrupted = []
    signal.signal(signal.SIGTERM, lambda s, f: interrupted.append(s))
    def record(**data):
        data['time'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with log.open('a') as f:
            f.write(json.dumps(data) + '\n')
        print(data, flush=True)
    for model, chunk, task, seed in queue:
        if interrupted or (ROOT / 'STOP').exists():
            record(event='stopped')
            return 130
        chunk_name = 'default' if chunk is None else str(chunk)
        result_path = (ROOT / ('pilot' if args.pilot else 'results') / model / task /
                       f'chunk_{chunk_name}' / f"index_{config['init_state_index']}" /
                       f'seed_{seed}' / 'result.json')
        if result_path.exists() and json.loads(result_path.read_text()).get('status') == 'completed':
            record(event='skip_completed', model=model, task=task, seed=seed, chunk=chunk)
            continue
        # Require 30 consecutive seconds nominal; do not automatically run while warm.
        normal_since = None
        wait_start = time.monotonic()
        while True:
            if interrupted or (ROOT / 'STOP').exists():
                return 130
            state = thermal_state()
            if state == 0:
                normal_since = normal_since or time.monotonic()
                if time.monotonic()-normal_since >= 30:
                    break
            else:
                normal_since = None
            if time.monotonic()-wait_start > 1800:
                record(event='cooldown_timeout')
                return 75
            time.sleep(2)
        command = [sys.executable, str(ROOT / 'scripts/thermal_guard.py'),
                   '--max-seconds', '600' if args.pilot else '1800', '--',
                   sys.executable, '-u', str(ROOT / 'scripts/run_episode.py'),
                   '--task', task, '--model', model, '--seed', str(seed),
                   '--index', str(config['init_state_index']),
                   '--steps', '5' if args.pilot else str(config['max_steps'])]
        if args.pilot:
            command.append('--pilot')
        if chunk is not None:
            command += ['--action-steps', str(chunk)]
        record(event='start', model=model, task=task, seed=seed, chunk=chunk)
        path = ROOT / 'logs' / f'{model}_{task}_{seed}_chunk_{chunk}_{"pilot" if args.pilot else "formal"}.log'
        with path.open('a') as output:
            child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
            try:
                while child.poll() is None:
                    if interrupted or (ROOT / 'STOP').exists():
                        child.send_signal(signal.SIGINT)
                    time.sleep(2)
            except BaseException:
                child.send_signal(signal.SIGINT)
                child.wait(timeout=20)
                raise
        record(event='exit', returncode=child.returncode, model=model, task=task, seed=seed)
        subprocess.run([sys.executable, str(ROOT / 'scripts/summarize.py')], check=True)
        if child.returncode != 0:
            # Stop for review; an error/thermal stop is never a benchmark failure.
            return child.returncode
    record(event='queue_completed', episodes=len(queue))
    return 0


if __name__ == '__main__':
    sys.exit(main())
