"""Run one experiment under macOS thermal supervision; stop on elevated/unknown state."""
import argparse
import ctypes
import datetime
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import time


def thermal_state():
    ctypes.CDLL('/System/Library/Frameworks/Foundation.framework/Foundation')
    objc = ctypes.CDLL('/usr/lib/libobjc.A.dylib')
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    msg = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(('objc_msgSend', objc))
    info = msg(objc.objc_getClass(b'NSProcessInfo'), objc.sel_registerName(b'processInfo'))
    integer_msg = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(('objc_msgSend', objc))
    state = integer_msg(info, objc.sel_registerName(b'thermalState'))
    if state not in range(4):
        raise RuntimeError(f'Unknown thermal state: {state}')
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default=str(Path(__file__).resolve().parents[1] / 'logs/thermal.jsonl'))
    parser.add_argument('--max-seconds', type=float, default=600)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    labels = ['nominal', 'fair', 'serious', 'critical']
    log = Path(args.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    def record(**data):
        data['time'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with log.open('a') as f:
            f.write(json.dumps(data) + '\n')
        print(json.dumps(data), flush=True)
    state = thermal_state()
    record(event='preflight', state=labels[state])
    if args.check:
        return 0 if state == 0 else 75
    command = args.command
    if command[:1] == ['--']:
        command = command[1:]
    if not command:
        parser.error('Provide a command after --, or use --check')
    if state != 0:
        record(event='refused_start', reason='elevated_thermal_state')
        return 75
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
    child = subprocess.Popen(command, env=env, start_new_session=True)
    start = time.monotonic()
    last_memory_check = -10.0
    def stop():
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    try:
        while child.poll() is None:
            try:
                state = thermal_state()
                record(event='sample', state=labels[state], elapsed=round(time.monotonic()-start, 2))
                if state != 0:
                    record(event='stop', reason='elevated_thermal_state')
                    stop()
                    return 75
                elapsed = time.monotonic() - start
                if elapsed - last_memory_check >= 10:
                    last_memory_check = elapsed
                    memory = subprocess.run(['/usr/bin/memory_pressure', '-Q'], capture_output=True,
                                            text=True, timeout=3, check=True)
                    match = re.search(r'System-wide memory free percentage:\s*(\d+)%', memory.stdout)
                    if match is None:
                        raise RuntimeError('Cannot read memory pressure')
                    free = int(match.group(1))
                    record(event='memory', free_percentage=free)
                    if free < 15:
                        record(event='stop', reason='memory_pressure')
                        stop()
                        return 77
            except Exception as exc:
                record(event='stop', reason='monitor_failed', error=str(exc))
                stop()
                return 76
            if time.monotonic()-start >= args.max_seconds:
                record(event='stop', reason='time_budget')
                stop()
                return 124
            time.sleep(2)
    finally:
        stop()
    record(event='exit', returncode=child.returncode)
    return child.returncode

if __name__ == '__main__':
    sys.exit(main())
