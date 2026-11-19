"""One resumable, isolated policy rollout. Invoke through thermal_guard.py."""
import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import random
import signal
import time

ROOT = Path(__file__).resolve().parents[2]
from metrics import measure, summarize
from geometry import bounds
os.environ['LIBERO_CONFIG_PATH'] = str(ROOT / 'libero_config')
os.environ['MUJOCO_GL'] = 'cgl'
os.environ['NUMBA_NUM_THREADS'] = '2'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['wine_bottle', 'black_bowl', 'milk'], required=True)
    parser.add_argument('--model', choices=['smolvla', 'xvla'], required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--steps', type=int, default=800)
    parser.add_argument('--action-steps', type=int)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--script-steps', type=int, default=40)
    parser.add_argument('--study', required=True)
    parser.add_argument('--mask', choices=['none','external','wrist'], default='none')
    parser.add_argument('--intervention', choices=['none','lift','sham','clearance'], default='none')
    args = parser.parse_args()
    if args.intervention != 'none' and (args.model != 'smolvla' or args.task != 'milk'):
        raise ValueError('Intervention defined only for relative-control SmolVLA milk')
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from lerobot.configs import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.envs.libero import LiberoEnv
    from lerobot.envs.utils import preprocess_observation
    from lerobot.policies import get_policy_class, make_pre_post_processors
    from libero.libero import benchmark

    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; do not silently switch to a slow CPU experiment')
    torch.mps.manual_seed(args.seed)
    device = torch.device('mps')
    model = (ROOT / 'offline/vla_course_offline/models/smolvla_libero' if args.model == 'smolvla'
             else ROOT / 'models/xvla-libero')
    task = json.loads((ROOT / 'experiment_config.json').read_text())['tasks'][args.task]
    slug = '_'.join(task['instruction'].lower().rstrip('.').split())
    metadata = json.loads((ROOT / f"tasks/scene_{task['scene']}/{slug}.json").read_text())
    action_label = str(args.action_steps) if args.action_steps else 'default'
    condition = args.intervention if args.intervention != 'none' else args.mask
    study_root = ROOT / 'phase2' / ('followup' if args.study == 'geometry_followup' else ('clearance_calibration' if args.study == 'clearance_calibration' else ''))
    folder = study_root / ('pilot' if args.pilot else 'results') / args.study / condition / args.model / args.task / f'chunk_{action_label}' / f'index_{args.index}' / f'seed_{args.seed}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'result.json').exists():
        previous = json.loads((folder / 'result.json').read_text())
        if previous.get('status') == 'completed':
            print('Already completed:', folder, flush=True)
            return
    cfg = PreTrainedConfig.from_pretrained(str(model))
    cfg.device = 'mps'
    cfg.pretrained_path = str(model)
    if hasattr(cfg, 'empty_cameras'):
        cfg.empty_cameras = 1
    if args.action_steps is not None:
        cfg.n_action_steps = args.action_steps
    camera_mapping = ({'agentview_image': 'camera1', 'robot0_eye_in_hand_image': 'camera2'}
                      if args.model == 'smolvla' else {'agentview_image': 'image', 'robot0_eye_in_hand_image': 'image2'})
    control = 'relative' if args.model == 'smolvla' else 'absolute'
    start = time.monotonic()
    print('Loading', args.model, flush=True)
    policy = get_policy_class(cfg.type).from_pretrained(str(model), config=cfg).to(device).eval()
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(model),
        preprocessor_overrides={'device_processor': {'device': 'mps'}})
    env_cfg = LiberoEnvConfig(task=metadata['base_suite'], task_ids=[metadata['base_task_id']],
        obs_type='pixels_agent_pos', camera_name_mapping=camera_mapping,
        observation_height=256, observation_width=256, control_mode=control)
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)
    suite = benchmark.get_benchmark_dict()[metadata['base_suite']]()
    env = LiberoEnv(task_suite=suite, task_id=metadata['base_task_id'],
        task_suite_name=metadata['base_suite'], episode_length=args.steps,
        camera_name=['agentview_image', 'robot0_eye_in_hand_image'],
        obs_type='pixels_agent_pos', render_mode='rgb_array',
        observation_height=256, observation_width=256,
        init_states=True, episode_index=args.index, n_envs=1,
        camera_name_mapping=camera_mapping, num_steps_wait=10, control_freq=20,
        control_mode=control, hard_reset=True)
    env._task_bddl_file = str(ROOT / metadata['bddl_file'])
    env.task_description = task['instruction']
    interrupted = []
    signal.signal(signal.SIGTERM, lambda signum, frame: interrupted.append(signum))
    rows = []
    actions = []
    states = []
    status = 'running'
    success = False
    writer = None
    result = {'task': args.task, 'instruction': task['instruction'], 'model': args.model,
        'seed': args.seed, 'index': args.index, 'max_steps': args.steps,
        'n_action_steps': cfg.n_action_steps, 'control_mode': control, 'pilot': args.pilot,
        'parameter_dtypes': dict(Counter(str(p.dtype) for p in policy.parameters())),
        'study':args.study,'mask':args.mask,'intervention':args.intervention,
        'intervention_trigger_step':None,'intervention_steps':0,'script_steps_planned':args.script_steps}
    def save():
        result.update(status=status, success=success if status == 'completed' else None,
                      steps=len(actions), elapsed_seconds=time.monotonic()-start)
        temp_result = folder / 'result.json.tmp'
        temp_result.write_text(json.dumps(result, indent=2))
        temp_result.replace(folder / 'result.json')
        np.savez_compressed(folder / 'trajectory.npz', actions=np.asarray(actions), states=np.asarray(states))
        (folder / 'timings.json').write_text(json.dumps(rows))
        if initial is not None:
            (folder/'stage_metrics.json').write_text(json.dumps({'initial':initial,'summary':summarize(rows,initial)},indent=2))
    initial = None
    contact_streak = 0
    intervention_remaining = 0
    lift_target = None
    def batched(value):
        if isinstance(value, dict):
            return {key: batched(item) for key, item in value.items()}
        if isinstance(value, np.ndarray):
            return value[None, ...]
        return value
    try:
        obs, _ = env.reset(seed=args.seed)
        policy.reset()
        initial = measure(env._env,args.task)
        result['initial_goal'] = bool(env._env.check_success())
        writer = imageio.get_writer(folder / 'rollout.mp4', fps=20, codec='libx264', quality=7)
        for step in range(args.steps):
            if interrupted:
                status = 'interrupted'
                break
            t0 = time.monotonic()
            scripted = intervention_remaining > 0
            if scripted:
                m = measure(env._env,args.task)
                if args.intervention == 'clearance':
                    bottom, _ = bounds(env._env,'milk_1')
                    _, rim = bounds(env._env,'basket_1')
                    dz = max(0,min(.25,(rim[2]+.02-bottom[2])/.05))
                else:
                    dz = max(0,min(.25,(lift_target-m['eef_xyz'][2])/.05)) if args.intervention == 'lift' else 0
                a = np.array([0,0,dz,0,0,0,1.],dtype=np.float32)
                intervention_remaining -= 1
                result['intervention_steps'] += 1
            else:
                observed = dict(obs)
                observed['pixels'] = dict(obs['pixels'])
                if args.mask != 'none':
                    source = 'agentview_image' if args.mask == 'external' else 'robot0_eye_in_hand_image'
                    key = camera_mapping[source]
                    observed['pixels'][key] = np.zeros_like(observed['pixels'][key])
                if step == 0 and args.mask != 'none':
                    for camera, pixels in observed['pixels'].items():
                        imageio.imwrite(folder/f'input_{camera}.png',np.asarray(pixels).astype(np.uint8))
                batch = preprocess_observation(batched(observed))
                batch['task'] = [task['instruction']]
                batch = env_pre(batch)
                batch = pre(batch)
                with torch.inference_mode():
                    action = post(policy.select_action(batch))
                action = env_post({'action': action})['action']
                a = action.detach().cpu().numpy().reshape(-1)
            if a.shape != (7,) or not np.isfinite(a).all():
                raise RuntimeError(f'Invalid simulator action: {a}')
            obs, _, terminated, truncated, _ = env.step(a)
            actions.append(a)
            states.append(env._env.sim.get_state().flatten())
            writer.append_data(env.render())
            success = bool(env._env.check_success())
            m = measure(env._env,args.task)
            if args.intervention == 'clearance' or args.study == 'clearance_calibration':
                bottom, _ = bounds(env._env,'milk_1')
                _, rim = bounds(env._env,'basket_1')
                m['carton_bottom_above_rim_m'] = float(bottom[2]-rim[2])
            rows.append({'step': step, 'seconds': time.monotonic()-t0, 'success': success, 'scripted':scripted, **m})
            if scripted and intervention_remaining == 0:
                policy.reset()
            if args.intervention != 'none' and result['intervention_trigger_step'] is None:
                contact_streak = contact_streak+1 if m['bilateral_contact'] else 0
                if contact_streak >= 3 and step < 700:
                    result['intervention_trigger_step'] = step+1
                    intervention_remaining = args.script_steps
                    lift_target = m['eef_xyz'][2]+.12
            if step % 25 == 0 or success:
                save()
                print('step', step, 'success', success, flush=True)
            if success or terminated or truncated:
                status = 'completed'
                break
        else:
            status = 'completed'
    except BaseException as exc:
        status = 'error'
        result['error'] = repr(exc)
        raise
    finally:
        if writer is not None:
            writer.close()
        env.close()
        save()
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
