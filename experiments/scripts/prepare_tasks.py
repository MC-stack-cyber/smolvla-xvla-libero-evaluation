"""Configure project-local LIBERO paths for the bundled task definitions."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / 'vendor/LIBERO/libero/libero'
CONFIG = ROOT / 'libero_config'
CONFIG.mkdir(exist_ok=True)
os.environ['LIBERO_CONFIG_PATH'] = str(CONFIG)
paths = {
    'benchmark_root': str(REPO),
    'bddl_files': str(REPO / 'bddl_files'),
    'init_states': str(REPO / 'init_files'),
    'assets': str(ROOT / 'offline/vla_course_offline/libero_assets'),
    'datasets': str(ROOT / 'datasets'),
}
(ROOT / 'datasets').mkdir(exist_ok=True)
# JSON is valid YAML, avoiding an extra dependency during configuration.
(CONFIG / 'config.yaml').write_text(json.dumps(paths, indent=2))
