"""Capture the exact interpreter, packages, upstream revision and hardware."""
import datetime
import json
import os
import platform
import site
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def capture(args):
    p = subprocess.run(args, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p.stdout.decode('utf-8', errors='replace').strip()


def main():
    import torch
    out = ROOT / 'results'
    out.mkdir(exist_ok=True)
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    info = {'captured_at_Asia_Shanghai': now, 'python': sys.version,
            'executable': sys.executable, 'platform': platform.platform(),
            'processor': platform.processor(), 'logical_cpu_count': os.cpu_count(),
            'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
            'user_site_enabled': site.ENABLE_USER_SITE,
            'cuda_available': torch.cuda.is_available(),
            'tfb_commit': capture(['git', '-C', 'vendor/TFB', 'rev-parse', 'HEAD']),
            'tfb_worktree_status': capture(['git', '-C', 'vendor/TFB', 'status', '--porcelain'])}
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        info.update(gpu=prop.name, gpu_memory_bytes=prop.total_memory,
                    compute_capability=list(torch.cuda.get_device_capability()))
    import numpy, pandas, scipy, sklearn, matplotlib, PIL
    info['module_provenance'] = {m.__name__: {'version': str(m.__version__), 'file': m.__file__}
                                 for m in [numpy, pandas, scipy, sklearn, matplotlib, PIL, torch]}
    frozen = capture([sys.executable, '-m', 'pip', 'freeze'])
    (out / 'requirements-lock.txt').write_text(frozen + '\n', encoding='utf-8')
    (out / 'environment.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    body = json.dumps(info, ensure_ascii=False, indent=2)
    body += '\n\n--- nvidia-smi ---\n' + capture(['nvidia-smi'])
    body += '\n\n--- pip check ---\n' + capture([sys.executable, '-m', 'pip', 'check'])
    body += '\n\n--- pip freeze ---\n' + frozen
    body += '\n\n--- PyTorch build ---\n' + torch.__config__.show()
    (out / 'environment.txt').write_text(body + '\n', encoding='utf-8')
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
