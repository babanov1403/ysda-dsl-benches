"""Machine-readable environment report; optional profilers are not prerequisites."""
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def command(args):
    try:
        result = subprocess.run(args, text=True, capture_output=True, timeout=15, cwd=ROOT)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def gpu_snapshot(uuid):
    return {
        'telemetry_csv': command(['nvidia-smi', f'--id={uuid}',
            '--query-gpu=uuid,name,driver_version,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,memory.used,utilization.gpu',
            '--format=csv']),
        'compute_processes_csv': command(['nvidia-smi', f'--id={uuid}',
            '--query-compute-apps=pid,used_memory', '--format=csv']),
    }


def report():
    packages = {}
    for name in ['torch', 'triton', 'nvidia-cutlass-dsl', 'cuda-bindings', 'numpy', 'cmake', 'ninja']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    available = torch.cuda.is_available()
    result = {
        'python': platform.python_version(), 'platform': platform.platform(),
        'packages': packages, 'torch_cuda': torch.version.cuda,
        'cuda_available': available, 'cuda_home': os.environ.get('CUDA_HOME'),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'nvcc': command(['nvcc', '--version']),
        'host_compiler': command(['c++', '--version']),
        'profilers': {name: shutil.which(name) for name in ['ncu', 'nsys', 'compute-sanitizer']},
        'git_commit': command(['git', 'rev-parse', 'HEAD']),
        'git_status': command(['git', 'status', '--porcelain']),
        'lock_sha256': hashlib.sha256((ROOT / 'uv.lock').read_bytes()).hexdigest() if (ROOT / 'uv.lock').exists() else None,
    }
    if available:
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        arch = f'{props.major}{props.minor}'
        uuid = str(props.uuid)
        if not uuid.startswith(('GPU-', 'MIG-')):
            uuid = 'GPU-' + uuid
        result['gpu'] = {
            'logical_index': device, 'name': props.name, 'uuid': uuid,
            'sm': arch, 'memory_bytes': props.total_memory,
            'multiprocessors': props.multi_processor_count,
        }
        result['cuda_library'] = str(ROOT / 'build' / f'sm_{arch}' / 'libysda_kernels.so')
        result['cuda_library_exists'] = Path(result['cuda_library']).is_file()
        result['telemetry'] = gpu_snapshot(result['gpu']['uuid'])
    return result


def main():
    result = report()
    print(json.dumps(result, indent=2))
    required_ok = result['cuda_available'] and result['nvcc'] and result.get('cuda_library_exists')
    if not required_ok or any(v is None for v in result['packages'].values()):
        raise SystemExit(1)
    if int(result['gpu']['sm']) < 80:
        raise SystemExit('This project baseline targets Ampere or newer (SM80+)')


if __name__ == '__main__':
    main()
