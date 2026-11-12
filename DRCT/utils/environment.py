"""Read-only environment report; does not install packages or start training."""
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import sys


def main():
    result = {
        'python': sys.version,
        'executable': sys.executable,
        'platform': platform.platform(),
        'cpu_count': os.cpu_count(),
        'working_directory': os.getcwd(),
        'disk_free_gib': round(shutil.disk_usage(os.getcwd()).free / 1024**3, 2),
        'target_image': 'ppu-training:2.1.0-pytorch2.6.0-ppu-py312-cu128-ubuntu24.04',
        'packages': {},
        'note': 'An existing directory is not proof of persistent storage. Check the DSW mount configuration.',
    }
    for name in ['torch', 'numpy', 'einops', 'pillow', 'scikit-image', 'matplotlib',
                 'opencv-python', 'opencv-python-headless']:
        try:
            result['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result['packages'][name] = None
    if importlib.util.find_spec('torch'):
        try:
            import torch
            result['torch_runtime'] = {
                'version': torch.__version__,
                'cuda_build': torch.version.cuda,
                'cuda_available': torch.cuda.is_available(),
                'device_count': torch.cuda.device_count(),
                'devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                'distributed_available': torch.distributed.is_available(),
                'nccl_available': torch.distributed.is_nccl_available(),
                'gloo_available': torch.distributed.is_gloo_available(),
            }
            result['other_device_interfaces'] = {}
            for name in ('ppu', 'xpu'):
                backend = getattr(torch, name, None)
                if backend is None or not callable(getattr(backend, 'is_available', None)):
                    continue
                try:
                    available = bool(backend.is_available())
                    result['other_device_interfaces'][name] = {
                        'available': available,
                        'device_count': backend.device_count() if available and callable(getattr(backend, 'device_count', None)) else 0,
                    }
                except Exception as exc:
                    result['other_device_interfaces'][name] = {'probe_error_type': type(exc).__name__}
        except Exception as exc:
            result['torch_import_error'] = repr(exc)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
