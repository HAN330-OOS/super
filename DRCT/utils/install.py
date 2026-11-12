"""Install missing training utilities only. Never install/replace PyTorch."""
import importlib.util
import subprocess
import sys

PACKAGES = {
    'numpy': 'numpy',
    'einops': 'einops>=0.8',
    'PIL': 'pillow>=10',
    'cv2': 'opencv-python-headless',
    'skimage': 'scikit-image>=0.24',
}


def main():
    if not importlib.util.find_spec('torch'):
        raise SystemExit('PyTorch is missing. First select the project CPU/PPU PyTorch image. '
                         'This script intentionally does not install torch.')
    missing = [package for module, package in PACKAGES.items()
               if not importlib.util.find_spec(module)]
    if missing:
        print('Installing missing utilities with:', sys.executable, flush=True)
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])
    else:
        print('All utility modules already exist; no packages changed.')
    print('PyTorch was not included in the installation request.')


if __name__ == '__main__':
    main()
