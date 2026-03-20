#!/usr/bin/env python3
"""Generate GitHub Actions build matrix for vLLM images.

Usage: python build-matrix.py YY.MM
Output: JSON matrix {"include": [{vllm_version, cuda_version, cudnn_version, pytorch_version}]}
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'cuda-mapping.conf')


# === Config parsing ===

def tag_to_int(tag: str) -> int:
  """Convert YY.MM to comparable int (e.g., '26.03' -> 2603)."""
  return int(tag.replace('.', '').ljust(4, '0')[:4])


def load_cuda_mapping() -> list:
  """Load threshold -> cuda_major from config. Returns [(threshold_int, cuda_major), ...] sorted desc."""
  mappings = [(0, "12")]  # Default fallback
  try:
    with open(CONFIG_PATH) as f:
      for line in f:
        line = line.split('#')[0].strip()  # Remove comments
        if ':' in line:
          threshold, cuda_major = line.split(':', 1)
          mappings.append((tag_to_int(threshold.strip()), cuda_major.strip()))
  except FileNotFoundError:
    pass
  return sorted(mappings, key=lambda x: x[0], reverse=True)


def get_cuda_major(month: str) -> str:
  """Get CUDA major version for month using threshold logic (highest threshold <= month)."""
  month_int = tag_to_int(month)
  for threshold, cuda_major in load_cuda_mapping():
    if threshold <= month_int:
      return cuda_major
  return "12"


# === Conda package search ===

def parse_ver(v: str) -> tuple:
  """Parse version string to tuple for comparison."""
  return tuple(int(x) for x in re.findall(r'\d+', v))


def conda_search(pkg: str, channel: str) -> list:
  """Search conda package in channel, return list of package info."""
  try:
    r = subprocess.run(
      ['conda', 'search', '-c', channel, pkg, '--json'],
      capture_output=True, text=True, timeout=120
    )
    return json.loads(r.stdout).get(pkg, []) if r.returncode == 0 else []
  except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
    return []


def get_cuda_versions() -> dict:
  """Get available CUDA versions from nvidia channel. Returns {major: highest_minor}."""
  packages = conda_search('cuda-toolkit', 'nvidia') or conda_search('cuda', 'nvidia')
  majors = {}
  for p in packages:
    v = p.get('version', '')
    if '.' in v:
      parts = v.split('.')
      majors.setdefault(parts[0], []).append(f"{parts[0]}.{parts[1]}")
  return {m: max(vs, key=parse_ver) for m, vs in majors.items()}


def get_cudnn(cuda_major: str) -> str:
  """Get latest cuDNN version compatible with given CUDA major version."""
  pkgs = conda_search('cudnn', 'nvidia') or conda_search('cudnn', 'nvidia/label/cudnn')

  # Filter packages by CUDA compatibility from dependencies
  compatible_versions = []
  for p in pkgs:
    deps = p.get('depends', [])
    version = p.get('version', '')
    if not version:
      continue

    # Check if this package is compatible with our CUDA version
    for dep in deps or []:
      if 'cuda-version' in dep:
        # Parse cuda-version constraint like ">=12,<13.0a0" or ">=13,<14.0a0"
        if f'>={cuda_major},' in dep or f'>={cuda_major}<' in dep or f'>={cuda_major} ' in dep:
          compatible_versions.append(version)
          break
        elif f'>={cuda_major},' in str(dep) or (f'>={cuda_major}' in str(dep) and f'<{int(cuda_major)+1}' in str(dep)):
          compatible_versions.append(version)
          break

  return max(compatible_versions, key=parse_ver, default='N/A') if compatible_versions else 'N/A'


def get_pytorch(min_ver: str) -> str:
  """Get PyTorch version >= min_ver from pytorch channel."""
  if not min_ver:
    return 'N/A'
  versions = sorted(
    [p.get('version', '') for p in conda_search('pytorch', 'pytorch')],
    key=parse_ver, reverse=True
  )
  for v in versions:
    if parse_ver(v) >= parse_ver(min_ver):
      return v
  return versions[0] if versions else min_ver


# === PyPI vLLM search ===

def retry_request(func, max_retries=3, delay=2):
  """Retry a function with exponential backoff."""
  last_error = None
  for attempt in range(max_retries):
    try:
      return func()
    except Exception as e:
      last_error = e
      if attempt < max_retries - 1:
        wait_time = delay * (2 ** attempt)
        print(f"Retry {attempt + 1}/{max_retries} after {wait_time}s due to: {e}", file=sys.stderr)
        time.sleep(wait_time)
  raise last_error


def get_vllm_info(year: int, month: int) -> dict:
  """Get highest vLLM version released in given month from PyPI."""
  try:
    import requests

    def fetch_vllm_releases():
      r = requests.get("https://pypi.org/pypi/vllm/json", timeout=30)
      if r.status_code != 200:
        raise Exception(f"PyPI returned status {r.status_code}")
      return r.json()

    data = retry_request(fetch_vllm_releases)

    # Filter by release month
    releases = data.get('releases', {})
    candidates = []
    for ver, files in releases.items():
      if not files or not files[0].get('upload_time'):
        continue
      dt = datetime.fromisoformat(files[0]['upload_time'].replace('Z', '+00:00'))
      if dt.year == year and dt.month == month:
        candidates.append(ver)

    if not candidates:
      return None

    vllm_ver = max(candidates, key=parse_ver)

    # Get PyTorch requirement
    pytorch = None

    def fetch_vllm_version():
      r = requests.get(f"https://pypi.org/pypi/vllm/{vllm_ver}/json", timeout=30)
      if r.status_code != 200:
        raise Exception(f"PyPI returned status {r.status_code}")
      return r.json()

    vllm_data = retry_request(fetch_vllm_version)
    for req in vllm_data.get('info', {}).get('requires_dist', []) or []:
      m = re.search(r'torch[=<>!~]+(\d+\.\d+)', req)
      if m:
        pytorch = m.group(1)
        break

    return {'version': vllm_ver, 'pytorch_version': pytorch}

  except Exception as e:
    print(f"Error fetching vLLM: {e}", file=sys.stderr)
    return None


# === Main ===

def main():
  if len(sys.argv) != 2:
    print("Usage: python build-matrix.py YY.MM", file=sys.stderr)
    sys.exit(1)

  month_str = sys.argv[1]

  # Parse month input
  try:
    yy, mm = map(int, month_str.split('.'))
    year, month = 2000 + yy, mm
    if not 1 <= month <= 12:
      raise ValueError
  except ValueError:
    print(f"Error: Invalid format '{month_str}'. Use YY.MM (e.g., 26.03)", file=sys.stderr)
    sys.exit(1)

  # Get vLLM version
  vllm_info = get_vllm_info(year, month)
  if not vllm_info:
    print(json.dumps({'error': f'No vLLM versions found in {year}-{month:02d}'}))
    sys.exit(1)

  # Get CUDA version based on config threshold
  cuda_major = get_cuda_major(month_str)
  cuda_versions = get_cuda_versions()

  if not cuda_versions:
    print(json.dumps({'error': 'No CUDA versions found from nvidia channel'}))
    sys.exit(1)

  cuda_ver = cuda_versions.get(cuda_major) or max(cuda_versions.values(), key=parse_ver)

  # Get cuDNN (compatible with CUDA major) and PyTorch
  cudnn_full = get_cudnn(cuda_major)
  pytorch_full = get_pytorch(vllm_info['pytorch_version'])

  # Output matrix - add 'latest' tag only for current month
  current_month = datetime.now().strftime('%y.%m')
  tag = f"{month_str},latest" if month_str == current_month else month_str

  item = {
    'tag': tag,
    'vllm_version': vllm_info['version'],
    'cuda_version': cuda_ver,
    'cudnn_version': '.'.join(cudnn_full.split('.')[:2]) if cudnn_full != 'N/A' else 'N/A',
    'pytorch_version': '.'.join(pytorch_full.split('.')[:2]) if pytorch_full != 'N/A' else 'N/A'
  }
  print(json.dumps({'include': [item]}))


if __name__ == '__main__':
  main()
