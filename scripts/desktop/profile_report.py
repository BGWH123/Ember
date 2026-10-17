#!/usr/bin/env python3
"""Export local profiler evidence without changing sampling or host settings."""
import argparse
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--binary', type=Path, required=True, help='Exact unstripped executable used for collection')
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    binary = args.binary.resolve()
    root = Path(__file__).resolve().parents[2]
    pprof = root / 'artifacts/desktop/stress/tools/root/usr/bin/google-pprof'
    for profile in sorted(directory.glob('cpu.prof*')):
        if profile.suffix == '.txt':
            continue
        with profile.with_suffix(profile.suffix + '.txt').open('w') as output:
            subprocess.run([str(pprof), '--text', str(binary), str(profile)], stdout=output, stderr=subprocess.STDOUT, check=True)
    # Exiting snapshot has cumulative allocations and residual live bytes. Both are useful;
    # neither is process RSS, and fixture allocations are included when history > 0.
    heaps = sorted(directory.glob('*.heap'))
    if heaps:
        for metric in ['alloc_space', 'inuse_space']:
            with (directory / f'{metric}.txt').open('w') as output:
                subprocess.run([str(pprof), '--text', f'--{metric}', str(binary), str(heaps[-1])], stdout=output, stderr=subprocess.STDOUT, check=True)
    report = directory / 'timeline.nsys-rep'
    if report.exists():
        with (directory / 'osrt.csv').open('w') as output:
            subprocess.run(['nsys', 'stats', '--report', 'osrt_sum', '--format', 'csv', str(report)], stdout=output, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
