"""
Compile every model in models.py with Vela for Halo's Ethos-U55-128 and print
the estimated inference time, arena (DTCM) and weight (MRAM) footprint.

    pip install tensorflow-cpu ethos-u-vela
    python tools/halo_npu/models.py build
    python tools/halo_npu/estimate.py build
"""

import csv
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MACS_PER_CYCLE = 128
CLOCK_HZ = 160e6


def main(build_dir):
    out_dir = os.path.join(build_dir, "vela")
    for tflite in sorted(glob.glob(os.path.join(build_dir, "*.tflite"))):
        subprocess.run(
            [
                "vela", tflite,
                "--accelerator-config", "ethos-u55-128",
                "--config", os.path.join(HERE, "b1_vela.ini"),
                "--system-config", "B1_HE_DTCM_MRAM",
                "--memory-mode", "Shared_Sram",
                "--output-dir", out_dir,
            ],
            check=True,
            capture_output=True,
        )

    print(f"{'network':26s} {'MMAC':>6s} {'DTCM KiB':>9s} {'MRAM KiB':>9s} {'ms':>6s} {'fps':>5s} {'MAC util':>8s}")
    for summary in sorted(glob.glob(os.path.join(out_dir, "*_summary_*.csv"))):
        row = next(csv.DictReader(open(summary)))
        macs, seconds = float(row["nn_macs"]), float(row["inference_time"])
        utilisation = macs / (seconds * MACS_PER_CYCLE * CLOCK_HZ)
        print(
            f"{row['network']:26s} {macs / 1e6:6.1f} {float(row['sram_memory_used']):9.1f} "
            f"{float(row['off_chip_flash_memory_used']):9.1f} {seconds * 1e3:6.2f} "
            f"{1 / seconds:5.0f} {utilisation:8.0%}"
        )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "build")
