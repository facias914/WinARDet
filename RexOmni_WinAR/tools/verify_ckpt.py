#!/usr/bin/env python3
"""
Simple checkpoint verification tool.
Verifies the MD5 checksums of two specific safetensors files.
"""

import hashlib
import os
import sys


def calculate_md5(file_path: str) -> str:
    """Calculate MD5 hash of a file."""
    md5_hash = hashlib.md5()

    try:
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                md5_hash.update(chunk)
        return md5_hash.hexdigest()
    except Exception as e:
        print(f"Error calculating MD5 for {file_path}: {e}")
        return ""


def verify_checkpoint(checkpoint_dir: str) -> bool:
    """Verify the two safetensors files have the expected MD5 checksums."""

    # Expected files and their MD5 checksums
    expected_files = {
        "model-00001-of-00002.safetensors": "3c292aee60ffb219cd394711766958c4",
        "model-00002-of-00002.safetensors": "c5d8181de13cc9c7148480485ec5c8d7",
    }

    print(f"Verifying checkpoint: {checkpoint_dir}")
    print()

    all_passed = True

    for filename, expected_md5 in expected_files.items():
        file_path = os.path.join(checkpoint_dir, filename)

        if not os.path.exists(file_path):
            print(f"❌ MISSING: {filename}")
            all_passed = False
            continue

        print(f"Checking: {filename}")
        file_size = os.path.getsize(file_path)
        print(f"  File size: {file_size / (1024**3):.2f} GB")

        actual_md5 = calculate_md5(file_path)

        if actual_md5 == expected_md5:
            print(f"✅ PASS: MD5 matches")
            print(f"  MD5: {actual_md5}")
        else:
            print(f"❌ FAIL: MD5 mismatch")
            print(f"  Expected: {expected_md5}")
            print(f"  Actual:   {actual_md5}")
            all_passed = False
        print()

    if all_passed:
        print("🎉 All files passed verification!")
    else:
        print("⚠️  Verification failed!")

    return all_passed


def main():
    if len(sys.argv) != 2:
        print("Usage: python verify_ckpt.py <checkpoint_dir>")
        sys.exit(1)

    checkpoint_dir = sys.argv[1]

    if not os.path.exists(checkpoint_dir):
        print(f"Error: Checkpoint directory '{checkpoint_dir}' does not exist!")
        sys.exit(1)

    success = verify_checkpoint(checkpoint_dir)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
