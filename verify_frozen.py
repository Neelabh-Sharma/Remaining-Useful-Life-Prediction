import hashlib, sys

all_ok = True
with open("FROZEN.sha256", "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        expected_hash = parts[0]
        filepath = parts[1].strip()
        try:
            with open(filepath, "rb") as target:
                actual_hash = hashlib.sha256(target.read()).hexdigest()
            if actual_hash == expected_hash:
                print(f"[OK] {filepath}")
            else:
                print(f"[MISMATCH] {filepath}")
                all_ok = False
        except Exception as e:
            print(f"[ERROR] {filepath}: {e}")
            all_ok = False

if all_ok:
    print("\nALL FROZEN BASELINE CHECKSUMS VERIFIED SUCCESSFULLY!")
else:
    print("\nCHECKSUM VERIFICATION FAILED!")
    sys.exit(1)
