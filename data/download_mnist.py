import urllib.request
import gzip
import os

BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images.bin": ("train-images-idx3-ubyte.gz", 16),
    "train_labels.bin": ("train-labels-idx1-ubyte.gz", 8),
    "test_images.bin":  ("t10k-images-idx3-ubyte.gz",  16),
    "test_labels.bin":  ("t10k-labels-idx1-ubyte.gz",  8)
}

data_dir = os.path.dirname(os.path.abspath(__file__))

print("--> Downloading and extracting MNIST binary files...")
for bin_name, (gz_name, offset) in FILES.items():
    out_path = os.path.join(data_dir, bin_name)
    if os.path.exists(out_path):
        print(f"  • {bin_name} already exists. Skipping.")
        continue
    
    url = BASE_URL + gz_name
    print(f"  • Fetching {gz_name}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp, gzip.GzipFile(fileobj=resp) as gz:
        gz.read(offset) # Skip IDX header
        raw_bytes = gz.read()
        with open(out_path, "wb") as f_out:
            f_out.write(raw_bytes)
    print(f"  ✓ Saved {bin_name} ({len(raw_bytes):,} bytes)")

print("✓ All dataset files ready in data/ directory!\n")
