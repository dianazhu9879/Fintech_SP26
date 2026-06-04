import hashlib
from pathlib import Path
from datasets import Dataset, Audio

def hash_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

audio_dir = Path("./qa_clips")

seen_hashes = {}
seen_names = set()
unique_files = []
skipped = []

for f in sorted(audio_dir.glob("*.mp3")):
    fhash = hash_file(f)
    if fhash in seen_hashes or f.name in seen_names:
        skipped.append(f.name)
        continue
    seen_hashes[fhash] = f
    seen_names.add(f.name)
    unique_files.append(str(f))

print(f"Unique: {len(unique_files)}, Skipped duplicates: {len(skipped)}")
if skipped:
    print("Skipped:", skipped)

# Build and push dataset
ds = Dataset.from_dict({"audio": unique_files}).cast_column("audio", Audio())
ds.push_to_hub("TQTFintech/qa_data")