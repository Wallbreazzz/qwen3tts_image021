import glob, os
chunks = []
for f in sorted(glob.glob("custom.wav.b64.*")):
    with open(f, "rb") as fh:
        chunks.append(fh.read())
combined = b"".join(chunks)
with open("custom.wav", "wb") as fh:
    fh.write(combined)
print(f"Merged {len(chunks)} chunks, {len(combined)} bytes")
for f in glob.glob("custom.wav.b64.*"):
    os.remove(f)
for f in ["merge_wav.sh", "merge_chunks.py", os.path.join(".github","workflows","merge.yml")]:
    if os.path.exists(f):
        os.remove(f)
