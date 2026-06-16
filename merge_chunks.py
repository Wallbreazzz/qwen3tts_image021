import base64, glob, os
chunks = []
for f in sorted(glob.glob('custom.wav.b64.*')):
    with open(f) as fh:
        chunks.append(fh.read().replace(chr(10), '').replace(chr(13), ''))
combined = ''.join(chunks)
with open('custom.wav', 'wb') as fh:
    fh.write(base64.b64decode(combined))
print(f'Merged {len(chunks)} chunks into custom.wav ({os.path.getsize("custom.wav")} bytes)')
for f in glob.glob('custom.wav.b64.*'):
    os.remove(f)
for f in ['merge_wav.sh', os.path.join('.github','workflows','merge.yml')]:
    if os.path.exists(f):
        os.remove(f)
