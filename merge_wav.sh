#!/bin/bash
cat custom.wav.b64.* | base64 -d > custom.wav
rm -f custom.wav.b64.*
echo "custom.wav restored successfully"
