#!/usr/bin/env bash
set -euo pipefail

revision=9dc1a8fdcf9b5676f87c2cdccac021988f6ff5af
base="https://huggingface.co/datasets/mteb/LongMemEval/resolve/$revision"
destination=${1:-data/longmemeval}
mkdir -p "$destination"

curl -fL --continue-at - \
  -o "$destination/corpus.parquet" \
  "$base/knowledge_update-corpus/test-00000-of-00001.parquet"
printf '%s  %s\n' \
  3dfd1c8fb301fb3663416404c74fc378c90456695a7cf2f1c2723172d5b952b2 \
  "$destination/corpus.parquet" | sha256sum --check

for task in \
  knowledge_update multi_session single_session_assistant \
  single_session_preference single_session_user temporal_reasoning
do
  for kind in queries qrels; do
    curl -fsSL \
      -o "$destination/$task-$kind.parquet" \
      "$base/$task-$kind/test-00000-of-00001.parquet"
  done
done
curl -fsSL -o "$destination/README.md" "$base/README.md"
printf '%s\n' "$revision" > "$destination/REVISION"
sha256sum "$destination"/*.parquet > "$destination/SHA256SUMS"
