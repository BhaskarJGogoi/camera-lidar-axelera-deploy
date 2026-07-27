#!/usr/bin/env bash
# Copies this project to the Axelera Metis device so the hardware backend
# can run there (it needs axelera.runtime, which only exists in the
# Voyager SDK's on-device Python environment).
#
# Prerequisite: the jump host (JUMP_HOST) must actually be reachable, which
# on this setup means a local docker container providing that network path
# needs to be running first -- start it before running this script.
#
# Usage:
#   ./scripts/deploy_to_device.sh
#   HW_HOST=my-other-device HW_USER=ubuntu JUMP_HOST= ./scripts/deploy_to_device.sh   # no jump host needed
#
# Override any of HW_HOST / HW_USER / JUMP_HOST / REMOTE_DIR via environment
# variables; defaults match this project's own validated test device.
set -euo pipefail

HW_HOST="${HW_HOST:-aquila-imx95-12594406}"
HW_USER="${HW_USER:-torizon}"
JUMP_HOST="${JUMP_HOST:-spleenlab@172.25.36.3}"
REMOTE_DIR="${REMOTE_DIR:-/home/${HW_USER}/axelera_workspace/data/camera-lidar-axelera-deploy}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SSH_OPTS=()
if [ -n "$JUMP_HOST" ]; then
    SSH_OPTS+=(-J "$JUMP_HOST")
fi

echo "Deploying to ${HW_USER}@${HW_HOST}:${REMOTE_DIR} (jump host: ${JUMP_HOST:-none})"
ssh "${SSH_OPTS[@]}" "${HW_USER}@${HW_HOST}" "mkdir -p ${REMOTE_DIR}"

# Exclude what the device doesn't need: CPU-only weights/compile assets
# (the device only needs backends/hardware/, not backends/cpu/ or the ONNX
# compile toolchain), the docs evidence video, and dev cruft.
rsync -avz -e "ssh ${SSH_OPTS[*]}" \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'backends/cpu' \
    --exclude 'backends/hardware/compile' \
    --exclude 'weights' \
    --exclude 'docs/evidence' \
    --exclude 'demo_*.mp4' \
    "$PROJECT_ROOT/" "${HW_USER}@${HW_HOST}:${REMOTE_DIR}/"

echo
echo "Deployed. Next steps (run these directly on the device, e.g. via:"
echo "  ssh ${SSH_OPTS[*]} ${HW_USER}@${HW_HOST}"
echo "):"
echo "  cd ${REMOTE_DIR}"
echo "  source /path/to/voyager-sdk/venv/bin/activate  # provides axelera.runtime"
echo "  python -m pytest tests/test_hardware_pipeline.py -v"
echo "  python scripts/demo.py --backend hardware --data-dir data/kitti_demo_clip --num-frames 50"
