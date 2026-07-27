# Camera+LiDAR 3D detection/tracking: CPU (FP32) and Axelera Metis hardware backends.
#
# Override device-connection defaults with env vars, e.g.:
#   make test-hardware HW_HOST=my-device HW_USER=ubuntu JUMP_HOST=
.DEFAULT_GOAL := help

VENV        := .venv
PYTHON      := $(VENV)/bin/python3
PIP         := $(VENV)/bin/pip

HW_HOST     ?= aquila-imx95-12594406
HW_USER     ?= torizon
JUMP_HOST   ?= spleenlab@172.25.36.3
REMOTE_DIR  ?= /home/$(HW_USER)/axelera_workspace/data/camera-lidar-axelera-deploy
SSH_JUMP    := $(if $(JUMP_HOST),-J $(JUMP_HOST),)

.PHONY: help setup-cpu test-cpu demo-cpu deploy test-hardware demo-hardware clean

help:
	@echo "CPU (FP32) backend -- runs right here, no hardware needed:"
	@echo "  make setup-cpu       create .venv and install requirements-cpu.txt"
	@echo "  make test-cpu        run geometry + CPU pipeline tests"
	@echo "  make demo-cpu        render demo_cpu.mp4 from the bundled KITTI clip"
	@echo ""
	@echo "Axelera Metis hardware backend -- needs the physical device:"
	@echo "  make deploy          copy this project to the device (see scripts/deploy_to_device.sh)"
	@echo "  make test-hardware   ssh in and run tests/test_hardware_pipeline.py on-device"
	@echo "  make demo-hardware   ssh in and render demo_hardware.mp4 on-device, then copy it back"
	@echo ""
	@echo "  make clean           remove caches/venv/generated demo videos"

setup-cpu:
	python3 -m venv $(VENV)
	$(PYTHON) -m ensurepip --upgrade
	$(PIP) install -r requirements-cpu.txt

test-cpu:
	$(PYTHON) -m pytest tests/test_geometry.py tests/test_cpu_pipeline.py -v

demo-cpu:
	$(PYTHON) scripts/demo.py --backend cpu --data-dir data/kitti_demo_clip --num-frames 50 --out demo_cpu.mp4

deploy:
	./scripts/deploy_to_device.sh

test-hardware:
	ssh $(SSH_JUMP) $(HW_USER)@$(HW_HOST) \
		'cd $(REMOTE_DIR) && python -m pytest tests/test_hardware_pipeline.py -v'

demo-hardware:
	ssh $(SSH_JUMP) $(HW_USER)@$(HW_HOST) \
		'cd $(REMOTE_DIR) && python scripts/demo.py --backend hardware --data-dir data/kitti_demo_clip --num-frames 50 --out demo_hardware.mp4'
	scp $(SSH_JUMP) $(HW_USER)@$(HW_HOST):$(REMOTE_DIR)/demo_hardware.mp4 .

clean:
	find . -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache demo_cpu.mp4 demo_hardware.mp4 demo_hw.mp4
