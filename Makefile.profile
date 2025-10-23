# Profiling Makefile
# Quick commands for common profiling tasks
# Usage: make <target>

.PHONY: help test profile-3090 profile-jetson compare clean

# Default Python and data paths - override with environment variables
PYTHON ?= python3
SPLIT_DIR ?= /path/to/sav_val
WEIGHTS_DIR ?= .
NSYS ?= nsys

# Default profiling parameters
WARMUP ?= 2
RUNS ?= 5
VIDEOS ?= 1
OBJECTS ?= 1

help:
	@echo "Available targets:"
	@echo "  test              - Run quick test with synthetic data"
	@echo "  profile-3090      - Profile on RTX 3090 with BF16"
	@echo "  profile-jetson    - Profile on Jetson Orin with FP16"
	@echo "  compare-models    - Compare all SAM2 model sizes"
	@echo "  compare-prompts   - Compare point vs bbox prompts"
	@echo "  profile-edgetam   - Profile EdgeTAM model"
	@echo "  stats             - Show summary statistics"
	@echo "  analyze           - Run nsys stats on reports"
	@echo "  clean             - Remove profiling outputs"
	@echo ""
	@echo "Environment variables:"
	@echo "  SPLIT_DIR=$(SPLIT_DIR)"
	@echo "  WEIGHTS_DIR=$(WEIGHTS_DIR)"
	@echo "  WARMUP=$(WARMUP), RUNS=$(RUNS), VIDEOS=$(VIDEOS), OBJECTS=$(OBJECTS)"

test:
	@echo "Running test mode..."
	./run_nsight_profile.sh --test_mode

profile-3090:
	@echo "Profiling SAM2 base on RTX 3090..."
	./run_nsight_profile.sh \
		--split_dir $(SPLIT_DIR) \
		--weights_dir $(WEIGHTS_DIR) \
		--models sam2_base_points \
		--autocast bf16 \
		--warmup_runs $(WARMUP) \
		--profile_runs $(RUNS) \
		--limit_videos $(VIDEOS) \
		--limit_objects $(OBJECTS)

profile-jetson:
	@echo "Profiling SAM2 tiny on Jetson Orin..."
	NSYS_CMD=/opt/nvidia/nsight-systems/bin/nsys \
	./run_nsight_profile.sh \
		--split_dir $(SPLIT_DIR) \
		--weights_dir $(WEIGHTS_DIR) \
		--models sam2_tiny_points \
		--autocast fp16 \
		--warmup_runs $(WARMUP) \
		--profile_runs $(RUNS) \
		--limit_videos $(VIDEOS) \
		--limit_objects $(OBJECTS)

compare-models:
	@echo "Comparing SAM2 model sizes..."
	./run_nsight_profile.sh \
		--split_dir $(SPLIT_DIR) \
		--weights_dir $(WEIGHTS_DIR) \
		--models sam2_tiny_points,sam2_small_points,sam2_base_points \
		--warmup_runs $(WARMUP) \
		--profile_runs $(RUNS) \
		--limit_videos $(VIDEOS) \
		--limit_objects $(OBJECTS) \
		--out_dir ./comparison_models

compare-prompts:
	@echo "Comparing point vs bbox prompts..."
	./run_nsight_profile.sh \
		--split_dir $(SPLIT_DIR) \
		--weights_dir $(WEIGHTS_DIR) \
		--models sam2_base_points,sam2_base_bbox \
		--warmup_runs $(WARMUP) \
		--profile_runs $(RUNS) \
		--limit_videos $(VIDEOS) \
		--limit_objects $(OBJECTS) \
		--out_dir ./comparison_prompts

profile-edgetam:
	@echo "Profiling EdgeTAM..."
	./run_nsight_profile.sh \
		--split_dir $(SPLIT_DIR) \
		--weights_dir $(WEIGHTS_DIR) \
		--models edgetam_points \
		--warmup_runs $(WARMUP) \
		--profile_runs $(RUNS) \
		--limit_videos $(VIDEOS) \
		--limit_objects $(OBJECTS)

profile-compiled:
	@echo "Profiling with torch.compile..."
	./run_nsight_profile.sh \
		--split_dir $(SPLIT_DIR) \
		--weights_dir $(WEIGHTS_DIR) \
		--models sam2_base_points \
		--compile_models \
		--compile_mode max-autotune \
		--warmup_runs 3 \
		--profile_runs $(RUNS) \
		--limit_videos $(VIDEOS)

stats:
	@echo "=== Profile Summary Statistics ==="
	@if [ -f nsight_profiles/../profile_outputs/profile_stats.json ]; then \
		cat nsight_profiles/../profile_outputs/profile_stats.json | $(PYTHON) -m json.tool; \
	elif [ -f profile_outputs/profile_stats.json ]; then \
		cat profile_outputs/profile_stats.json | $(PYTHON) -m json.tool; \
	elif [ -f test_profile_outputs/profile_stats.json ]; then \
		cat test_profile_outputs/profile_stats.json | $(PYTHON) -m json.tool; \
	else \
		echo "No profile statistics found. Run a profiling target first."; \
	fi

analyze:
	@echo "=== Nsight Reports Analysis ==="
	@if [ -d nsight_profiles ] && [ -n "$$(ls -A nsight_profiles/*.nsys-rep 2>/dev/null)" ]; then \
		echo "Analyzing nsight_profiles/*.nsys-rep..."; \
		$(NSYS) stats nsight_profiles/*.nsys-rep; \
	else \
		echo "No nsight reports found. Run a profiling target first."; \
	fi

analyze-kernels:
	@echo "=== CUDA Kernel Summary ==="
	@if [ -d nsight_profiles ] && [ -n "$$(ls -A nsight_profiles/*.nsys-rep 2>/dev/null)" ]; then \
		$(NSYS) stats --report cuda_gpu_kern_sum nsight_profiles/*.nsys-rep; \
	else \
		echo "No nsight reports found."; \
	fi

analyze-memory:
	@echo "=== CUDA Memory Summary ==="
	@if [ -d nsight_profiles ] && [ -n "$$(ls -A nsight_profiles/*.nsys-rep 2>/dev/null)" ]; then \
		$(NSYS) stats --report cuda_mem_sum nsight_profiles/*.nsys-rep; \
	else \
		echo "No nsight reports found."; \
	fi

clean:
	@echo "Cleaning profiling outputs..."
	rm -rf nsight_profiles/
	rm -rf profile_outputs/
	rm -rf test_profile_outputs/
	rm -rf comparison_*/
	rm -rf test_synthetic_data/
	@echo "Done."

clean-all: clean
	@echo "Cleaning all generated data including overlays..."
	rm -rf test_outputs/
	@echo "Done."
