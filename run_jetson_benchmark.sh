#!/bin/bash
# Jetson benchmark helper script
# Provides common benchmark configurations for Jetson Orin platform

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BENCHMARK_SCRIPT="${SCRIPT_DIR}/jetson_benchmark.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${GREEN}======================================================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}======================================================================${NC}"
}

print_warning() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

print_error() {
    echo -e "${RED}ERROR: $1${NC}"
}

check_requirements() {
    if [ ! -f "$BENCHMARK_SCRIPT" ]; then
        print_error "Benchmark script not found: $BENCHMARK_SCRIPT"
        exit 1
    fi
    
    if ! command -v python3 &> /dev/null; then
        print_error "python3 not found"
        exit 1
    fi
}

show_help() {
    cat << EOF
Jetson Benchmark Helper Script

Usage: $0 <command> [options]

Commands:
    quick       Quick test with 1 video, low resolution
    standard    Standard benchmark (1024px, cuDNN tuning)
    memory-safe Memory-constrained mode (512px, no tuning)
    full        Full benchmark with all models
    setup-only  Only run setup phase (preprocessing)
    inference   Only run inference (requires existing setup)
    
Common Options:
    --split-dir <path>    Path to dataset (required)
    --weights-dir <path>  Path to model weights (default: .)
    --out-dir <path>      Output directory (default: auto)
    --models <list>       Comma-separated model list
    --limit-videos <n>    Limit number of videos (0=all)
    
Examples:
    # Quick test
    $0 quick --split-dir /data/sav_val
    
    # Standard benchmark
    $0 standard --split-dir /data/sav_val --out-dir ./results
    
    # Memory-constrained
    $0 memory-safe --split-dir /data/sav_val --limit-videos 5
    
    # Custom models
    $0 standard --split-dir /data/sav_val --models "edgetam_points,sam2_base_points"

EOF
}

# Parse common arguments
parse_common_args() {
    SPLIT_DIR=""
    WEIGHTS_DIR="."
    OUT_DIR=""
    MODELS=""
    LIMIT_VIDEOS=""
    EXTRA_ARGS=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --split-dir)
                SPLIT_DIR="$2"
                shift 2
                ;;
            --weights-dir)
                WEIGHTS_DIR="$2"
                shift 2
                ;;
            --out-dir)
                OUT_DIR="$2"
                shift 2
                ;;
            --models)
                MODELS="$2"
                shift 2
                ;;
            --limit-videos)
                LIMIT_VIDEOS="$2"
                shift 2
                ;;
            *)
                EXTRA_ARGS="$EXTRA_ARGS $1"
                shift
                ;;
        esac
    done
    
    if [ -z "$SPLIT_DIR" ]; then
        print_error "Missing required argument: --split-dir"
        echo ""
        show_help
        exit 1
    fi
    
    if [ ! -d "$SPLIT_DIR" ]; then
        print_error "Dataset directory not found: $SPLIT_DIR"
        exit 1
    fi
}

build_common_args() {
    ARGS="--split_dir $SPLIT_DIR --weights_dir $WEIGHTS_DIR"
    
    if [ -n "$OUT_DIR" ]; then
        ARGS="$ARGS --out_dir $OUT_DIR"
    fi
    
    if [ -n "$MODELS" ]; then
        ARGS="$ARGS --models $MODELS"
    fi
    
    if [ -n "$LIMIT_VIDEOS" ]; then
        ARGS="$ARGS --limit_videos $LIMIT_VIDEOS"
    fi
    
    ARGS="$ARGS $EXTRA_ARGS"
    echo "$ARGS"
}

cmd_quick() {
    print_header "Quick Test Mode"
    print_warning "This is a quick test with limited scope"
    echo ""
    
    parse_common_args "$@"
    
    BASE_ARGS=$(build_common_args)
    
    python3 "$BENCHMARK_SCRIPT" \
        $BASE_ARGS \
        --imgsz 384 \
        --limit_videos 1 \
        --limit_objects 1 \
        --models "edgetam_points" \
        --max_frames_in_mem 300
}

cmd_standard() {
    print_header "Standard Benchmark Mode"
    echo "Resolution: 1024px"
    echo "cuDNN: Auto-tuned"
    echo "Scope: All available videos"
    echo ""
    
    parse_common_args "$@"
    
    BASE_ARGS=$(build_common_args)
    
    python3 "$BENCHMARK_SCRIPT" \
        $BASE_ARGS \
        --imgsz 1024 \
        --enable_cudnn_tuning \
        --max_frames_in_mem 600
}

cmd_memory_safe() {
    print_header "Memory-Safe Mode"
    print_warning "Using conservative settings for memory-constrained systems"
    echo "Resolution: 512px"
    echo "cuDNN: Disabled"
    echo "Max frames in memory: 300"
    echo ""
    
    parse_common_args "$@"
    
    BASE_ARGS=$(build_common_args)
    
    python3 "$BENCHMARK_SCRIPT" \
        $BASE_ARGS \
        --imgsz 512 \
        --max_frames_in_mem 300 \
        --workspace_mb 128
}

cmd_full() {
    print_header "Full Benchmark Mode"
    echo "All models: SAM2 (tiny, small, base) + EdgeTAM"
    echo "All prompt types: points + bbox"
    echo "Resolution: 1024px"
    echo ""
    
    parse_common_args "$@"
    
    BASE_ARGS=$(build_common_args)
    
    FULL_MODELS="sam2_tiny_points,sam2_tiny_bbox,sam2_small_points,sam2_small_bbox,sam2_base_points,sam2_base_bbox,edgetam_points,edgetam_bbox"
    
    python3 "$BENCHMARK_SCRIPT" \
        $BASE_ARGS \
        --imgsz 1024 \
        --models "$FULL_MODELS" \
        --enable_cudnn_tuning \
        --max_frames_in_mem 600
}

cmd_setup_only() {
    print_header "Setup Phase Only"
    echo "Preprocessing data and testing configurations"
    echo "No inference will be run"
    echo ""
    
    parse_common_args "$@"
    
    BASE_ARGS=$(build_common_args)
    
    python3 "$BENCHMARK_SCRIPT" \
        $BASE_ARGS \
        --skip_inference \
        --enable_cudnn_tuning
}

cmd_inference() {
    print_header "Inference Phase Only"
    print_warning "Requires existing preprocessed data from setup phase"
    echo ""
    
    parse_common_args "$@"
    
    BASE_ARGS=$(build_common_args)
    
    python3 "$BENCHMARK_SCRIPT" \
        $BASE_ARGS \
        --skip_setup
}

# Main command router
main() {
    check_requirements
    
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi
    
    COMMAND=$1
    shift
    
    case $COMMAND in
        quick)
            cmd_quick "$@"
            ;;
        standard)
            cmd_standard "$@"
            ;;
        memory-safe)
            cmd_memory_safe "$@"
            ;;
        full)
            cmd_full "$@"
            ;;
        setup-only)
            cmd_setup_only "$@"
            ;;
        inference)
            cmd_inference "$@"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown command: $COMMAND"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

main "$@"
