#!/bin/bash

# Test script for the unified training script
# This script tests all major functionality of train.py

set -e  # Exit on any error

echo "🧪 Running tests for unified training script..."
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Function to run a test
run_test() {
    local test_name="$1"
    local test_command="$2"
    local expected_pattern="$3"

    echo -e "${BLUE}Testing: ${test_name}${NC}"

    if eval "$test_command" 2>&1 | grep -q "$expected_pattern"; then
        echo -e "  ${GREEN}✓ PASS${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "  ${RED}✗ FAIL${NC}"
        echo "  Command: $test_command"
        echo "  Expected pattern: $expected_pattern"
        echo "  Output:"
        eval "$test_command" 2>&1 | head -10
        ((TESTS_FAILED++))
    fi
    echo
}

# Function to run a test that should fail
run_test_fail() {
    local test_name="$1"
    local test_command="$2"
    local expected_pattern="$3"

    echo -e "${BLUE}Testing (should fail): ${test_name}${NC}"

    if eval "$test_command" 2>&1 | grep -q "$expected_pattern"; then
        echo -e "  ${GREEN}✓ PASS (failed as expected)${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "  ${RED}✗ FAIL (should have failed)${NC}"
        echo "  Command: $test_command"
        echo "  Expected pattern: $expected_pattern"
        echo "  Output:"
        eval "$test_command" 2>&1 | head -10
        ((TESTS_FAILED++))
    fi
    echo
}

# Check if train.py exists
if [ ! -f "train.py" ]; then
    echo -e "${RED}Error: train.py not found in current directory${NC}"
    exit 1
fi

# No need to change directory since we're already in the right place
# cd ..

echo -e "${YELLOW}1. Testing basic help and argument parsing${NC}"
run_test "Help message" "python train.py --help" "usage: train.py"
run_test "No arguments (should fail)" "python train.py" "error: the following arguments are required"
run_test "Invalid preset" "python train.py --preset nonexistent_preset test_run" "Preset 'nonexistent_preset' not found"

echo -e "${YELLOW}2. Testing configuration file loading${NC}"
run_test "Load default config" "python train.py --preset default test_run" "time accelerate launch"
run_test "Load models config" "python train.py --preset flux_adafactor_simple test_run" "flux_train_network.py"
run_test "Load characters config" "python train.py --preset character_simple test_run" "time accelerate launch"
run_test "Load effects config" "python train.py --preset effect_simple test_run" "time accelerate launch"

echo -e "${YELLOW}3. Testing model family specific commands${NC}"
run_test "Flux model family" "python train.py --preset flux_adafactor_simple test_run" "flux_train_network.py"
run_test "SDXL model family" "python train.py --preset sdxl_adafactor_simple test_run" "sdxl_train_network.py"
run_test "Pony model family" "python train.py --preset pony_adafactor_simple test_run" "sdxl_train_network.py"

echo -e "${YELLOW}4. Testing optimizer configurations${NC}"
run_test "Adafactor optimizer" "python train.py --preset flux_adafactor_simple test_run" "adafactor"
run_test "Prodigy optimizer" "python train.py --preset flux_prodigy_simple test_run" "prodigy"
run_test "Ranger optimizer" "python train.py --preset flux_ranger_simple test_run" "ranger"

echo -e "${YELLOW}5. Testing training duration options${NC}"
run_test "Total images duration" "python train.py --preset flux_total_images_example test_run" "Calculated.*steps from.*total images"
run_test "Max steps override" "python train.py --preset flux_adafactor_simple --max-steps 500 test_run" "--max_train_steps 500"
run_test "Max epochs override" "python train.py --preset flux_adafactor_simple --max-epochs 10 test_run" "--max_train_epochs 10"
run_test "Batch size override" "python train.py --preset flux_adafactor_simple --batch-size 4 test_run" "--train_batch_size 4"

echo -e "${YELLOW}6. Testing VRAM and network options${NC}"
run_test "High VRAM mode" "python train.py --preset flux_adafactor_simple --vram-mode highvram test_run" "--highvram"
run_test "Low VRAM mode" "python train.py --preset flux_adafactor_simple --vram-mode lowvram test_run" "--lowvram"
run_test "Train UNet only" "python train.py --preset sdxl_adafactor_simple --network-train-unet-only test_run" "--network_train_unet_only"

echo -e "${YELLOW}7. Testing LyCORIS support${NC}"
run_test "LyCORIS LoHa" "python train.py --preset flux_loha_simple --lycoris-subtype loha test_run" "lycoris.kohya"
run_test "LyCORIS LoKR" "python train.py --preset flux_lokr_simple --lycoris-subtype lokr test_run" "lycoris.kohya"

echo -e "${YELLOW}8. Testing resume functionality${NC}"
run_test "Resume from weights" "python train.py --preset flux_adafactor_simple --resume-from test_weights.safetensors test_run" "--network_weights"

echo -e "${YELLOW}9. Testing model family overrides${NC}"
run_test "Override to Flux" "python train.py --preset sdxl_adafactor_simple --model-family flux test_run" "flux_train_network.py"
run_test "Override to SDXL" "python train.py --preset flux_adafactor_simple --model-family sdxl test_run" "sdxl_train_network.py"
run_test "Override to Pony" "python train.py --preset flux_adafactor_simple --model-family pony test_run" "sdxl_train_network.py"

echo -e "${YELLOW}10. Testing preset inheritance${NC}"
run_test "Inheritance from default" "python train.py --preset flux_adafactor_simple test_run" "cosine"
run_test "Complex preset inheritance" "python train.py --preset flux_prodigy_complex test_run" "prodigy"

echo -e "${YELLOW}11. Testing environment variable handling${NC}"
run_test "Custom REMOTE_ROOT" "REMOTE_ROOT=/custom/path python train.py --preset flux_adafactor_simple test_run" "/custom/path"
run_test "Custom FLUX_PATH" "FLUX_PATH=/custom/flux python train.py --preset flux_adafactor_simple test_run" "/custom/flux"
run_test "Custom SDXL_PATH" "SDXL_PATH=/custom/sdxl python train.py --preset sdxl_adafactor_simple test_run" "/custom/sdxl"

echo -e "${YELLOW}12. Testing error conditions${NC}"
run_test_fail "Invalid model family" "python train.py --preset flux_adafactor_simple --model-family invalid test_run" "invalid choice"
run_test_fail "Invalid VRAM mode" "python train.py --preset flux_adafactor_simple --vram-mode invalid test_run" "invalid choice"
run_test_fail "Invalid LyCORIS subtype" "python train.py --preset flux_adafactor_simple --lycoris-subtype invalid test_run" "invalid choice"

echo -e "${YELLOW}13. Testing configuration validation${NC}"
# Test with a malformed YAML file
echo "presets:" > /tmp/test_malformed.yaml
echo "  - name: test" >> /tmp/test_malformed.yaml
echo "    metadata:" >> /tmp/test_malformed.yaml
echo "      category: test" >> /tmp/test_malformed.yaml
echo "      description: test" >> /tmp/test_malformed.yaml
echo "    parameters:" >> /tmp/test_malformed.yaml
echo "      model_family: invalid_family" >> /tmp/test_malformed.yaml

run_test_fail "Invalid model family in config" "python train.py --sources /tmp/test_malformed.yaml --preset test test_run" "validation error"

# Clean up
rm -f /tmp/test_malformed.yaml

echo -e "${YELLOW}14. Testing wizard mode (non-interactive)${NC}"
# Test that wizard mode requires a training name
run_test "Wizard mode help" "python train.py --wizard --help" "usage: train.py"

echo "================================================"
echo -e "${GREEN}Tests completed!${NC}"
echo -e "Passed: ${GREEN}${TESTS_PASSED}${NC}"
echo -e "Failed: ${RED}${TESTS_FAILED}${NC}"

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}❌ Some tests failed!${NC}"
    exit 1
fi