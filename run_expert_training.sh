#!/bin/bash
# Expert Training Launcher Script
# Runs staged training with monitoring and evaluation

echo "============================================================"
echo "NEAT SLIMEVOLLEY EXPERT TRAINING SYSTEM"
echo "============================================================"
echo "This training uses staged learning:"
echo "  Stage 1: Ball Contact (static opponent)"
echo "  Stage 2: Positioning (easy opponent)"
echo "  Stage 3: Winning (adaptive opponent)"
echo "============================================================"

# Configuration
GENERATIONS=150
EPISODES=3
POPULATION=150
OUTPUT_DIR="models/expert"

# Create output directory
mkdir -p $OUTPUT_DIR

# Kill any existing training
echo "Stopping any existing training processes..."
pkill -f train_neat_expert 2>/dev/null
pkill -f monitor_training 2>/dev/null
sleep 2

# Start training in background
echo ""
echo "Starting expert training..."
echo "  Generations: $GENERATIONS"
echo "  Population: $POPULATION"
echo "  Episodes per eval: $EPISODES"
echo "  Output: $OUTPUT_DIR"
echo ""

nohup python train_neat_expert.py \
    --config neat_config_expert.cfg \
    --generations $GENERATIONS \
    --episodes $EPISODES \
    --population $POPULATION \
    > $OUTPUT_DIR/training.log 2>&1 &

TRAINING_PID=$!
echo "Training started with PID: $TRAINING_PID"

# Wait a moment for training to initialize
sleep 5

# Start monitoring in foreground
echo ""
echo "Starting training monitor..."
echo "Press Ctrl+C to stop monitoring (training will continue)"
echo "============================================================"

# Monitor the training log
tail -f $OUTPUT_DIR/training.log | grep -E "Generation|Stage|Fitness|Hits|Win|TRANSITION|EXPERT|ERROR" &
TAIL_PID=$!

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping monitor..."
    kill $TAIL_PID 2>/dev/null
    echo ""
    echo "Training continues in background (PID: $TRAINING_PID)"
    echo "To stop training: kill $TRAINING_PID"
    echo "To check progress: tail -f $OUTPUT_DIR/training.log"
    echo "To evaluate: python evaluate_performance.py --model $OUTPUT_DIR/best_genome.pkl"
    exit 0
}

# Set trap for Ctrl+C
trap cleanup INT

# Wait for training to complete or user interrupt
wait $TAIL_PID 2>/dev/null

echo ""
echo "============================================================"
echo "Training session ended"
echo "Check results in: $OUTPUT_DIR/"
echo "============================================================"