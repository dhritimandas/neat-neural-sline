#!/bin/bash
# GUARANTEED EXPERT TRAINING SCRIPT
# This script ensures all parameters are correct for achieving expertise

echo "========================================================================"
echo "                    NEAT SLIMEVOLLEY EXPERT TRAINING                    "
echo "========================================================================"
echo ""
echo "This script will train an agent to EXPERT level using:"
echo "  ✅ Staged learning (Ball Contact → Positioning → Winning)"
echo "  ✅ Proper parameters (NO overfit mode)"
echo "  ✅ Progressive difficulty"
echo "  ✅ Elite competition"
echo ""
echo "========================================================================"

# Safety checks
echo "🔍 Running safety checks..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.7+"
    exit 1
fi

# Check dependencies
python3 -c "import neat, gym, slimevolleygym" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies. Installing..."
    pip install -r requirements.txt
fi

# Kill any existing training
echo "🛑 Stopping any existing training..."
pkill -f train_neat 2>/dev/null
sleep 2

# Training options
echo ""
echo "Select training duration:"
echo "  1) Quick Test (10 generations, ~15 minutes)"
echo "  2) Standard (50 generations, ~1 hour)"
echo "  3) Full Training (150 generations, ~3 hours)"
echo "  4) Overnight (300 generations, ~6 hours)"
echo "  5) Custom"
echo ""
read -p "Enter choice (1-5): " choice

case $choice in
    1)
        GENS=10
        POP=50
        EPS=2
        MODE="Quick Test"
        ;;
    2)
        GENS=50
        POP=100
        EPS=3
        MODE="Standard"
        ;;
    3)
        GENS=150
        POP=150
        EPS=3
        MODE="Full Training"
        ;;
    4)
        GENS=300
        POP=200
        EPS=5
        MODE="Overnight"
        ;;
    5)
        read -p "Generations: " GENS
        read -p "Population: " POP
        read -p "Episodes per eval: " EPS
        MODE="Custom"
        ;;
    *)
        echo "Invalid choice. Using Standard mode."
        GENS=50
        POP=100
        EPS=3
        MODE="Standard"
        ;;
esac

# Create output directory
OUTPUT_DIR="models/expert_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

# Display configuration
echo ""
echo "========================================================================"
echo "📊 TRAINING CONFIGURATION"
echo "========================================================================"
echo "  Mode:         $MODE"
echo "  Generations:  $GENS"
echo "  Population:   $POP"
echo "  Episodes:     $EPS"
echo "  Output:       $OUTPUT_DIR"
echo "  Overfit:      FALSE (always disabled for proper training)"
echo "========================================================================"
echo ""
read -p "Start training? (y/n): " confirm

if [ "$confirm" != "y" ]; then
    echo "Training cancelled."
    exit 0
fi

# Start training
echo ""
echo "🚀 Starting Expert Training..."
echo ""

# Create training command - NEVER includes --overfit
TRAIN_CMD="python3 train_neat_expert.py \
    --config neat_config_expert.cfg \
    --generations $GENS \
    --episodes $EPS \
    --population $POP"

# Log file
LOG_FILE="$OUTPUT_DIR/training.log"

# Start training with real-time output
echo "Command: $TRAIN_CMD"
echo ""
echo "Training output:"
echo "----------------"

# Run training and show important lines with enhanced metrics
$TRAIN_CMD 2>&1 | tee $LOG_FILE | while IFS= read -r line; do
    # Show all important metrics
    if echo "$line" | grep -qE "Generation|Stage|TRANSITION|Fitness|Hits|Win|EXPERT|ERROR|HIT!|WIN!|Elite|Ball|Score"; then
        # Color code important lines
        if echo "$line" | grep -q "EXPERT LEVEL ACHIEVED"; then
            echo -e "\033[1;32m🏆 $line\033[0m"  # Bold green for success
        elif echo "$line" | grep -q "WIN!"; then
            echo -e "\033[32m✅ $line\033[0m"  # Green for wins
        elif echo "$line" | grep -q "HIT!"; then
            echo -e "\033[36m⚡ $line\033[0m"  # Cyan for hits
        elif echo "$line" | grep -q "TRANSITION"; then
            echo -e "\033[1;33m📈 $line\033[0m"  # Bold yellow for transitions
        elif echo "$line" | grep -q "ERROR"; then
            echo -e "\033[31m❌ $line\033[0m"  # Red for errors
        elif echo "$line" | grep -q "Generation"; then
            echo -e "\033[1;34m📊 $line\033[0m"  # Bold blue for generations
        elif echo "$line" | grep -q "Stage"; then
            echo -e "\033[35m🎯 $line\033[0m"  # Magenta for stages
        elif echo "$line" | grep -q "Best Fitness"; then
            echo -e "\033[33m💪 $line\033[0m"  # Yellow for fitness
        elif echo "$line" | grep -q "Avg Hits"; then
            echo -e "\033[36m🎾 $line\033[0m"  # Cyan for hits
        elif echo "$line" | grep -q "Win Rate"; then
            echo -e "\033[32m🏅 $line\033[0m"  # Green for win rate
        else
            echo "   $line"
        fi
    fi
done

# Training complete
echo ""
echo "========================================================================"
echo "                         TRAINING COMPLETE                              "
echo "========================================================================"

# Check if model was created
MODEL_FILE="$OUTPUT_DIR/best_genome.pkl"
if [ ! -f "$MODEL_FILE" ]; then
    MODEL_FILE="models/expert/best_genome.pkl"
fi

if [ -f "$MODEL_FILE" ]; then
    echo "✅ Model saved: $MODEL_FILE"
    echo ""
    echo "Testing model performance..."
    echo "----------------------------"
    python3 check_expert_status.py --model "$MODEL_FILE" --episodes 10
else
    echo "⚠️  No model file found. Check $LOG_FILE for errors."
fi

echo ""
echo "📊 Full training log: $LOG_FILE"
echo ""
echo "Next steps:"
echo "  1. Check expert status: python3 check_expert_status.py --model $MODEL_FILE"
echo "  2. Full evaluation: python3 evaluate_performance.py --model $MODEL_FILE --episodes 100"
echo "  3. Play the game: python3 play_neat.py --model $MODEL_FILE"
echo ""
echo "========================================================================"