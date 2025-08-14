#!/bin/bash
# Background Training with Progress Monitoring

echo "======================================================================"
echo "                 BACKGROUND EXPERT TRAINING                          "
echo "======================================================================"
echo ""
echo "This will run training in the background and show progress."
echo ""

# Configuration
POPULATION=30  # Small for speed
GENERATIONS=100
EPISODES=2
MAX_STEPS=400
OUTPUT_DIR="models/expert_bg"

# Create output directory
mkdir -p $OUTPUT_DIR

# Kill any existing training
pkill -f train_neat 2>/dev/null
pkill -f QUICK_EXPERT 2>/dev/null
sleep 2

# Create simplified training script
cat > quick_train.py << 'EOF'
import sys, os, pickle, numpy as np, gym, neat, time
from pathlib import Path

try:
    import slimevolleygym
except:
    print("ERROR: slimevolleygym required")
    sys.exit(1)

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

def make_env():
    try:
        return gym.make("SlimeVolley-v0", render_mode=None)
    except:
        return gym.make("SlimeVolley-v0")

def evaluate_simple(genome, config, gen=0):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = make_env()
    
    # Easy opponent for learning
    if hasattr(env, 'opponent_difficulty'):
        env.opponent_difficulty = min(0.3, gen * 0.01)
    
    obs = np.asarray(env.reset(), dtype=np.float32)
    if isinstance(obs, tuple):
        obs = obs[0]
    
    fitness = 0.0
    hits = 0
    done = False
    steps = 0
    
    while not done and steps < 300:
        # Simple network evaluation
        action = (net.activate(np.tanh(obs * 0.5))[:3] > 0.0).astype(np.int8)
        
        out = env.step(action)
        if len(out) == 5:
            next_obs, rew, terminated, truncated, info = out
            done = terminated or truncated
        else:
            next_obs, rew, done, info = out
        
        if isinstance(next_obs, tuple):
            next_obs = next_obs[0]
        next_obs = np.asarray(next_obs, dtype=np.float32)
        
        # Simple rewards
        if gen < 20:  # Focus on ball contact
            if obs[1] > 0.5 and next_obs[1] < 0.5 and abs(next_obs[0] - next_obs[4]) < 0.2:
                fitness += 100
                hits += 1
                print(f"HIT! Gen {gen}")
            fitness += max(0, 2 - abs(next_obs[0] - next_obs[4]))
        else:  # Focus on winning
            fitness += rew * 50
            if obs[1] > 0.5 and next_obs[1] < 0.5 and abs(next_obs[0] - next_obs[4]) < 0.2:
                fitness += 20
                hits += 1
        
        obs = next_obs
        steps += 1
    
    env.close()
    genome.hits = hits
    return fitness

# Main training
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     "neat_config_expert.cfg")
config.pop_size = 30

pop = neat.Population(config)
output_dir = Path("models/expert_bg")
output_dir.mkdir(exist_ok=True)

print("\nStarting Quick Training - 30 pop, 100 gens")
print("="*50)

for gen in range(100):
    start = time.time()
    
    # Evaluate
    for gid, genome in pop.population.items():
        genome.fitness = evaluate_simple(genome, config, gen)
    
    # Stats
    fitnesses = [g.fitness for g in pop.population.values()]
    hits = [getattr(g, 'hits', 0) for g in pop.population.values()]
    
    # Print progress
    print(f"Gen {gen:3d} | Time: {time.time()-start:3.1f}s | "
          f"Best: {max(fitnesses):6.1f} | Avg: {np.mean(fitnesses):6.1f} | "
          f"Hits: {np.mean(hits):.2f}")
    
    # Save progress to file
    with open(output_dir / "progress.txt", "a") as f:
        f.write(f"{gen},{max(fitnesses)},{np.mean(fitnesses)},{np.mean(hits)}\n")
    
    # Save best every 10 gens
    if gen % 10 == 0:
        best = max(pop.population.values(), key=lambda g: g.fitness)
        with open(output_dir / f"gen_{gen}.pkl", 'wb') as f:
            pickle.dump((best, config), f)
    
    # Check success
    if np.mean(hits) > 2:
        print(f"\nSUCCESS at gen {gen}! Avg hits: {np.mean(hits):.2f}")
        best = max(pop.population.values(), key=lambda g: g.fitness)
        with open(output_dir / "best.pkl", 'wb') as f:
            pickle.dump((best, config), f)
        break
    
    # Next generation
    pop.population = pop.reproduction.reproduce(config, pop.species, 30, gen)
    pop.species.speciate(config, pop.population, gen)

print("\nTraining complete!")
EOF

echo "Starting training in background..."
nohup python quick_train.py > $OUTPUT_DIR/training.log 2>&1 &
TRAIN_PID=$!

echo "Training started with PID: $TRAIN_PID"
echo ""

# Monitor progress
echo "Monitoring progress (Ctrl+C to stop monitoring, training continues)..."
echo "======================================================================"
echo ""

# Function to show progress
monitor_progress() {
    while true; do
        if [ -f "$OUTPUT_DIR/progress.txt" ]; then
            echo -e "\033[2J\033[H"  # Clear screen
            echo "======================================================================"
            echo "                    TRAINING PROGRESS                                "
            echo "======================================================================"
            echo ""
            
            # Get latest stats
            LAST_LINE=$(tail -1 $OUTPUT_DIR/progress.txt)
            GEN=$(echo $LAST_LINE | cut -d',' -f1)
            BEST=$(echo $LAST_LINE | cut -d',' -f2)
            AVG=$(echo $LAST_LINE | cut -d',' -f3)
            HITS=$(echo $LAST_LINE | cut -d',' -f4)
            
            echo "Generation:    $GEN / 100"
            echo "Best Fitness:  $BEST"
            echo "Avg Fitness:   $AVG"
            echo "Avg Hits:      $HITS"
            echo ""
            
            # Progress bar
            PROGRESS=$((GEN * 100 / 100))
            echo -n "Progress: ["
            for i in $(seq 1 20); do
                if [ $((i * 5)) -le $PROGRESS ]; then
                    echo -n "■"
                else
                    echo -n "□"
                fi
            done
            echo "] $PROGRESS%"
            echo ""
            
            # Check if still running
            if ! kill -0 $TRAIN_PID 2>/dev/null; then
                echo "Training complete!"
                break
            fi
            
            echo "Recent output:"
            tail -5 $OUTPUT_DIR/training.log | sed 's/^/  /'
            echo ""
            echo "======================================================================"
            echo "PID: $TRAIN_PID | Log: $OUTPUT_DIR/training.log"
            echo "Ctrl+C to stop monitoring (training continues)"
        else
            echo "Waiting for training to start..."
        fi
        
        sleep 2
    done
}

# Trap Ctrl+C
trap 'echo -e "\n\nStopped monitoring. Training continues with PID: $TRAIN_PID"; exit 0' INT

# Start monitoring
monitor_progress