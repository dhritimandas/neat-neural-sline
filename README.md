# NEAT Neural Slime Volleyball 🏐🤖

An advanced implementation of NEAT (NeuroEvolution of Augmenting Topologies) to train AI agents that master the SlimeVolley game. This project features staged learning, comprehensive performance tracking, and expert-level training strategies.

## 🎯 Project Goals

Train a neural network agent to:
- **Master ball control** - Consistently hit the ball
- **Strategic positioning** - Optimal court coverage
- **Win against AI** - Achieve 75%+ win rate
- **Expert gameplay** - 3+ hits per game with <3 step reaction time

## 📊 Current Performance

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Win Rate | 0% | 75%+ | ❌ Training needed |
| Ball Hits/Game | 0 | 3+ | ❌ Training needed |
| Reaction Time | 10 steps | <3 | ❌ Training needed |
| Point Differential | -24 | >0 | ❌ Training needed |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/neat-neural-slime.git
cd neat-neural-slime

# Install dependencies
pip install -r requirements.txt
```

### Training

#### Option 1: Expert Training (Recommended)
```bash
# Run staged expert training
./START_EXPERT_TRAINING.sh
# Select option 2 for standard (1 hour) or 3 for full (3 hours)
```

#### Option 2: Quick Test
```bash
# Fast training with small population
python train_neat_expert.py --generations 20 --episodes 2 --population 30
```

#### Option 3: Background Training
```bash
# Run in background and monitor
./start_background_training.sh
```

### Evaluation

```bash
# Check if model achieved expert status
python check_expert_status.py --model models/robust/winner.pkl --episodes 10

# Full performance evaluation
python evaluate_performance.py --model models/robust/winner.pkl --episodes 100 --optimize
```

### Playing

```bash
# Watch the trained agent play
python play_neat.py --model models/robust/winner.pkl --episodes 5
```

## 🏗️ Project Structure

```
neat-neural-slime/
├── Core Training Scripts
│   ├── train_neat.py                 # Original training script
│   ├── train_neat_optimized.py       # Optimized with enhanced rewards
│   └── train_neat_expert.py          # Staged learning for expertise
│
├── Configuration Files
│   ├── neat_config_slime.cfg         # Original NEAT config
│   ├── neat_config_optimized.cfg     # Optimized parameters
│   ├── neat_config_expert.cfg        # Expert training config
│   └── neat_config_fast.cfg          # Quick testing config
│
├── Evaluation & Monitoring
│   ├── evaluate_performance.py       # Comprehensive performance analysis
│   ├── check_expert_status.py        # Quick expert criteria check
│   ├── monitor_training.py           # Real-time training dashboard
│   └── live_metrics.sh              # Live metrics display
│
├── Utilities
│   ├── play_neat.py                  # Play/visualize trained models
│   ├── START_EXPERT_TRAINING.sh      # Automated training launcher
│   ├── run_expert_training.sh        # Expert training script
│   └── run_safe_training.py          # Parameter validation launcher
│
├── Models
│   └── robust/                       # Pre-trained models
│       ├── winner.pkl                # Best from final generation
│       ├── best_genome.pkl           # Overall best fitness
│       └── top[1-3]_genome.pkl      # Top 3 performers
│
└── Documentation
    ├── README.md                      # This file
    ├── CLAUDE.md                      # Claude AI instructions
    └── requirements.txt               # Python dependencies
```

## 🎓 Training Strategy: Staged Learning

### Stage 1: Ball Contact (Generations 0-15)
- **Opponent**: Static (difficulty 0.0)
- **Focus**: Learn to hit the ball
- **Reward**: 100 points per hit
- **Success**: >0.5 hits/game → Stage 2

### Stage 2: Positioning (Generations 15-30)
- **Opponent**: Easy (difficulty 0.2-0.3)
- **Focus**: Ball contact + positioning
- **Reward**: 50 points per hit + position quality
- **Success**: >2.0 hits/game → Stage 3

### Stage 3: Winning (Generations 30+)
- **Opponent**: Adaptive (difficulty 0.3-1.0)
- **Focus**: Win games
- **Reward**: 100x win/loss impact
- **Success**: 75%+ win rate = EXPERT

## 📈 Key Features

### 1. Advanced Reward Shaping
- **Ball contact rewards** - Massive bonuses for hitting
- **Positioning rewards** - Encourage optimal court coverage
- **Predictive rewards** - Bonus for anticipating ball trajectory
- **Win bonuses** - 200+ points for winning games

### 2. Performance Tracking
- **Real-time metrics** - Live training dashboard
- **Detailed evaluation** - 30+ performance metrics
- **Expert criteria** - 5-point checklist for expertise
- **Visual progress** - Color-coded terminal output

### 3. Training Optimizations
- **No overfitting** - Disabled single-scenario training
- **Elite competition** - Best agents compete against each other
- **Curriculum learning** - Progressive difficulty increase
- **Staged objectives** - One skill at a time

## 🔧 Configuration

Key parameters in `neat_config_expert.cfg`:
```ini
pop_size = 150          # Population size (150 balanced, 300 for best results)
num_hidden = 15         # Hidden nodes (15 for speed, 30 for complexity)
weight_mutate_rate = 0.8  # Weight mutation (0.8 balanced)
conn_add_prob = 0.5     # Add connection probability
```

## 🐛 Troubleshooting

### Training is too slow
- Reduce population: `--population 30`
- Reduce episodes: `--episodes 1`
- Use cloud compute or run overnight

### No ball hits after 10 generations
- Check if slimevolleygym is working: `python -c "import slimevolleygym"`
- Reduce opponent difficulty in early stages
- Increase ball hit reward in train_neat_expert.py

### High fitness but 0% win rate
- **This was our main issue!** Caused by overfitting
- Solution: Never use `--overfit` flag
- Ensure multiple episodes for generalization

## 📊 Expert Criteria

An agent is considered "expert" when it achieves ALL:
- ✅ **Win Rate ≥ 75%**
- ✅ **Ball Hits ≥ 3.0 per game**
- ✅ **Reaction Time < 3 steps**
- ✅ **Point Differential > 0**
- ✅ **Hit Success Rate ≥ 80%**

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Parallelize training for speed
- Implement self-play training
- Add more sophisticated reward shaping
- Create difficulty levels for human play

## 📄 License

MIT License - feel free to use this code for your own projects!

## 🙏 Acknowledgments

- [NEAT-Python](https://github.com/CodeReclaimers/neat-python) - NEAT implementation
- [SlimeVolleyGym](https://github.com/hardmaru/slimevolleygym) - Game environment
- OpenAI Gym - Reinforcement learning framework

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: Training to expert level requires significant compute time (3-6 hours on standard hardware). Consider using cloud compute or running overnight for best results.