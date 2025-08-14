# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a NEAT (NeuroEvolution of Augmenting Topologies) neural network project for training agents to play SlimeVolley, a simple volleyball-like game. The project uses the NEAT-Python library to evolve neural networks that can compete against opponents in the SlimeVolley environment.

## Key Commands

### Training
```bash
# Train a new NEAT population
python train_neat.py

# Resume training from checkpoint
python train_neat.py --checkpoint models/chkpt-10

# Train with specific parameters
python train_neat.py --generations 100 --episodes 5 --workers 4

# Overfit on single episode for debugging
python train_neat.py --overfit
```

### Playing/Testing
```bash
# Play with the best trained model
python play_neat.py --best

# Play with specific model
python play_neat.py --model models/top1_genome.pkl

# Play multiple episodes with debug info
python play_neat.py --episodes 5 --debug
```

### Dependencies Installation
```bash
pip install -r requirements.txt
```

## Architecture

### Core Components

**train_neat.py**: Main training script that evolves neural networks using NEAT algorithm
- Implements reward shaping system with ball contact, positioning, and strategic play rewards
- Features curriculum learning with progressive opponent difficulty
- Includes cross-species competition using best genomes from previous generations
- Supports parallel evaluation for faster training

**play_neat.py**: Evaluation script for testing trained models
- Renders gameplay with visual feedback
- Matches reward shaping logic from training for consistency
- Supports JAX acceleration (optional)
- Provides performance metrics and debugging output

**neat_config_slime.cfg**: NEAT configuration file defining:
- Network topology (12 inputs, 3 outputs, 20 hidden nodes)
- Mutation rates and evolutionary parameters
- Species and population settings

### Key Features

**Observation Normalization**: Both scripts use `normalize_obs()` to standardize input features to [-1,1] range for better neural network training.

**Reward Shaping**: Complex reward system beyond game score:
- Ball contact rewards (10+ points for hits)
- Positioning rewards for staying near ball
- Strategic play rewards for defensive positioning
- Penalty system for inactivity or poor movement

**Cross-Species Competition**: Training uses best genomes from previous generations as opponents to maintain learning pressure.

**Curriculum Learning**: Opponent difficulty progressively increases based on generation number.

## File Structure

- `train_neat.py` - Training script
- `play_neat.py` - Evaluation/testing script  
- `neat_config_slime.cfg` - NEAT algorithm configuration
- `requirements.txt` - Python dependencies
- `models/` - Directory for saved models and checkpoints
  - `winner.pkl` - Best genome from final generation
  - `best_genome.pkl` - Highest fitness genome overall
  - `top1_genome.pkl`, `top2_genome.pkl`, `top3_genome.pkl` - Top performing genomes
  - `chkpt-*` - Training checkpoints for resuming

## Development Notes

- The project uses slimevolleygym environment which requires OpenCV for rendering
- Network expects 12-dimensional observation space (ball position/velocity, player position/velocity, opponent position/velocity)
- Actions are 3-dimensional binary: [left, right, jump]
- Training supports both single-threaded and parallel evaluation modes
- Models are saved as pickled tuples of (genome, config) for compatibility