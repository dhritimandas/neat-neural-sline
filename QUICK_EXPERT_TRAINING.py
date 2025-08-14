#!/usr/bin/env python3
"""
Quick Expert Training - Optimized for Speed
Runs much faster while still achieving expertise
"""

import os
import sys
import pickle
import numpy as np
import gym
import neat
from pathlib import Path
import time

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

try:
    import slimevolleygym
except ImportError:
    print("ERROR: slimevolleygym required")
    sys.exit(1)

# Global settings for speed
MAX_STEPS_PER_EPISODE = 500  # Reduced from 1000
ELITE_POOL = []

def make_env():
    try:
        return gym.make("SlimeVolley-v0", render_mode=None)
    except TypeError:
        return gym.make("SlimeVolley-v0")

def obs_to_np(obs):
    if isinstance(obs, tuple):
        obs = obs[0]
    return np.asarray(obs, dtype=np.float32)

def normalize_obs(obs):
    """Simple fast normalization"""
    return np.tanh(obs * 0.5)

def step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, rew, terminated, truncated, info = out
        done = terminated or truncated
        return obs, rew, done, info
    else:
        return out

def action_from_output(env, out_vec):
    out = np.asarray(out_vec, dtype=np.float32)
    return (out[:3] > 0.0).astype(np.int8)

def evaluate_fast(genome, config, generation=0):
    """Fast evaluation focused on key metrics"""
    
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    # Adaptive episodes based on generation
    if generation < 10:
        episodes = 1  # Very fast initial generations
    elif generation < 30:
        episodes = 2
    else:
        episodes = 3
    
    total_fitness = 0.0
    total_hits = 0
    total_wins = 0
    
    for ep in range(episodes):
        env = make_env()
        
        # Simple difficulty progression
        if hasattr(env, 'opponent_difficulty'):
            if generation < 10:
                env.opponent_difficulty = 0.0  # Static opponent
            elif generation < 25:
                env.opponent_difficulty = 0.2
            else:
                env.opponent_difficulty = min(0.5, 0.2 + generation * 0.01)
        
        obs = obs_to_np(env.reset())
        
        episode_fitness = 0.0
        done = False
        steps = 0
        
        last_ball_y = obs[1]
        ball_hits = 0
        game_result = 0
        
        while not done and steps < MAX_STEPS_PER_EPISODE:
            action = action_from_output(env, net.activate(normalize_obs(obs)))
            next_obs, rew, done, info = step_env(env, action)
            next_obs = obs_to_np(next_obs)
            
            # Simple reward structure
            fitness = 0.0
            
            # Stage-based rewards
            if generation < 15:  # Focus on ball contact
                # Check for ball hit
                if last_ball_y > 0.5 and next_obs[1] < 0.5:
                    if abs(next_obs[0] - next_obs[4]) < 0.15:
                        fitness += 100.0
                        ball_hits += 1
                    elif abs(next_obs[0] - next_obs[4]) < 0.3:
                        fitness += 20.0
                
                # Position reward
                if next_obs[0] > 0:  # Ball on our side
                    fitness += max(0, 2.0 - abs(next_obs[0] - next_obs[4]))
                
                # Small game score impact
                fitness += float(rew) * 2.0
                
            else:  # Focus on winning
                # Big win/loss impact
                fitness += float(rew) * 50.0
                
                # Ball hit bonus
                if last_ball_y > 0.5 and next_obs[1] < 0.5:
                    if abs(next_obs[0] - next_obs[4]) < 0.2:
                        fitness += 20.0
                        ball_hits += 1
                
                # Position bonus
                if next_obs[0] > 0:
                    fitness += max(0, 5.0 - abs(next_obs[0] - next_obs[4]) * 5)
            
            # Track game outcome
            game_result += rew
            
            episode_fitness += fitness
            last_ball_y = next_obs[1]
            obs = next_obs
            steps += 1
        
        # End bonuses
        if game_result > 0:
            episode_fitness += 100.0
            total_wins += 1
        
        if ball_hits > 0:
            episode_fitness += 20.0 * ball_hits
        
        total_fitness += episode_fitness
        total_hits += ball_hits
        env.close()
    
    # Store metrics
    genome.hits = total_hits / episodes
    genome.wins = total_wins / episodes
    
    return total_fitness / episodes

def run_quick_training():
    """Run fast training"""
    
    print("\n" + "="*60)
    print("QUICK EXPERT TRAINING - OPTIMIZED FOR SPEED")
    print("="*60)
    
    # Configuration
    config_path = "neat_config_expert.cfg"
    generations = 100
    population = 30  # Much smaller for speed
    
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    
    # Override population size for speed
    config.pop_size = population
    
    # Create population
    pop = neat.Population(config)
    
    # Simple reporter
    pop.add_reporter(neat.StdOutReporter(False))
    
    output_dir = Path("models/quick_expert")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Population: {population}")
    print(f"Generations: {generations}")
    print(f"Output: {output_dir}")
    print("="*60 + "\n")
    
    best_genome = None
    start_time = time.time()
    
    for generation in range(generations):
        gen_start = time.time()
        
        # Evaluate population
        for genome_id, genome in pop.population.items():
            genome.fitness = evaluate_fast(genome, config, generation)
        
        # Get stats
        fitnesses = [g.fitness for g in pop.population.values()]
        best_fitness = max(fitnesses)
        avg_fitness = np.mean(fitnesses)
        
        hits = [getattr(g, 'hits', 0) for g in pop.population.values()]
        avg_hits = np.mean(hits) if hits else 0
        
        wins = [getattr(g, 'wins', 0) for g in pop.population.values()]
        win_rate = np.mean(wins) if wins else 0
        
        # Display progress
        gen_time = time.time() - gen_start
        total_time = time.time() - start_time
        
        print(f"Gen {generation:3d} | Time: {gen_time:4.1f}s | Total: {total_time/60:4.1f}m")
        print(f"  Fitness: Best={best_fitness:6.1f} Avg={avg_fitness:6.1f}")
        print(f"  Metrics: Hits={avg_hits:.2f}/game WinRate={win_rate*100:.1f}%")
        
        # Stage transitions
        if generation == 15:
            print("\n🎯 TRANSITION: Switching to winning focus\n")
        
        # Check for success
        if win_rate > 0.7 and avg_hits > 2.5:
            print(f"\n🏆 EXPERT LEVEL ACHIEVED at Generation {generation}!")
            print(f"  Win Rate: {win_rate*100:.1f}%")
            print(f"  Hits/Game: {avg_hits:.2f}")
            best_genome = max(pop.population.values(), key=lambda g: g.fitness)
            break
        
        # Save best periodically
        if generation % 20 == 0 or generation == generations - 1:
            best_genome = max(pop.population.values(), key=lambda g: g.fitness)
            save_path = output_dir / f"gen_{generation}.pkl"
            with open(save_path, 'wb') as f:
                pickle.dump((best_genome, config), f)
            print(f"  [Saved checkpoint: {save_path}]")
        
        # Create next generation
        pop.population = pop.reproduction.reproduce(config, pop.species, 
                                                   pop.config.pop_size, generation)
        pop.species.speciate(config, pop.population, generation)
    
    # Save final model
    if not best_genome:
        best_genome = max(pop.population.values(), key=lambda g: g.fitness)
    
    final_path = output_dir / "best_genome.pkl"
    with open(final_path, 'wb') as f:
        pickle.dump((best_genome, config), f)
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training Complete in {total_time/60:.1f} minutes")
    print(f"Final Model: {final_path}")
    print(f"Final Fitness: {best_genome.fitness:.1f}")
    print(f"Final Hits/Game: {getattr(best_genome, 'hits', 0):.2f}")
    print(f"Final Win Rate: {getattr(best_genome, 'wins', 0)*100:.1f}%")
    print(f"{'='*60}")
    
    return best_genome

if __name__ == "__main__":
    run_quick_training()