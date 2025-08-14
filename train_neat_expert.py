#!/usr/bin/env python3
"""
Advanced NEAT Training for Expert-Level SlimeVolley Performance
Uses staged learning: Ball Contact -> Positioning -> Winning
"""

import os
import sys
import pickle
import argparse
import traceback
import random
import numpy as np
import gym
import neat
from pathlib import Path
from collections import deque
from datetime import datetime

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

print(f"[EXPERT] Advanced training initialized at {datetime.now()}", flush=True)

try:
    import slimevolleygym
except ImportError:
    print("ERROR: slimevolleygym required")
    sys.exit(1)

# Global tracking
TRAINING_STAGE = 1  # 1: Ball Contact, 2: Positioning, 3: Winning
STAGE_TRANSITIONS = {1: 0.5, 2: 2.0, 3: 10.0}  # Fitness thresholds for stage transitions
ELITE_POOL = []
MAX_ELITE = 5
GENERATION_COUNTER = 0
PERFORMANCE_TRACKER = {
    'hits_per_game': [],
    'win_rate': [],
    'avg_fitness': []
}

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
    """Aggressive normalization for better neural network performance"""
    norm_obs = np.copy(obs)
    # Position normalization
    norm_obs[0] = np.tanh(norm_obs[0] * 2.0)  # Ball x - more sensitive
    norm_obs[1] = np.tanh(norm_obs[1] * 2.0)  # Ball y - more sensitive
    norm_obs[4] = np.tanh(norm_obs[4] * 2.0)  # Player x
    norm_obs[8] = np.tanh(norm_obs[8] * 2.0)  # Opponent x
    
    # Velocity normalization - make very sensitive to small changes
    for i in [2, 3, 6, 7, 10, 11]:
        norm_obs[i] = np.tanh(norm_obs[i] * 0.5)
    
    return norm_obs

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
    # Add small noise during training for exploration
    if random.random() < 0.05:  # 5% random actions
        return np.random.randint(0, 2, size=3).astype(np.int8)
    return (out[:3] > 0.0).astype(np.int8)

def flip_observation(obs):
    """Flip observation for opponent's perspective"""
    flipped = np.copy(obs)
    flipped[0] = -flipped[0]  # ball x
    flipped[2] = -flipped[2]  # ball vx
    flipped[4], flipped[8] = -flipped[8], -flipped[4]  # swap x positions
    flipped[5], flipped[9] = flipped[9], flipped[5]    # swap y positions
    flipped[6], flipped[10] = -flipped[10], -flipped[6]  # swap vx
    flipped[7], flipped[11] = flipped[11], flipped[7]    # swap vy
    return flipped

def evaluate_genome_staged(genome, config, stage=1, episodes=3, max_steps=1000):
    """Staged evaluation focusing on different skills"""
    global GENERATION_COUNTER
    
    try:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        total_fitness = 0.0
        
        # Performance metrics
        total_hits = 0
        total_wins = 0
        total_touches = 0
        total_points_scored = 0
        total_points_lost = 0
        
        for episode in range(episodes):
            env = make_env()
            
            # CRITICAL: Set difficulty based on stage and generation
            if hasattr(env, 'opponent_difficulty'):
                if stage == 1:  # Learning ball contact
                    difficulty = 0.0  # Static opponent
                elif stage == 2:  # Learning positioning
                    difficulty = min(0.3, 0.1 + GENERATION_COUNTER * 0.01)
                else:  # Stage 3: Winning
                    difficulty = min(1.0, 0.3 + GENERATION_COUNTER * 0.02)
                env.opponent_difficulty = difficulty
            
            # Use elite opponent occasionally in later stages
            use_elite = stage >= 2 and len(ELITE_POOL) > 0 and random.random() < 0.3
            
            if use_elite:
                try:
                    elite_genome, elite_config = random.choice(ELITE_POOL)
                    elite_net = neat.nn.FeedForwardNetwork.create(elite_genome, elite_config)
                    original_policy = env.policy if hasattr(env, 'policy') else None
                    
                    def elite_policy(obs):
                        flipped = flip_observation(obs)
                        return action_from_output(env, elite_net.activate(normalize_obs(flipped)))
                    
                    env.policy = elite_policy
                except:
                    use_elite = False
            
            # Reset environment
            obs = obs_to_np(env.reset())
            
            # Episode variables
            episode_reward = 0.0
            done = False
            steps = 0
            
            # Tracking variables
            ball_hits = 0
            near_misses = 0
            ball_touches = 0
            
            last_ball_x = obs[0]
            last_ball_y = obs[1]
            last_player_x = obs[4]
            
            position_scores = []
            movement_count = 0
            correct_movements = 0
            
            while not done and steps < max_steps:
                # Get action from network
                action = action_from_output(env, net.activate(normalize_obs(obs)))
                
                # Step environment
                next_obs, rew, done, info = step_env(env, action)
                next_obs = obs_to_np(next_obs)
                
                ball_x = next_obs[0]
                ball_y = next_obs[1]
                player_x = next_obs[4]
                
                # STAGE-SPECIFIC REWARDS
                shaped_reward = 0.0
                
                if stage == 1:  # STAGE 1: BALL CONTACT FOCUS
                    # Massive reward for any ball contact
                    if last_ball_y > 0.5 and ball_y < 0.5:
                        dist = abs(ball_x - player_x)
                        if dist < 0.1:
                            shaped_reward = 100.0  # HUGE reward
                            ball_hits += 1
                            print(f"[STAGE 1] HIT! Distance={dist:.3f}")
                        elif dist < 0.2:
                            shaped_reward = 25.0  # Good attempt
                            near_misses += 1
                        elif dist < 0.3:
                            shaped_reward = 10.0  # Getting closer
                            ball_touches += 1
                    
                    # Reward being under the ball
                    if ball_x > 0 and ball_y > 0.3:
                        position_score = max(0, 1.0 - abs(ball_x - player_x))
                        shaped_reward += position_score * 5.0
                        position_scores.append(position_score)
                    
                    # Small movement reward
                    if abs(player_x - last_player_x) > 0.01:
                        shaped_reward += 0.5
                        movement_count += 1
                        
                        # Reward moving toward ball
                        if (ball_x > player_x and action[1] > 0) or (ball_x < player_x and action[0] > 0):
                            shaped_reward += 2.0
                            correct_movements += 1
                    
                    # Ignore win/loss in stage 1
                    shaped_reward += float(rew) * 0.1  # Minimal game score impact
                
                elif stage == 2:  # STAGE 2: POSITIONING + BALL CONTACT
                    # Ball contact still important
                    if last_ball_y > 0.5 and ball_y < 0.5:
                        dist = abs(ball_x - player_x)
                        if dist < 0.1:
                            shaped_reward = 50.0
                            ball_hits += 1
                        elif dist < 0.25:
                            shaped_reward = 10.0
                            ball_touches += 1
                    
                    # Strong positioning rewards
                    if ball_x > 0:  # Ball on our side
                        position_score = max(0, 1.0 - abs(ball_x - player_x) * 2)
                        shaped_reward += position_score * 10.0
                        position_scores.append(position_score)
                    else:  # Defensive position
                        center_dist = abs(player_x - 0.7)
                        shaped_reward += (1.0 - center_dist) * 3.0
                    
                    # Predictive positioning
                    if ball_y > 0.4 and ball_y < 0.8 and ball_x > 0:
                        # Estimate landing position
                        y_vel = ball_y - last_ball_y
                        if y_vel < 0:  # Ball descending
                            x_vel = ball_x - last_ball_x
                            time_to_ground = max(0.1, (0.1 - ball_y) / abs(y_vel))
                            predicted_x = ball_x + x_vel * time_to_ground
                            
                            if 0 < predicted_x < 2:
                                prediction_error = abs(player_x - predicted_x)
                                if prediction_error < 0.15:
                                    shaped_reward += 20.0  # Great prediction
                    
                    # Moderate game score impact
                    shaped_reward += float(rew) * 5.0
                
                else:  # STAGE 3: WINNING FOCUS
                    # MASSIVE win/loss impact
                    shaped_reward = float(rew) * 100.0
                    
                    # Ball contact bonus
                    if last_ball_y > 0.5 and ball_y < 0.5:
                        dist = abs(ball_x - player_x)
                        if dist < 0.15:
                            shaped_reward += 30.0
                            ball_hits += 1
                    
                    # Rally continuation bonus
                    if steps > 20 and not done:
                        shaped_reward += steps * 0.05
                    
                    # Positioning bonus
                    if ball_x > 0:
                        shaped_reward += max(0, 5.0 - abs(ball_x - player_x) * 10)
                    
                    # Aggressive play bonus
                    if ball_hits > 0 and ball_x < 0 and last_ball_x > 0:
                        shaped_reward += 20.0  # Successfully returned ball
                
                # Track game outcome
                if rew > 0:
                    total_points_scored += 1
                    if done:
                        total_wins += 1
                elif rew < 0:
                    total_points_lost += 1
                
                # Update state
                episode_reward += shaped_reward
                last_ball_x = ball_x
                last_ball_y = ball_y
                last_player_x = player_x
                obs = next_obs
                steps += 1
            
            # Episode completion bonuses
            if stage == 1:
                # Bonus for any ball contact
                if ball_hits > 0:
                    episode_reward += 50.0 * ball_hits
                    print(f"[Episode {episode+1}] Ball hits: {ball_hits}")
                if near_misses > 0:
                    episode_reward += 10.0 * near_misses
                if movement_count > 0 and correct_movements / movement_count > 0.5:
                    episode_reward += 20.0  # Good movement patterns
            
            elif stage == 2:
                # Bonus for consistent positioning
                if position_scores:
                    avg_position = np.mean(position_scores)
                    if avg_position > 0.6:
                        episode_reward += 50.0
                if ball_hits >= 2:
                    episode_reward += 100.0
            
            else:  # Stage 3
                # Massive win bonus
                if total_wins > 0:
                    episode_reward += 200.0
                    print(f"[STAGE 3] WIN!")
                # Bonus for competitive games
                if total_points_scored > 0:
                    episode_reward += 50.0 * total_points_scored
            
            total_fitness += episode_reward
            total_hits += ball_hits
            total_touches += ball_touches
            
            env.close()
        
        # Calculate average fitness
        avg_fitness = total_fitness / episodes
        
        # Track performance
        genome.hits_per_game = total_hits / episodes
        genome.win_rate = total_wins / episodes
        genome.touch_rate = total_touches / episodes
        
        # Update elite pool if performance is good
        if avg_fitness > 50 and stage >= 2:
            if len(ELITE_POOL) < MAX_ELITE:
                ELITE_POOL.append((genome, config))
                print(f"[ELITE] Added genome with fitness {avg_fitness:.2f}")
            elif avg_fitness > min(e[0].fitness for e in ELITE_POOL if hasattr(e[0], 'fitness')):
                # Replace worst elite
                ELITE_POOL.sort(key=lambda x: getattr(x[0], 'fitness', 0))
                ELITE_POOL[0] = (genome, config)
                print(f"[ELITE] Replaced with genome fitness {avg_fitness:.2f}")
        
        return avg_fitness
        
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        traceback.print_exc()
        return -100.0

def run_expert_training(config_path, generations=100, episodes=3, population=150):
    """Run expert training with staged learning"""
    global TRAINING_STAGE, GENERATION_COUNTER
    
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    
    # Create output directory
    output_dir = Path("models/expert")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize population
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    
    print(f"\n{'='*60}")
    print(f"EXPERT TRAINING - STAGED LEARNING")
    print(f"{'='*60}")
    print(f"Stage 1: Ball Contact Focus")
    print(f"Stage 2: Positioning + Ball Contact")
    print(f"Stage 3: Winning Focus")
    print(f"Population: {population}")
    print(f"Episodes per eval: {episodes}")
    print(f"{'='*60}\n")
    
    best_genome = None
    
    for generation in range(generations):
        GENERATION_COUNTER = generation
        
        # Evaluate population
        print(f"\n[Generation {generation}] Stage {TRAINING_STAGE}")
        
        for genome_id, genome in pop.population.items():
            genome.fitness = evaluate_genome_staged(genome, config, TRAINING_STAGE, episodes)
        
        # Get statistics
        fitnesses = [g.fitness for g in pop.population.values()]
        best_fitness = max(fitnesses)
        avg_fitness = np.mean(fitnesses)
        
        # Track hit rate
        hit_rates = [getattr(g, 'hits_per_game', 0) for g in pop.population.values()]
        avg_hits = np.mean(hit_rates) if hit_rates else 0
        
        # Track win rate
        win_rates = [getattr(g, 'win_rate', 0) for g in pop.population.values()]
        avg_win_rate = np.mean(win_rates) if win_rates else 0
        
        print(f"Best Fitness: {best_fitness:.2f}")
        print(f"Avg Fitness: {avg_fitness:.2f}")
        print(f"Avg Hits/Game: {avg_hits:.2f}")
        print(f"Avg Win Rate: {avg_win_rate*100:.1f}%")
        
        # Stage transition logic
        if TRAINING_STAGE == 1 and avg_hits > 0.5:
            print(f"\n🎯 STAGE TRANSITION: Moving to Stage 2 (Positioning)")
            TRAINING_STAGE = 2
        elif TRAINING_STAGE == 2 and avg_hits > 2.0:
            print(f"\n🎯 STAGE TRANSITION: Moving to Stage 3 (Winning)")
            TRAINING_STAGE = 3
        
        # Save best genome
        best_genome = max(pop.population.values(), key=lambda g: g.fitness)
        
        # Save checkpoint every 10 generations
        if generation % 10 == 0:
            checkpoint_path = output_dir / f"checkpoint_gen{generation}.pkl"
            with open(checkpoint_path, 'wb') as f:
                pickle.dump((best_genome, config), f)
            print(f"[SAVE] Checkpoint saved: {checkpoint_path}")
        
        # Check for expert level
        if avg_win_rate > 0.75 and avg_hits > 3.0:
            print(f"\n🏆 EXPERT LEVEL ACHIEVED!")
            print(f"Win Rate: {avg_win_rate*100:.1f}%")
            print(f"Hits/Game: {avg_hits:.2f}")
            break
        
        # Early stopping if stuck
        if generation > 50 and avg_fitness < 10:
            print(f"\n⚠️ Training stuck, consider adjusting parameters")
        
        # Create next generation
        pop.population = pop.reproduction.reproduce(config, pop.species, 
                                                   pop.config.pop_size, generation)
        pop.species.speciate(config, pop.population, generation)
    
    # Save final best genome
    if best_genome:
        final_path = output_dir / "best_genome.pkl"
        with open(final_path, 'wb') as f:
            pickle.dump((best_genome, config), f)
        print(f"\n[FINAL] Best genome saved: {final_path}")
        print(f"Final Fitness: {best_genome.fitness:.2f}")
        print(f"Final Hits/Game: {getattr(best_genome, 'hits_per_game', 0):.2f}")
        print(f"Final Win Rate: {getattr(best_genome, 'win_rate', 0)*100:.1f}%")
    
    return best_genome

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expert NEAT Training")
    parser.add_argument("--config", type=str, default="neat_config_expert.cfg")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--population", type=int, default=150)
    args = parser.parse_args()
    
    run_expert_training(args.config, args.generations, args.episodes, args.population)