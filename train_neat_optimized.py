# train_neat_optimized.py - Optimized training for expert-level performance
import os
import sys
import pickle
import argparse
import traceback
import random
import time
from pathlib import Path
from collections import defaultdict, deque
import numpy as np
import gym
import neat
import cv2

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

# Boot info
print(f"[boot] python={sys.version.split()[0]} exe={sys.executable}", flush=True)
print(f"[boot] cwd={os.getcwd()} argv={sys.argv}", flush=True)
print(f"[boot] OPTIMIZED VERSION - Enhanced for expert-level training", flush=True)

# Import slimevolleygym
SLIME_OK = True
try:
    import slimevolleygym
    print("[import] slimevolleygym=OK", flush=True)
except Exception as e:
    print(f"[import] slimevolleygym=FAILED msg={e}", flush=True)
    SLIME_OK = False

SCRIPT_DIR = Path(__file__).resolve().parent
CWD = Path.cwd()

# Enhanced global tracking for better training
BEST_GENOMES = []
ELITE_GENOMES = []  # Top performers for elite competition
TARGET_FITNESS_MIN = 0.5
TARGET_FITNESS_MAX = 1.0
PERFORMANCE_HISTORY = defaultdict(list)  # Track detailed performance metrics
GENERATION_STATS = []

def make_env(render: bool = False):
    if not SLIME_OK:
        raise RuntimeError("slimevolleygym import failed")
    try:
        return gym.make("SlimeVolley-v0", render_mode="human" if render else None)
    except TypeError:
        return gym.make("SlimeVolley-v0")

def obs_to_np(obs):
    if isinstance(obs, tuple):
        obs = obs[0]
    return np.asarray(obs, dtype=np.float32)

def normalize_obs(obs):
    """Enhanced normalization for better generalization"""
    norm_obs = np.copy(obs)
    
    # More aggressive normalization for positions
    norm_obs[0] = np.clip(norm_obs[0] * 1.5, -1, 1)  # Ball x
    norm_obs[4] = np.clip(norm_obs[4] * 1.5, -1, 1)  # Player x
    norm_obs[8] = np.clip(norm_obs[8] * 1.5, -1, 1)  # Opponent x
    
    # Better velocity normalization
    for i in [2, 3, 6, 7, 10, 11]:
        norm_obs[i] = np.tanh(norm_obs[i] * 0.3)  # More sensitive to small velocities
        
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
    return (out[:3] > 0.0).astype(np.int8)

def flip_observation(obs):
    """Flip observation for opponent perspective"""
    flipped = np.copy(obs)
    flipped[0] = -flipped[0]  # ball x
    flipped[2] = -flipped[2]  # ball vx
    flipped[4], flipped[8] = -flipped[8], -flipped[4]  # x positions
    flipped[5], flipped[9] = flipped[9], flipped[5]    # y positions 
    flipped[6], flipped[10] = -flipped[10], -flipped[6]  # vx velocities
    flipped[7], flipped[11] = flipped[11], flipped[7]    # vy velocities
    return flipped

def evaluate_genome(genome, config, episodes=3, max_steps=1000, seed=None, overfit=False):
    """Enhanced evaluation with focus on ball contact and winning"""
    try:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness_total = 0.0
        detailed_metrics = {
            'wins': 0,
            'losses': 0,
            'total_ball_hits': 0,
            'total_rallies': 0,
            'avg_reaction_time': 0,
            'movement_efficiency': 0
        }
        
        # Elite competition - use best performers more often
        use_elite = len(ELITE_GENOMES) > 0 and random.random() < 0.4  # 40% elite competition
        use_self_play = random.random() < 0.2  # 20% self-play
        
        # CRITICAL: Never use overfit mode - it prevents learning!
        if overfit:
            print("WARNING: Overfit mode limits learning to single scenario!")
            print("WARNING: Disabling overfit mode for proper training.")
            overfit = False  # Force disable for real training

        for ep in range(episodes):
            env = make_env(render=False)
            original_policy = None
            
            # CRITICAL: Progressive difficulty with slower ramp
            if hasattr(env, 'opponent_difficulty'):
                try:
                    generation = getattr(genome, 'generation', 0)
                    if generation < 20:
                        # First 20 generations: very easy opponent
                        curriculum_level = 0.1 + 0.1 * (generation / 20)
                    elif generation < 50:
                        # Next 30 generations: gradual increase
                        curriculum_level = 0.2 + 0.3 * ((generation - 20) / 30)
                    elif generation < 100:
                        # Next 50 generations: moderate difficulty
                        curriculum_level = 0.5 + 0.3 * ((generation - 50) / 50)
                    else:
                        # After 100 generations: near full difficulty
                        curriculum_level = min(1.0, 0.8 + 0.2 * ((generation - 100) / 50))
                    
                    env.opponent_difficulty = curriculum_level
                    print(f"Gen {generation}: difficulty={curriculum_level:.2f}")
                except Exception as e:
                    print(f"Curriculum setup failed: {e}")
            
            # Setup competition type
            if use_elite and ELITE_GENOMES:
                try:
                    if hasattr(env, 'policy'):
                        original_policy = env.policy
                        opponent_genome = random.choice(ELITE_GENOMES)
                        opponent_net = neat.nn.FeedForwardNetwork.create(opponent_genome, config)
                        
                        def elite_policy(obs):
                            flipped_obs = flip_observation(obs)
                            return action_from_output(env, opponent_net.activate(normalize_obs(flipped_obs)))
                        
                        env.policy = elite_policy
                        print("Using elite competition")
                except Exception as e:
                    print(f"Elite setup failed: {e}")
            elif use_self_play:
                try:
                    if hasattr(env, 'policy'):
                        original_policy = env.policy
                        
                        def self_play_policy(obs):
                            flipped_obs = flip_observation(obs)
                            return action_from_output(env, net.activate(normalize_obs(flipped_obs)))
                        
                        env.policy = self_play_policy
                        print("Using self-play")
                except Exception as e:
                    print(f"Self-play setup failed: {e}")
            
            # Reset environment
            obs = None
            if seed is not None:
                s = int(seed + ep)
                try:
                    obs0 = env.reset(seed=s)
                    obs = obs_to_np(obs0)
                except TypeError:
                    try:
                        if hasattr(env, "seed"):
                            env.seed(s)
                        if hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
                            env.action_space.seed(s)
                        if hasattr(env, "observation_space") and hasattr(env.observation_space, "seed"):
                            env.observation_space.seed(s)
                    except Exception:
                        pass
            
            if obs is None:
                obs = obs_to_np(env.reset())
            
            # Game loop with ENHANCED reward shaping
            ep_rew = 0.0
            steps = 0
            done = False
            
            # Enhanced tracking
            ball_hits = 0
            attempted_hits = 0
            reaction_times = []
            ball_on_our_side_timer = 0
            successful_rallies = 0
            
            last_ball_y = obs[1]
            last_player_x = obs[4]
            last_ball_x = obs[0]
            
            max_x_position = obs[4]
            min_x_position = obs[4]
            
            # Action history for pattern detection
            action_history = deque(maxlen=20)
            movement_directions = []
            jumps_near_ball = 0
            wasted_jumps = 0

            while not done and steps < max_steps:
                # Get agent's action
                action = action_from_output(env, net.activate(normalize_obs(obs)))
                action_history.append(action)
                
                # Take step
                next_obs, rew, done, info = step_env(env, action)
                next_obs = obs_to_np(next_obs)
                
                # CRITICAL: Amplify game score much more
                shaped_reward = float(rew) * 50.0  # 50x amplification for win/loss
                
                # Track ball position
                ball_y = next_obs[1]
                ball_x = next_obs[0]
                player_x = next_obs[4]
                
                # ENHANCED BALL HIT DETECTION AND REWARD
                if last_ball_y > 0.5 and ball_y < 0.5:
                    distance_to_ball = abs(ball_x - player_x)
                    
                    if distance_to_ball < 0.1:
                        # Direct hit - MASSIVE reward
                        ball_hits += 1
                        shaped_reward += 50.0  # Huge reward for hitting
                        
                        if action[2] > 0:  # Jumped to hit
                            shaped_reward += 20.0
                            jumps_near_ball += 1
                        
                        # Reaction time bonus
                        if ball_on_our_side_timer > 0 and ball_on_our_side_timer < 5:
                            shaped_reward += 30.0 / ball_on_our_side_timer
                            reaction_times.append(ball_on_our_side_timer)
                        
                        successful_rallies += 1
                        print(f"HIT! Hits={ball_hits}, Distance={distance_to_ball:.3f}")
                        
                    elif distance_to_ball < 0.25:
                        # Near miss - still reward for being close
                        shaped_reward += 15.0
                        attempted_hits += 1
                        
                        if action[2] > 0:  # Tried to jump
                            shaped_reward += 5.0
                    
                    elif distance_to_ball < 0.4:
                        # In vicinity - small reward
                        shaped_reward += 5.0
                        attempted_hits += 1
                
                # Track reaction time
                if ball_x > 0 and last_ball_x <= 0:
                    ball_on_our_side_timer = 1
                elif ball_x > 0:
                    ball_on_our_side_timer += 1
                else:
                    ball_on_our_side_timer = 0
                
                # MOVEMENT REWARDS - Encourage active play
                distance_to_ball = abs(ball_x - player_x)
                
                # Strong reward for moving toward ball
                if ball_x > player_x and action[1] > 0:  # Moving right toward ball
                    shaped_reward += 5.0
                    movement_directions.append('correct')
                elif ball_x < player_x and action[0] > 0:  # Moving left toward ball
                    shaped_reward += 5.0
                    movement_directions.append('correct')
                elif abs(player_x - last_player_x) > 0.01:
                    shaped_reward += 1.0  # Any movement is better than standing
                    movement_directions.append('neutral')
                else:
                    shaped_reward -= 1.0  # Penalize standing still
                    movement_directions.append('static')
                
                # Position quality rewards
                if ball_x > 0:  # Ball on our side
                    position_reward = 10.0 * (1.0 - min(1.0, distance_to_ball))
                    shaped_reward += position_reward
                else:  # Ball on opponent side - defensive position
                    center_distance = abs(player_x - 0.7)
                    shaped_reward += 3.0 * (1.0 - min(1.0, center_distance))
                
                # Jump accuracy tracking
                if action[2] > 0:
                    if distance_to_ball < 0.2 and ball_y < 0.6:
                        shaped_reward += 10.0  # Good jump timing
                    elif distance_to_ball > 0.4 or ball_y > 0.8:
                        shaped_reward -= 2.0  # Wasted jump
                        wasted_jumps += 1
                
                # Predictive positioning reward
                if ball_y > 0.3 and ball_y < 0.7 and ball_x > 0:
                    # Calculate trajectory
                    if last_ball_y != ball_y and ball_y < last_ball_y:  # Ball descending
                        try:
                            y_vel = ball_y - last_ball_y
                            x_vel = ball_x - last_ball_x
                            time_to_ground = (0.1 - ball_y) / abs(y_vel) if y_vel < 0 else 5
                            predicted_x = ball_x + x_vel * time_to_ground
                            
                            if 0 < predicted_x < 2:
                                prediction_error = abs(player_x - predicted_x)
                                if prediction_error < 0.2:
                                    shaped_reward += 15.0  # Excellent prediction
                                elif prediction_error < 0.4:
                                    shaped_reward += 5.0   # Good prediction
                        except:
                            pass
                
                # Rally continuation bonus
                if steps > 30 and not done:
                    shaped_reward += 0.1 * (steps / 30)  # Progressive rally bonus
                
                # Ball return success
                if ball_hits > 0 and ball_x < 0 and last_ball_x > 0:
                    shaped_reward += 20.0  # Successfully returned ball
                
                # Update tracking
                min_x_position = min(min_x_position, player_x)
                max_x_position = max(max_x_position, player_x)
                
                last_ball_y = ball_y
                last_player_x = player_x
                last_ball_x = ball_x
                obs = next_obs
                ep_rew += shaped_reward
                steps += 1
            
            # Episode completion bonuses
            if float(rew) > 0:  # Won the point
                ep_rew += 200.0  # Massive win bonus
                detailed_metrics['wins'] += 1
                print(f"WIN! Episode {ep+1}, Hits={ball_hits}")
            elif float(rew) < 0:
                detailed_metrics['losses'] += 1
                if ball_hits > 0:
                    ep_rew += 20.0 * ball_hits  # Consolation for trying
                    print(f"Loss but hit ball {ball_hits} times")
            
            # Movement assessment
            movement_range = max_x_position - min_x_position
            if movement_range < 0.3:
                ep_rew -= 20.0  # Penalty for being too static
                print(f"Static penalty: range={movement_range:.2f}")
            elif movement_range > 0.6:
                ep_rew += 10.0  # Bonus for good court coverage
            
            # Skill bonuses
            if ball_hits >= 3:
                ep_rew += 50.0  # Excellent ball control
            if jumps_near_ball > wasted_jumps:
                ep_rew += 20.0  # Good jump accuracy
            if reaction_times and np.mean(reaction_times) < 3:
                ep_rew += 30.0  # Fast reactions
            
            # Update detailed metrics
            detailed_metrics['total_ball_hits'] += ball_hits
            detailed_metrics['total_rallies'] += successful_rallies
            if reaction_times:
                detailed_metrics['avg_reaction_time'] += np.mean(reaction_times)
            if movement_directions:
                correct_moves = movement_directions.count('correct')
                detailed_metrics['movement_efficiency'] += correct_moves / len(movement_directions)
            
            # Restore original policy
            if original_policy is not None:
                try:
                    env.policy = original_policy
                except Exception:
                    pass
            
            fitness_total += ep_rew
            env.close()
        
        # Calculate final fitness
        avg_fitness = fitness_total / float(episodes)
        
        # Track detailed performance
        PERFORMANCE_HISTORY[genome.key].append({
            'fitness': avg_fitness,
            'metrics': detailed_metrics,
            'generation': getattr(genome, 'generation', 0)
        })
        
        # Update elite genomes (top performers)
        if avg_fitness > 100:  # High performance threshold
            if len(ELITE_GENOMES) >= 10:
                # Replace weakest elite
                weakest_idx = min(range(len(ELITE_GENOMES)), 
                                key=lambda i: getattr(ELITE_GENOMES[i], 'fitness', 0))
                weakest_fitness = getattr(ELITE_GENOMES[weakest_idx], 'fitness', 0)
                if avg_fitness > weakest_fitness:
                    genome.fitness = avg_fitness  # Ensure fitness is set
                    ELITE_GENOMES[weakest_idx] = genome
            else:
                genome.fitness = avg_fitness  # Ensure fitness is set
                ELITE_GENOMES.append(genome)
            print(f"Elite genome added: fitness={avg_fitness:.2f}")
        
        # Regular competition pool
        if TARGET_FITNESS_MIN <= avg_fitness <= TARGET_FITNESS_MAX:
            if len(BEST_GENOMES) >= 5:
                BEST_GENOMES.pop(0)
            BEST_GENOMES.append(genome)
        
        return float(avg_fitness)
        
    except Exception as e:
        print(f"Error evaluating genome: {e}")
        traceback.print_exc()
        return -10.0

def eval_genomes(genomes, config, episodes, max_steps, seed):
    for genome_id, genome in genomes:
        fitness = evaluate_genome(genome, config, episodes, max_steps, seed)
        # Ensure fitness is always a valid number
        if fitness is None or not isinstance(fitness, (int, float)):
            fitness = -10.0
        genome.fitness = float(fitness)
        # Track generation
        genome.generation = getattr(genome, 'generation', 0)

def resolve_config_path(user_path: str) -> Path:
    p = Path(user_path)
    if p.is_file():
        return p.resolve()
    p2 = (CWD / user_path).resolve()
    if p2.is_file():
        return p2
    p3 = (SCRIPT_DIR / user_path).resolve()
    if p3.is_file():
        return p3
    raise FileNotFoundError(f"Config not found at '{user_path}'")

def evaluate_genome_with_args(genome, config, *, episodes=3, max_steps=1000, seed=None, overfit=False):
    try:
        # Track generation for curriculum
        genome.generation = getattr(genome, 'generation', 0)
        # NEVER use overfit in real training
        fitness = evaluate_genome(genome, config, episodes, max_steps, seed, overfit=False)
        if fitness is None or not isinstance(fitness, (int, float)) or np.isnan(fitness):
            return -10.0
        return float(fitness)
    except Exception as e:
        print(f"Parallel evaluation error: {e}")
        return -10.0

def save_generation_stats(generation, population, outdir):
    """Save statistics for each generation"""
    fitnesses = [g.fitness for g in population.values() if g.fitness is not None]
    if not fitnesses:
        return
    
    stats = {
        'generation': generation,
        'best_fitness': max(fitnesses),
        'avg_fitness': np.mean(fitnesses),
        'std_fitness': np.std(fitnesses),
        'min_fitness': min(fitnesses),
        'population_size': len(fitnesses)
    }
    
    # Count wins if tracked
    wins = sum(1 for g in population.values() 
              if g.key in PERFORMANCE_HISTORY and 
              PERFORMANCE_HISTORY[g.key] and 
              PERFORMANCE_HISTORY[g.key][-1]['metrics']['wins'] > 0)
    
    stats['agents_with_wins'] = wins
    GENERATION_STATS.append(stats)
    
    # Save to file
    stats_file = outdir / "generation_stats.pkl"
    with open(stats_file, "wb") as f:
        pickle.dump(GENERATION_STATS, f)
    
    print(f"\n[Gen {generation}] Best: {stats['best_fitness']:.2f}, "
          f"Avg: {stats['avg_fitness']:.2f}, Winners: {wins}/{len(fitnesses)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overfit", action="store_true", help="Overfit on single episode")
    parser.add_argument("--config", type=str, default="neat_config_slime.cfg")
    parser.add_argument("--generations", type=int, default=200)  # More generations
    parser.add_argument("--episodes", type=int, default=5)  # More episodes per eval
    parser.add_argument("--max-steps", type=int, default=1500)  # Longer episodes
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--outdir", type=str, default="models/optimized")
    parser.add_argument("--checkpoint", type=str, help="Resume from checkpoint")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"OPTIMIZED NEAT TRAINING FOR EXPERT PERFORMANCE")
    print(f"{'='*60}")
    print(f"Generations: {args.generations}")
    print(f"Episodes per eval: {args.episodes}")
    print(f"Workers: {args.workers}")
    print(f"Output: {args.outdir}")
    print(f"{'='*60}\n")

    cfg_path = resolve_config_path(args.config)
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = (CWD / outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_prefix = outdir / "chkpt-"

    # Test environment
    try:
        env = make_env(render=False)
        ob0 = obs_to_np(env.reset())
        print(f"[env] obs_shape={getattr(ob0, 'shape', None)}")
        env.close()
    except Exception as e:
        print(f"[env] probe failed: {e}")
        return

    try:
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(cfg_path),
        )

        # Load or create population
        if args.checkpoint and os.path.exists(args.checkpoint):
            print(f"Resuming from: {args.checkpoint}")
            pop = neat.Checkpointer.restore_checkpoint(args.checkpoint)
            
            # Load elite genomes
            elite_path = outdir / "elite_genomes.pkl"
            if elite_path.exists():
                try:
                    with open(elite_path, "rb") as f:
                        ELITE_GENOMES = pickle.load(f)
                    print(f"Loaded {len(ELITE_GENOMES)} elite genomes")
                except Exception as e:
                    print(f"Error loading elite genomes: {e}")
        else:
            pop = neat.Population(config)
        
        # Add reporters
        pop.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)
        pop.add_reporter(neat.Checkpointer(args.checkpoint_every, filename_prefix=str(ckpt_prefix)))

        # Custom fitness function that tracks generation
        def run_generation(genomes, config):
            # Update generation counter for all genomes
            current_gen = pop.generation if hasattr(pop, 'generation') else 0
            for _, genome in genomes:
                genome.generation = current_gen
            
            # Evaluate
            eval_genomes(genomes, config, args.episodes, args.max_steps, args.seed)
            
            # Save generation stats
            save_generation_stats(current_gen, pop.population, outdir)

        # Run evolution
        if args.workers > 1:
            from functools import partial
            eval_function = partial(
                evaluate_genome_with_args,
                episodes=args.episodes,
                max_steps=args.max_steps,
                seed=args.seed,
                overfit=False  # ALWAYS False for real training
            )
            pe = neat.ParallelEvaluator(args.workers, eval_function)
            
            # Custom run with generation tracking
            for generation in range(args.generations):
                # Update generation for all genomes
                for genome in pop.population.values():
                    genome.generation = generation
                
                # Evaluate
                pop.run(pe.evaluate, 1)
                
                # Save stats
                save_generation_stats(generation, pop.population, outdir)
                
                # Save elite genomes periodically
                if generation % 10 == 0 and ELITE_GENOMES:
                    elite_path = outdir / "elite_genomes.pkl"
                    with open(elite_path, "wb") as f:
                        pickle.dump(ELITE_GENOMES, f)
                    print(f"Saved {len(ELITE_GENOMES)} elite genomes")
        else:
            winner = pop.run(run_generation, args.generations)

        # Save final results
        winner = max(pop.population.values(), key=lambda g: g.fitness)
        winner_path = outdir / "winner.pkl"
        with open(winner_path, "wb") as f:
            pickle.dump((winner, config), f)
        print(f"\n[FINAL] Winner fitness: {winner.fitness:.2f}")
        print(f"[save] winner_path={winner_path}")

        # Save top performers
        top_genomes = sorted(
            [(g.fitness, i, g) for i, g in enumerate(pop.population.values())],
            key=lambda x: x[0],
            reverse=True
        )[:5]
        
        for i, (fitness, _, genome) in enumerate(top_genomes):
            top_path = outdir / f"top{i+1}_genome.pkl"
            with open(top_path, "wb") as f:
                pickle.dump((genome, config), f)
            print(f"[save] Top {i+1}: fitness={fitness:.2f}")

        # Save elite genomes
        if ELITE_GENOMES:
            elite_path = outdir / "elite_genomes.pkl"
            with open(elite_path, "wb") as f:
                pickle.dump(ELITE_GENOMES, f)
            print(f"[save] {len(ELITE_GENOMES)} elite genomes")

        # Save performance history
        history_path = outdir / "performance_history.pkl"
        with open(history_path, "wb") as f:
            pickle.dump(dict(PERFORMANCE_HISTORY), f)
        print(f"[save] Performance history")

        print(f"\n{'='*60}")
        print("TRAINING COMPLETE - Run evaluation to check performance")
        print(f"python evaluate_performance.py --model {winner_path} --optimize")
        print(f"{'='*60}")

    except Exception as e:
        print(f"[error] Training failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()