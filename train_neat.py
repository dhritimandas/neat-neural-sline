# train_neat.py
import os
import sys
import pickle
import argparse
import traceback
import random
from pathlib import Path
from collections import defaultdict, deque
import numpy as np
import gym
import neat
import cv2

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

# Boot info printed ASAP
print(f"[boot] python={sys.version.split()[0]} exe={sys.executable}", flush=True)
print(f"[boot] cwd={os.getcwd()} argv={sys.argv}", flush=True)
print(f"[boot] __file__={__file__}", flush=True)

# Try to import slimevolleygym early to see errors (cv2, etc.)
SLIME_OK = True  # Set to True to bypass the check
try:
    import slimevolleygym  # noqa: F401
    print("[import] slimevolleygym=OK", flush=True)
except Exception as e:
    print(f"[import] slimevolleygym=FAILED msg={e}", flush=True)

SCRIPT_DIR = Path(__file__).resolve().parent
CWD = Path.cwd()

# Store best genomes for cross-species competition
BEST_GENOMES = []
TARGET_FITNESS_MIN = 0.5
TARGET_FITNESS_MAX = 1.0

def make_env(render: bool = False):
    if not SLIME_OK:
        raise RuntimeError("slimevolleygym import failed; install its deps (opencv-python).")
    try:
        return gym.make("SlimeVolley-v0", render_mode="human" if render else None)
    except TypeError:
        return gym.make("SlimeVolley-v0")

def obs_to_np(obs):
    if isinstance(obs, tuple):
        obs = obs[0]
    return np.asarray(obs, dtype=np.float32)

def normalize_obs(obs):
    """Normalize observations to improve generalization"""
    # Copy to avoid modifying original
    norm_obs = np.copy(obs)
    
    # Normalize positions to [-1, 1] range
    norm_obs[0] = np.clip(norm_obs[0] * 1.2, -1, 1)  # Ball x
    norm_obs[4] = np.clip(norm_obs[4] * 1.2, -1, 1)  # Player x
    norm_obs[8] = np.clip(norm_obs[8] * 1.2, -1, 1)  # Opponent x
    
    # Normalize velocities
    for i in [2, 3, 6, 7, 10, 11]:  # All velocity components
        norm_obs[i] = np.tanh(norm_obs[i] * 0.5)  # Squash to [-1, 1]
        
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
    # Always return 3-bit vector [left, right, jump] expected by slimevolleygym
    return (out[:3] > 0.0).astype(np.int8)


def flip_observation(obs):
    """Flip observation for opponent perspective"""
    flipped = np.copy(obs)
    # Swap ball positions
    flipped[0] = -flipped[0]  # ball x
    flipped[2] = -flipped[2]  # ball vx
    # Swap player (4-7) and opponent (8-11) data
    flipped[4], flipped[8] = -flipped[8], -flipped[4]  # x positions
    flipped[5], flipped[9] = flipped[9], flipped[5]    # y positions 
    flipped[6], flipped[10] = -flipped[10], -flipped[6]  # vx velocities
    flipped[7], flipped[11] = flipped[11], flipped[7]    # vy velocities
    return flipped

def evaluate_genome(genome, config, episodes=3, max_steps=1000, seed=None, overfit=False):
    try:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness_total = 0.0
        
        # If there are good genomes in the population, use for competition
        use_competition = len(BEST_GENOMES) > 0 and random.random() < 0.25
        
        if overfit:
            episodes = 1
            seed = 42

        for ep in range(episodes):
            env = make_env(render=False)
            original_policy = None
            
            # Set curriculum difficulty if supported
            if hasattr(env, 'opponent_difficulty'):
                try:
                    # Get generation safely
                    generation = getattr(genome, 'generation', 0)
                    if generation < 10:
                        # First 10 generations, focus on basic ball contact
                        curriculum_level = 0.1
                    elif generation < 25:
                        # Next 15 generations, slightly harder
                        curriculum_level = 0.2 + (0.3 * (generation - 10) / 15)
                    else:
                        # Standard progression
                        curriculum_level = min(1.0, 0.5 + 0.5 * (generation / 50.0))
                    env.opponent_difficulty = curriculum_level
                except Exception as e:
                    print(f"Curriculum setup failed: {e}")
            
            # Try to learn from opponent behavior (separate from the evaluation)
            if random.random() < 0.2:  # 20% chance per episode
                try:
                    print("Learning from opponent for this episode")
                    temp_env = make_env(render=False)
                    temp_obs = obs_to_np(temp_env.reset())
                    
                    for _ in range(random.randint(30, 50)):
                        if hasattr(temp_env, 'policy'):
                            if callable(temp_env.policy):
                                opponent_action = temp_env.policy(temp_obs)
                            elif hasattr(temp_env.policy, 'predict'):
                                opponent_action = temp_env.policy.predict(temp_obs)
                            else:
                                raise AttributeError("Cannot determine how to call opponent policy")
                            # Take step with opponent's action
                            next_temp_obs, _, done, _ = step_env(temp_env, opponent_action)
                            if done:
                                break
                            temp_obs = obs_to_np(next_temp_obs)
                    temp_env.close()
                except Exception as e:
                    print(f"Opponent learning failed: {e}")
            
            # Setup cross-species competition if enabled
            if use_competition and len(BEST_GENOMES) > 0:
                try:
                    # Save original policy
                    if hasattr(env, 'policy'):
                        original_policy = env.policy
                        
                        # Create opponent from a good genome
                        opponent_genome = random.choice(BEST_GENOMES)
                        opponent_net = neat.nn.FeedForwardNetwork.create(opponent_genome, config)
                        
                        # Create custom policy function
                        def custom_policy(obs):
                            flipped_obs = flip_observation(obs)
                            return action_from_output(env, opponent_net.activate(normalize_obs(flipped_obs)))
                        
                        # Replace policy
                        env.policy = custom_policy
                        print("Using cross-species competition")
                except Exception as e:
                    print(f"Cross-species setup failed: {e}")
                    original_policy = None
            
            # Reset environment with seed if provided
            obs = None
            if seed is not None:
                s = int(seed + ep)
                try:
                    obs0 = env.reset(seed=s)  # newer gym API
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
            
            # Game loop with reward shaping
            ep_rew = 0.0
            steps = 0
            done = False
            
            # Track metrics for reward shaping
            last_actions = [None, None, None]  # Remember last 3 actions
            last_episode_reward = 0.0
            # Track successful action sequences for learning
            action_history = deque(maxlen=10)
            successful_sequence = False
            ball_hits = 0
            last_ball_y = obs[1]  # Ball Y position
            last_player_x = obs[4]  # Player X position
            last_ball_x = obs[0]  # Ball X position
            
            max_x_position = obs[4]  # Initialize max player position
            min_x_position = obs[4]

            while not done and steps < max_steps:
                # Get agent's action from neural network (use normalized observation)
                action = action_from_output(env, net.activate(normalize_obs(obs)))
                
                # Take step in environment
                next_obs, rew, done, info = step_env(env, action)
                next_obs = obs_to_np(next_obs)
                # Save the original game reward for episodic bonuses
                last_episode_reward = float(rew)
                # BASE REWARD: Direct game score (+1 for winning point, -1 for losing point)
                shaped_reward = float(rew) * 10.0 # Amplify game score impact by 10x
                
                # REWARD SHAPING:
                # 1. Reward for hitting the ball
                ball_y = next_obs[1]
                if last_ball_y > 0.5 and ball_y < 0.5 and abs(next_obs[0] - next_obs[4]) < 0.1:
                    ball_hits += 1
                    successful_sequence = True
                    shaped_reward += 10.0  # Increased bonus for any ball contact

                    # Bonus for successful action sequences
                    if len(action_history) >= 5:
                        shaped_reward += 10.0  # Extra reward for building up to this hit

                    # Extra reward if the hit changes the ball's direction
                    if np.sign(next_obs[2]) != np.sign(obs[2]):  # Ball X velocity direction changed
                        shaped_reward += 3.0  # Reward for controlled hits

                # 2. Reward for positioning (staying close to ball x-position)
                ball_x = next_obs[0]
                player_x = next_obs[4]
                min_x_position = min(min_x_position, player_x)
                max_x_position = max(max_x_position, player_x)

                if action[2] > 0 and last_ball_y > 0.5 and ball_y < 0.5 and abs(ball_x - player_x) < 0.15:
                    shaped_reward += 5.0  # Substantial bonus for jumping and hitting in same step
                distance_to_ball = abs(ball_x - player_x)
                position_reward = 0.01 * (1.0 - min(1.0, distance_to_ball * 2))  
                shaped_reward += position_reward

                if abs(player_x - last_player_x) > 0.02:
                    shaped_reward += 0.005  # Reduced from previous values - minimal reward for just movement
                else:
                    shaped_reward -= 0.05  # Increased from 0.01 - penalize standing still more
                # Add directional movement reward to encourage moving toward ball
                if (ball_x > player_x and action[1] > 0) or (ball_x < player_x and action[0] > 0):
                    # Reward for moving in the direction of the ball
                    shaped_reward += 1  # Significantly higher reward for purposeful movement

                # Add "prediction" reward for positioning where the ball will be
                if ball_y > 0.5 and ball_x > 0 and last_ball_y != ball_y:       #Ball descending on player's side
                    # Calculate where ball will land based on trajectory
                    try:
                        y_ratio = (0.05 - ball_y) / (ball_y - last_ball_y) if ball_y != last_ball_y else 0
                        trajectory_x = ball_x + (ball_x - last_ball_x) * y_ratio
                        # Only apply when prediction is reasonable
                        if -2.5 < trajectory_x < 2.5:
                            # Reward moving toward predicted landing spot
                            prediction_reward = 0.5 * (1.0 - min(1.0, abs(player_x - trajectory_x)))
                            shaped_reward += prediction_reward
                    except:
                        pass  # Ignore errors in prediction calculation
                # Store the history of actions
                
                if steps > 0: # Only update history after first step
                    last_actions.pop(0)  # Remove oldest action
                    last_actions.append(action)  # Add newest action
                
                    # Store actions in history
                    action_history.append(action)
                
                # Reward for coordinated action sequences
                if steps > 3 and all(a is not None for a in last_actions):
                    # Check if agent has a pattern of appropriate actions based on ball position
                    if ball_y < 0.5:  # Ball is descending
                        distance_for_jump = 0.15 if ball_x < 0.5 else 0.3  # Jump sooner when ball is farther away
                        if distance_to_ball < distance_for_jump and action[2] > 0:  # Jump at appropriate distance
                            shaped_reward += 1.0  # Stronger reward for timing jumps right
                        elif distance_to_ball < 1.0 and action[2] == 0 and ball_y < 0.3:  # Should jump but didn't
                            shaped_reward -= 0.5  # Penalty for not jumping when should
                # Dynamic jump distance evaluation based on ball trajectory
                if ball_y > 0.3 and ball_y < 0.8 and ball_x > 0:  # Ball on way down on player's side
                    # Calculate approximate landing position
                    time_to_ground = ball_y / max(0.1, abs(next_obs[3]))  # Rough time estimate
                    landing_x = ball_x + next_obs[2] * time_to_ground  # Estimated x landing
                    
                    if 0 < landing_x < 2.0:  # Reachable landing spot
                        jump_distance = 0.1 + 0.1 * abs(landing_x - 1.0)  # Jump earlier when farther from center
                        if abs(player_x - landing_x) < jump_distance and action[2] > 0:
                            shaped_reward += 2.0  # Major reward for jumping at the right place and time                    
                if steps > 3 and all(a is not None for a in last_actions):
                    # Reward for moving consistently toward the ball
                    if ball_x > player_x and all(a[1] > 0 for a in last_actions):  # Moving right toward ball
                        shaped_reward += 0.1
                    elif ball_x < player_x and all(a[0] > 0 for a in last_actions):  # Moving left toward ball
                        shaped_reward += 0.1
                              
                # 3. Penalize unnecessary jumping (when ball is far away)
                if action[2] > 0 and distance_to_ball > 0.3 and ball_y > 0.5:
                    shaped_reward -= 0.02  # Small penalty for pointless jumps
                
                elif ball_x < -0.5:  # Ball is far on opponent side
                    # Reward returning to center court position when ball is away
                    center_reward = 0.05 * (1.0 - min(1.0, abs(player_x - 0.5)))
                    shaped_reward += center_reward
                # 4. Reward for hitting ball to opponent's side
                if ball_hits > 0 and ball_x < 0 and last_ball_x > 0:
                    shaped_reward += 1.0  # Good bonus for getting ball to opponent's side
                if ball_hits > 0 and ball_x < 0 and next_obs[3] > 0.2:  # Ball has positive y-velocity (downward)
                    shaped_reward += 0.3  # Bonus for making the ball harder to return
                # 5. Reward for successful defensive position
                if ball_x < 0:  # Ball on opponent side
                    defensive_reward = 0.05 * (1.0 - min(1.0, abs(player_x - 0.7)))  # Reward being near center-right
                    shaped_reward += defensive_reward
                
                # 6. Reward for aggressive play when opponent is off-position
                opponent_x = next_obs[8]  # Opponent X position
                if abs(opponent_x) > 0.6 and ball_x > 0:  # Opponent out of position and ball on our side
                    shaped_reward += 0.1  # Opportunity to score
                
                # 7. Reward for extended rallies (builds skill)
                if steps > 50:
                    shaped_reward += 0.001 * steps  # Small cumulative bonus for longer play
                if ball_y < 0.8 and steps > 10:  # Ball is still in play after some steps
                    shaped_reward += 0.01  # Small continuous reward for keeping rally going
                if ball_hits > 1:
                    # Reward for multiple successful hits in same episode
                    shaped_reward += 0.5 * ball_hits  # Progressive reward for continued success
                
                # Update for next step
                last_ball_y = ball_y
                last_player_x = player_x
                last_ball_x = ball_x
                obs = next_obs
                ep_rew += shaped_reward
                steps += 1

            if last_episode_reward > 0:  # Won the point
                ep_rew += 50.0  # Substantial bonus for winning
                print(f"Added win bonus for episode {ep+1}")
            elif ball_hits > 3 and steps > 200:
                # Played well even if lost
                ep_rew += 5.0  # Reward for putting up a good fight
                print(f"Added 'good effort' bonus for episode {ep+1}, hits={ball_hits}")
            if steps > 50 and abs(max_x_position - min_x_position) < 0.3:
                # Severe penalty for not exploring horizontal space
                ep_rew -= 5.0
                print(f"Applied movement range penalty for episode {ep+1}, range={max_x_position-min_x_position:.2f}")
            # Restore original policy if we used competition
            if original_policy is not None:
                try:
                    env.policy = original_policy
                except Exception:
                    pass
            
            # Extra reward for competing well against good opponents
            if use_competition and ep_rew > -2.0:
                ep_rew += 0.5  # Bonus for competitive performance
            
            fitness_total += ep_rew
            env.close()
        
        # Ensure we return a valid numeric value
        avg_fitness = fitness_total / float(episodes)
        
        # Check if this genome is good enough to be used for competition
        if TARGET_FITNESS_MIN <= avg_fitness <= TARGET_FITNESS_MAX:
            # Add to our collection if not already too many
            if len(BEST_GENOMES) >= 5:  # Limit collection size
                BEST_GENOMES.pop(0)  # Remove oldest
            BEST_GENOMES.append(genome)
            print(f"Added genome with fitness {avg_fitness:.2f} to competition pool (size: {len(BEST_GENOMES)})")
                
        return float(avg_fitness)  # Explicitly cast to float
    except Exception as e:
        print(f"Error evaluating genome: {e}")
        return -5.0  # Default bad fitness value if evaluation fails


def eval_genomes(genomes, config, episodes, max_steps, seed):
    for _, genome in genomes:
        genome.fitness = evaluate_genome(genome, config, episodes, max_steps, seed)

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
    raise FileNotFoundError(f"Config not found at '{user_path}', tried: {p}, {p2}, {p3}")

# Helper function for parallel evaluation
def evaluate_genome_with_args(genome, config, *, episodes=3, max_steps=1000, seed=None, overfit=False):
    try:
        fitness = evaluate_genome(genome, config, episodes, max_steps, seed, overfit)
        # Ensure we have a valid numeric value
        if fitness is None or not isinstance(fitness, (int, float)) or np.isnan(fitness):
            return -5.0
        return float(fitness)
    except Exception as e:
        print(f"Parallel evaluation error: {e}")
        return -5.0  # Default bad fitness for failed evaluations

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overfit", action="store_true", help="Overfit on a single episode with fixed seed")
    parser.add_argument("--config", type=str, default="neat_config_slime.cfg")
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--outdir", type=str, default="models")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file to resume from")
    args = parser.parse_args()

    print(f"[paths] cwd={CWD} script_dir={SCRIPT_DIR}", flush=True)
    cfg_path = resolve_config_path(args.config)
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = (CWD / outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_prefix = outdir / "chkpt-"
    print(f"[config] using={cfg_path}", flush=True)
    print(f"[output] outdir={outdir}", flush=True)
    print(f"[output] checkpoint_prefix={ckpt_prefix}", flush=True)

    # Probe env once
    try:
        env = make_env(render=False)
        ob0 = obs_to_np(env.reset())
        print(f"[env] obs_shape={getattr(ob0, 'shape', None)} action_space={env.action_space}", flush=True)
        env.close()
    except Exception as e:
        print(f"[env] probe failed: {e}", flush=True)
        traceback.print_exc()
        return

    try:
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(cfg_path),
        )

        if args.checkpoint and os.path.exists(args.checkpoint):
            print(f"Resuming from checkpoint: {args.checkpoint}")
            pop = neat.Checkpointer.restore_checkpoint(args.checkpoint)
            
            # Try to load best genomes from previous run
            best_genomes_path = outdir / "best_genomes.pkl"
            if best_genomes_path.exists():
                try:
                    with open(best_genomes_path, "rb") as f:
                        global BEST_GENOMES
                        BEST_GENOMES = pickle.load(f)
                    print(f"Loaded {len(BEST_GENOMES)} genomes for competition")
                except Exception as e:
                    print(f"Error loading competition genomes: {e}")
        else:
            pop = neat.Population(config)
            
        pop.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)
        pop.add_reporter(neat.Checkpointer(args.checkpoint_every, filename_prefix=str(ckpt_prefix)))

        if args.overfit:
            print("[Overfit] Training on a single fixed scenario with seed=42")

        # Create a function that remembers the args but can be pickled
        if args.workers > 1:   
            from functools import partial
            eval_function = partial(
                evaluate_genome_with_args, 
                episodes=args.episodes, 
                max_steps=args.max_steps, 
                seed=args.seed,
                overfit = args.overfit
            )
            pe = neat.ParallelEvaluator(args.workers, eval_function)
            winner = pop.run(pe.evaluate, args.generations)
        else:
            winner = pop.run(
                lambda gens, cfg: eval_genomes(gens, cfg, args.episodes, args.max_steps, args.seed),
                args.generations,
            )

        winner_path = outdir / "winner.pkl"
        
        with open(winner_path, "wb") as f:
            pickle.dump((winner, config), f)
        print(f"[save] winner_path={winner_path}", flush=True)

        best_genome = max([(g.fitness, i, g) for i, g in enumerate(pop.population.values())], key=lambda x: x[0])[2]
        best_path = outdir / "best_genome.pkl"
        with open(best_path, "wb") as f:
            pickle.dump((best_genome, config), f)
        
        top_genomes = sorted([(g.fitness, i, g) for i, g in enumerate(pop.population.values())], key=lambda x: x[0],  # Sort only by fitness
        reverse=True)[:3]
        for i, (fitness, _, genome) in enumerate(top_genomes):
            top_path = outdir / f"top{i+1}_genome.pkl"
            with open(top_path, "wb") as f:
                pickle.dump((genome, config), f)
            print(f"[save] Saved top {i+1} genome (fitness={fitness:.2f}) to {top_path}")

        if winner_path.exists():
            print(f"[save] exists=1 size_bytes={winner_path.stat().st_size}", flush=True)
        else:
            print("[save] exists=0", flush=True)

        # List outdir contents
        for p in sorted(outdir.iterdir()):
            try:
                sz = p.stat().st_size if p.is_file() else -1
            except Exception:
                sz = -1
            print(f"[outdir] {p.name} size={sz}", flush=True)

    except Exception as e:
        print(f"[error] training failed: {e}", flush=True)
        traceback.print_exc()
        print(f"[debug] outdir listing:", flush=True)
        try:
            for p in sorted(outdir.iterdir()):
                print(f"  - {p.name}", flush=True)
        except Exception:
            print("  (cannot list outdir)", flush=True)

if __name__ == "__main__":
    main()