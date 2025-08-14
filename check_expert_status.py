#!/usr/bin/env python3
"""
Quick Expert Status Checker
Tests model against key performance metrics
"""

import sys
import pickle
import numpy as np
import gym
import neat
from pathlib import Path

try:
    import slimevolleygym
except ImportError:
    print("ERROR: slimevolleygym required")
    sys.exit(1)

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

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
    norm_obs = np.copy(obs)
    norm_obs[0] = np.tanh(norm_obs[0] * 2.0)
    norm_obs[1] = np.tanh(norm_obs[1] * 2.0)
    norm_obs[4] = np.tanh(norm_obs[4] * 2.0)
    norm_obs[8] = np.tanh(norm_obs[8] * 2.0)
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
    return (out[:3] > 0.0).astype(np.int8)

def quick_evaluate(model_path, episodes=10):
    """Quick evaluation for expert status"""
    
    # Load model
    with open(model_path, 'rb') as f:
        genome, config = pickle.load(f)
    
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    print(f"\n{'='*60}")
    print(f"EXPERT STATUS CHECK")
    print(f"Model: {model_path}")
    print(f"Testing {episodes} episodes...")
    print(f"{'='*60}\n")
    
    # Metrics
    wins = 0
    total_points_scored = 0
    total_points_conceded = 0
    total_ball_hits = 0
    total_reaction_times = []
    episodes_with_hits = 0
    
    for ep in range(episodes):
        env = make_env()
        obs = obs_to_np(env.reset())
        
        done = False
        steps = 0
        ball_hits = 0
        points_scored = 0
        points_conceded = 0
        reaction_timer = 0
        
        last_ball_y = obs[1]
        last_ball_x = obs[0]
        
        while not done and steps < 1000:
            action = action_from_output(env, net.activate(normalize_obs(obs)))
            next_obs, rew, done, info = step_env(env, action)
            next_obs = obs_to_np(next_obs)
            
            # Track ball hits
            if last_ball_y > 0.5 and next_obs[1] < 0.5:
                if abs(next_obs[0] - next_obs[4]) < 0.15:
                    ball_hits += 1
                    if reaction_timer > 0:
                        total_reaction_times.append(reaction_timer)
            
            # Track reaction time
            if next_obs[0] > 0 and last_ball_x <= 0:
                reaction_timer = 1
            elif next_obs[0] > 0:
                reaction_timer += 1
            
            # Track scoring
            if rew > 0:
                points_scored += 1
            elif rew < 0:
                points_conceded += 1
            
            last_ball_y = next_obs[1]
            last_ball_x = next_obs[0]
            obs = next_obs
            steps += 1
        
        # Episode results
        if points_scored > points_conceded:
            wins += 1
            result = "WIN"
        elif points_scored < points_conceded:
            result = "LOSS"
        else:
            result = "DRAW"
        
        if ball_hits > 0:
            episodes_with_hits += 1
        
        total_points_scored += points_scored
        total_points_conceded += points_conceded
        total_ball_hits += ball_hits
        
        print(f"Episode {ep+1:2d}: {result} | Score: {points_scored}-{points_conceded} | Hits: {ball_hits}")
        
        env.close()
    
    # Calculate metrics
    win_rate = (wins / episodes) * 100
    avg_hits = total_ball_hits / episodes
    hit_success_rate = (episodes_with_hits / episodes) * 100
    avg_reaction = np.mean(total_reaction_times) if total_reaction_times else 10.0
    point_diff = total_points_scored - total_points_conceded
    
    print(f"\n{'='*60}")
    print(f"PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(f"Win Rate:           {win_rate:.1f}% ({wins}/{episodes})")
    print(f"Points Scored:      {total_points_scored}")
    print(f"Points Conceded:    {total_points_conceded}")
    print(f"Point Differential: {point_diff:+d}")
    print(f"Avg Hits/Game:      {avg_hits:.2f}")
    print(f"Hit Success Rate:   {hit_success_rate:.1f}%")
    print(f"Avg Reaction Time:  {avg_reaction:.1f} steps")
    
    # Expert criteria
    print(f"\n{'='*60}")
    print(f"EXPERT CRITERIA")
    print(f"{'='*60}")
    
    criteria = {
        "Win Rate >= 75%": win_rate >= 75,
        "Avg Hits >= 3.0": avg_hits >= 3.0,
        "Reaction Time < 3": avg_reaction < 3.0,
        "Point Diff > 0": point_diff > 0,
        "Hit Success >= 80%": hit_success_rate >= 80
    }
    
    met = 0
    for criterion, passed in criteria.items():
        status = "✅" if passed else "❌"
        print(f"{status} {criterion}")
        if passed:
            met += 1
    
    print(f"\n{'='*60}")
    if met == len(criteria):
        print(f"🏆 EXPERT STATUS ACHIEVED!")
    elif met >= 4:
        print(f"⭐ NEAR EXPERT ({met}/{len(criteria)} criteria met)")
    elif met >= 3:
        print(f"✅ ADVANCED ({met}/{len(criteria)} criteria met)")
    elif met >= 2:
        print(f"📈 IMPROVING ({met}/{len(criteria)} criteria met)")
    else:
        print(f"💪 KEEP TRAINING ({met}/{len(criteria)} criteria met)")
    print(f"{'='*60}\n")
    
    return {
        'win_rate': win_rate,
        'avg_hits': avg_hits,
        'reaction_time': avg_reaction,
        'expert_level': met == len(criteria)
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--episodes", type=int, default=10, help="Number of test episodes")
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        print(f"ERROR: Model not found: {args.model}")
        sys.exit(1)
    
    quick_evaluate(args.model, args.episodes)