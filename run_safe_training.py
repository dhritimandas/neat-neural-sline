#!/usr/bin/env python3
"""
Safe Training Launcher - Ensures optimal parameters for expert training
Prevents common mistakes that limit learning
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def validate_training_params(args):
    """Validate and fix common training parameter issues"""
    
    issues = []
    warnings = []
    
    # Check episodes
    if args.episodes < 2:
        issues.append(f"Episodes too low ({args.episodes}). Minimum 2 required for generalization.")
        args.episodes = 3
    elif args.episodes < 3:
        warnings.append(f"Low episodes ({args.episodes}). Recommend 3-5 for better learning.")
    
    # Check population
    if args.population < 50:
        issues.append(f"Population too small ({args.population}). Minimum 50 for diversity.")
        args.population = 100
    elif args.population > 500:
        warnings.append(f"Large population ({args.population}) will be slow. Consider 100-200.")
    
    # Check generations
    if args.generations < 20:
        warnings.append(f"Low generations ({args.generations}). May not reach expertise.")
    
    # Ensure overfit is NEVER used
    if args.overfit:
        issues.append("Overfit mode prevents learning! Disabling it.")
        args.overfit = False
    
    return issues, warnings, args

def run_training(training_type="expert"):
    """Run training with validated parameters"""
    
    parser = argparse.ArgumentParser(description="Safe NEAT Training Launcher")
    parser.add_argument("--type", choices=["expert", "optimized", "standard"], 
                       default="expert", help="Training type to use")
    parser.add_argument("--generations", type=int, default=100, 
                       help="Number of generations")
    parser.add_argument("--episodes", type=int, default=3, 
                       help="Episodes per evaluation")
    parser.add_argument("--population", type=int, default=150, 
                       help="Population size")
    parser.add_argument("--workers", type=int, default=1, 
                       help="Parallel workers (1 recommended for stability)")
    parser.add_argument("--overfit", action="store_true", 
                       help="NEVER USE - prevents learning")
    parser.add_argument("--quick", action="store_true", 
                       help="Quick test mode (10 gens)")
    
    args = parser.parse_args()
    
    # Quick mode adjustments
    if args.quick:
        args.generations = 10
        args.population = 50
        args.episodes = 2
        print("🚀 QUICK TEST MODE")
    
    # Validate parameters
    issues, warnings, args = validate_training_params(args)
    
    print("="*60)
    print("SAFE NEAT TRAINING LAUNCHER")
    print("="*60)
    
    # Report issues and warnings
    if issues:
        print("\n⚠️  CRITICAL ISSUES FIXED:")
        for issue in issues:
            print(f"  - {issue}")
    
    if warnings:
        print("\n⚡ WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
    
    # Display final parameters
    print("\n📊 TRAINING PARAMETERS:")
    print(f"  Type:        {args.type}")
    print(f"  Generations: {args.generations}")
    print(f"  Episodes:    {args.episodes}")
    print(f"  Population:  {args.population}")
    print(f"  Workers:     {args.workers}")
    print(f"  Overfit:     {args.overfit} (MUST BE False)")
    
    # Confirm parameters
    print("\n" + "="*60)
    response = input("Proceed with these parameters? (y/n): ").lower().strip()
    if response != 'y':
        print("Training cancelled.")
        return
    
    # Build command based on training type
    if args.type == "expert":
        script = "train_neat_expert.py"
        config = "neat_config_expert.cfg"
        output_dir = "models/expert"
    elif args.type == "optimized":
        script = "train_neat_optimized.py"
        config = "neat_config_optimized.cfg"
        output_dir = "models/optimized"
    else:
        script = "train_neat.py"
        config = "neat_config_slime.cfg"
        output_dir = "models"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable,
        script,
        "--config", config,
        "--generations", str(args.generations),
        "--episodes", str(args.episodes),
        "--workers", str(args.workers)
    ]
    
    # Add population for expert training
    if args.type == "expert":
        cmd.extend(["--population", str(args.population)])
    
    # NEVER add --overfit flag
    
    print(f"\n🚀 Starting {args.type} training...")
    print(f"Command: {' '.join(cmd)}")
    print(f"Output: {output_dir}/")
    print("\nPress Ctrl+C to stop training\n")
    print("="*60 + "\n")
    
    try:
        # Run training
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {e}")
    
    print("\n" + "="*60)
    print("Training session ended.")
    print(f"Check results in: {output_dir}/")
    
    # Offer to evaluate
    best_model = Path(output_dir) / "best_genome.pkl"
    if best_model.exists():
        print(f"\nModel saved: {best_model}")
        response = input("Run expert status check? (y/n): ").lower().strip()
        if response == 'y':
            subprocess.run([sys.executable, "check_expert_status.py", 
                          "--model", str(best_model), "--episodes", "10"])

if __name__ == "__main__":
    run_training()