#!/usr/bin/env python3
import wandb

print("Testing wandb access...")
api = wandb.Api()

try:
    run = api.run("/jackytu/jaxgcrl_benchmark/runs/ig3f8p5s")
    print(f"Successfully accessed run: {run.name}")
    print(f"Run state: {run.state}")
    print(f"Run duration: {run.summary.get('_runtime', 'N/A')} seconds")
    
    # Check for success metric in summary
    print("\nChecking for success metric in summary...")
    for key in run.summary.keys():
        if 'success' in key.lower():
            print(f"  Found: {key} = {run.summary[key]}")
            
    # Try to get just a few history rows
    print("\nTrying to get limited history...")
    history = run.history(keys=['eval/episode_success_any'], samples=10)
    print(f"Got {len(history)} history rows")
    if history:
        print("Sample data:")
        for i, row in enumerate(history[:3]):
            print(f"  Row {i}: {row}")
            
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()