#!/usr/bin/env python3 
"""
Batch processing script for training NeCA on multiple 3D models.
This script processes all models specified in the configuration file.
Supports multi-GPU training by distributing models across available GPUs.
"""
import os
import yaml
import argparse
import torch
import torch.multiprocessing as mp
import numpy as np
from src.config.configloading import load_config
from train import BasicTrainer, print_memory_stats

def process_models_on_gpu(rank, model_numbers, config_path):
    """Process assigned models on a specific GPU."""
    
    # Load base configuration
    cfg = load_config(config_path)
    
    input_data_dir = cfg["exp"].get("input_data_dir", "./data/GT_volumes/")
    output_recon_dir = cfg["exp"].get("output_recon_dir", "./logs/reconstructions/")
    
    # Set device for this process
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)
    
    print(f"[GPU {rank}] Processing {len(model_numbers)} models: {model_numbers}")
    print(f"[GPU {rank}] Input directory: {input_data_dir}")
    print(f"[GPU {rank}] Output directory: {output_recon_dir}")
    
    # Ensure output directory exists
    os.makedirs(output_recon_dir, exist_ok=True)
    
    successful_models = []
    failed_models = []
    
    # Get experiment parameters
    lrates = cfg["train"]["lrate"] if isinstance(cfg["train"]["lrate"], list) else [cfg["train"]["lrate"]]
    loss_weight_experiments = cfg["train"].get("loss_weight_experiments", [[1.0, 1.0, 0.1]])
    
    for i, model_id in enumerate(model_numbers):
        print(f"\n[GPU {rank}] {'='*60}")
        print(f"[GPU {rank}] Processing Model {model_id} ({i+1}/{len(model_numbers)})")
        print(f"[GPU {rank}] {'='*60}")
        
        try:
            # Check if input file exists
            input_file = os.path.join(input_data_dir, f"{model_id}.npy")
            if not os.path.exists(input_file):
                print(f"[GPU {rank}] WARNING: Input file not found: {input_file}")
                print(f"[GPU {rank}] Skipping model {model_id}")
                failed_models.append(model_id)
                continue
            
            # Run experiments for each learning rate and loss weight combination
            for exp_idx, (lr, loss_weights) in enumerate([(lr, weights) for lr in lrates for weights in loss_weight_experiments]):
                proj_w, sdf_w = loss_weights
                experiment_name = f"{model_id}_lr{lr}_proj{proj_w}_sdf{sdf_w}"
                print(f"\n[GPU {rank}] --- Experiment {exp_idx+1}: LR={lr}, Weights=[proj:{proj_w}, sdf:{sdf_w}] ---")
                
                # Update configuration for current experiment
                cfg["exp"]["current_model_id"] = experiment_name
                cfg["train"]["lrate"] = lr
                cfg["train"]["current_loss_weights"] = loss_weights
                
                # Set same seeds for all experiments (fair comparison)
                torch.manual_seed(42)
                np.random.seed(42)
                torch.cuda.manual_seed(42)
                torch.cuda.manual_seed_all(42)
                
                print(f"[GPU {rank}] Starting training for experiment {experiment_name}...")
                print(f"[GPU {rank}] Input file: {input_file}")
                print(f"[GPU {rank}] Expected output: {os.path.join(output_recon_dir, f'recon_{experiment_name}.npy')}")
                
                # Create trainer and start training
                trainer = BasicTrainer.__new__(BasicTrainer)
                trainer.__init__(cfg, device)
                
                print(f"[GPU {rank}] Initial memory stats:")
                print_memory_stats()
                
                # Start training
                trainer.start()
                
                print(f"[GPU {rank}] ✅ Successfully completed experiment {experiment_name}")
                
                # Clear memory between experiments
                del trainer
                torch.cuda.empty_cache()
            
            successful_models.append(model_id)
            
            # Memory is already cleared after each experiment
                
        except Exception as e:
            print(f"[GPU {rank}] ❌ Error processing model {model_id}: {str(e)}")
            failed_models.append(model_id)
            
            # Clear memory even on failure
            torch.cuda.empty_cache()
            continue
    
    # Print summary for this GPU
    print(f"\n[GPU {rank}] {'=' * 60}")
    print(f"[GPU {rank}] BATCH PROCESSING SUMMARY")
    print(f"[GPU {rank}] {'=' * 60}")
    print(f"[GPU {rank}] Total models processed: {len(model_numbers)}")
    print(f"[GPU {rank}] Successful: {len(successful_models)} - {successful_models}")
    print(f"[GPU {rank}] Failed: {len(failed_models)} - {failed_models}")
    
    if successful_models:
        print(f"\n[GPU {rank}] ✅ Successfully generated reconstructions:")
        for model_id in successful_models:
            recon_file = os.path.join(output_recon_dir, f"recon_{model_id}.npy")
            network_file = os.path.join(output_recon_dir, f"network_{model_id}.pth")
            print(f"[GPU {rank}]   Model {model_id}: {recon_file}")
            print(f"[GPU {rank}]   Network {model_id}: {network_file}")
    
    if failed_models:
        print(f"\n[GPU {rank}] ❌ Failed models: {failed_models}")
        print(f"[GPU {rank}] Please check the error messages above and ensure input files exist.")
    
    return successful_models, failed_models


def batch_process_models(config_path):
    """Coordinate multi-GPU training by distributing models across available GPUs."""
    
    # Load base configuration to get model numbers
    cfg = load_config(config_path)
    model_numbers = cfg["exp"].get("model_numbers", [1])
    
    # Check available GPUs
    n_gpus = torch.cuda.device_count()
    print(f"🔍 MULTI-GPU BATCH TRAINING")
    print(f"📊 Found {n_gpus} GPU(s)")
    print(f"📋 Total models to process: {len(model_numbers)} - {model_numbers}")
    
    if n_gpus == 0:
        raise RuntimeError("No GPUs available! This script requires CUDA.")
    elif n_gpus == 1:
        print("⚠️  Warning: Only one GPU detected. Using single GPU mode.")
        # Run on single GPU
        successful, failed = process_models_on_gpu(0, model_numbers, config_path)
    else:
        # Distribute models across GPUs
        models_per_gpu = []
        for i in range(n_gpus):
            # Distribute models round-robin across GPUs
            gpu_models = [model_numbers[j] for j in range(i, len(model_numbers), n_gpus)]
            models_per_gpu.append(gpu_models)
            print(f"🎯 GPU {i}: {gpu_models}")
        
        # Start multiprocessing
        print(f"\n🚀 Starting training on {n_gpus} GPUs...")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        # Use spawn method for better CUDA compatibility
        mp.set_start_method('spawn', force=True)
        
        processes = []
        for rank in range(n_gpus):
            if models_per_gpu[rank]:  # Only start process if there are models to process
                p = mp.Process(target=process_models_on_gpu, 
                             args=(rank, models_per_gpu[rank], config_path))
                p.start()
                processes.append(p)
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        print(f"\n🎉 All GPU processes completed!")
    
    print(f"\n✅ Multi-GPU batch processing finished!")

def main():
    parser = argparse.ArgumentParser(description="Batch process multiple 3D models with NeCA")
    parser.add_argument("--config", default="./config/CCTA.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--single", type=int, default=None,
                       help="Process only a single model with the specified ID")
    
    args = parser.parse_args()
    
    if args.single is not None:
        # Process single model
        cfg = load_config(args.config)
        cfg["exp"]["model_numbers"] = [args.single]
        cfg["exp"]["current_model_id"] = args.single
        
        # Save temporary config
        temp_config_path = "./config/temp_single_model.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)
        
        batch_process_models(temp_config_path)
        
        # Clean up temporary config
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
    else:
        # Process all models in config
        batch_process_models(args.config)

if __name__ == "__main__":
    main()
