"""
Custom Training System - Osmix
Supports training models with different output formats (.ckpt, .pth, .ggml, .gguf, .safetensors, .onnx)
"""

import os
import sys
import argparse
import torch
from model import GPTConfig, GPT

# Supported output formats
SUPPORTED_FORMATS = ['ckpt', 'pth', 'ggml', 'gguf', 'safetensors', 'onnx']

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Osmix Custom Training System')
    
    # Training script path
    parser.add_argument('train_script', type=str, nargs='?', 
                       help='Path to training script (e.g., system/Natural Language Processing/Text Generation/train_osmix.py)')
    
    # Basic configurations
    parser.add_argument('--model_name', type=str, default=None, help='Model name')
    parser.add_argument('--output_format', type=str, choices=SUPPORTED_FORMATS, default='ckpt',
                       help='Output format')
    parser.add_argument('--output_dir', type=str, default='product/models', help='Output directory')
    
    # Training configurations
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--block_size', type=int, default=None, help='Block size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=None,
                       help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--max_iters', type=int, default=None, help='Maximum iterations')
    parser.add_argument('--n_layer', type=int, default=None, help='Number of layers')
    parser.add_argument('--n_head', type=int, default=None, help='Number of attention heads')
    parser.add_argument('--n_embd', type=int, default=None, help='Embedding dimension')
    parser.add_argument('--dropout', type=float, default=None, help='Dropout rate')
    
    # Data configurations
    parser.add_argument('--dataset', type=str, default=None, help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    
    # System configurations
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--dtype', type=str, default='bfloat16', help='Data type')
    parser.add_argument('--compile', action='store_true', help='Compile model with torch.compile')
    
    return parser.parse_args()

def interactive_config():
    """Interactive configuration if no script is provided"""
    print("\n=== Osmix Training System ===")
    print("No training script provided. Starting interactive configuration...\n")
    
    config = {}
    
    # Model selection
    print("Available models:")
    print("1. GPT-2 (124M)")
    print("2. GPT-2 Medium (350M)")
    print("3. GPT-2 Large (774M)")
    print("4. GPT-2 XL (1558M)")
    print("5. Custom")
    
    model_choice = input("\nSelect model (1-5): ").strip()
    model_map = {
        '1': 'gpt2',
        '2': 'gpt2-medium',
        '3': 'gpt2-large',
        '4': 'gpt2-xl',
        '5': 'custom'
    }
    config['model_type'] = model_map.get(model_choice, 'gpt2')
    
    if config['model_type'] == 'custom':
        config['n_layer'] = int(input("Number of layers: ") or "12")
        config['n_head'] = int(input("Number of heads: ") or "12")
        config['n_embd'] = int(input("Embedding dimension: ") or "768")
    
    # Output format
    print("\nAvailable output formats:")
    for i, fmt in enumerate(SUPPORTED_FORMATS, 1):
        print(f"{i}. {fmt}")
    format_choice = input(f"\nSelect format (1-{len(SUPPORTED_FORMATS)}): ").strip()
    try:
        config['output_format'] = SUPPORTED_FORMATS[int(format_choice) - 1]
    except:
        config['output_format'] = 'ckpt'
    
    # Training parameters
    config['model_name'] = input("\nModel name (default: custom_model): ").strip() or "custom_model"
    config['batch_size'] = int(input("Batch size (default: 12): ").strip() or "12")
    config['block_size'] = int(input("Block size (default: 1024): ").strip() or "1024")
    config['learning_rate'] = float(input("Learning rate (default: 6e-4): ").strip() or "6e-4")
    config['max_iters'] = int(input("Max iterations (default: 600000): ").strip() or "600000")
    
    return config

def main():
    args = parse_args()

    # If training script is provided, execute it
    if args.train_script:
        if os.path.exists(args.train_script):
            print(f"Executing training script: {args.train_script}")
            # Execute the script with remaining arguments
            exec(open(args.train_script).read())
        else:
            print(f"Error: Training script not found: {args.train_script}")
            sys.exit(1)
    else:
        # Interactive mode
        config = interactive_config()

        # Create output directory
        output_dir = 'product/models'
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure device and data type
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        dtype_map = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16
        }
        dtype = dtype_map.get(args.dtype, torch.bfloat16)
        
        # Create model configuration
        if config['model_type'] == 'custom':
            model_config = GPTConfig(
                block_size=config.get('block_size', 1024),
                vocab_size=50304,
                n_layer=config.get('n_layer', 12),
                n_head=config.get('n_head', 12),
                n_embd=config.get('n_embd', 768),
                dropout=0.0,
                bias=False
            )
        else:
            # Use GPT-2 preset
            from model import GPT
            model = GPT.from_pretrained(config['model_type'])
            model.to(device)
            print(f"\nModel loaded: {config['model_type']}")
            print(f"Output format: {config['output_format']}")
            print(f"Model name: {config['model_name']}")
            return
        
        # Create model
        print(f"\nCreating model: {config['model_name']}")
        model = GPT(model_config)
        model.to(device)
        
        # Compile if requested
        if args.compile:
            print("Compiling model...")
            model = torch.compile(model)
        
        # Prepare output filename
        output_filename = f"{config['model_name']}.{config['output_format']}"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"\nTraining Configuration:")
        print(f"  Model: {config['model_name']}")
        print(f"  Format: {config['output_format']}")
        print(f"  Batch Size: {config['batch_size']}")
        print(f"  Block Size: {config['block_size']}")
        print(f"  Learning Rate: {config['learning_rate']}")
        print(f"  Max Iters: {config['max_iters']}")
        print(f"  Device: {device}")
        print(f"\nModel will be saved to: {output_path}")
        print("\nNote: Full training loop integration coming soon...")

if __name__ == '__main__':
    main()
