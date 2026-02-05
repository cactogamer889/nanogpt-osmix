"""
Osmix Training Script for Text Generation
Custom training configuration for Natural Language Processing - Text Generation tasks
"""

import os
import sys
import torch
from model import GPTConfig, GPT

# Default configuration for Text Generation
DEFAULT_CONFIG = {
    'model_name': 'osmix_text_generation',
    'output_format': 'ckpt',
    'batch_size': 12,
    'block_size': 1024,
    'gradient_accumulation_steps': 5,
    'learning_rate': 6e-4,
    'max_iters': 600000,
    'n_layer': 12,
    'n_head': 12,
    'n_embd': 768,
    'dropout': 0.0,
    'device': 'cuda',
    'dtype': 'bfloat16',
    'save_interval': 1000  # Save checkpoint every N iterations
}

def get_user_config():
    """Get configuration from user input or command line arguments"""
    import argparse
    
    # Parse command line arguments first
    parser = argparse.ArgumentParser(description='Osmix Text Generation Training')
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--output_format', type=str, default=None, choices=['ckpt', 'pth', 'safetensors', 'onnx', 'ggml', 'gguf'])
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--block_size', type=int, default=None)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--max_iters', type=int, default=None)
    parser.add_argument('--n_layer', type=int, default=None)
    parser.add_argument('--n_head', type=int, default=None)
    parser.add_argument('--n_embd', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--dtype', type=str, default=None)
    parser.add_argument('--save_interval', type=int, default=None)
    
    args = parser.parse_args()
    
    config = DEFAULT_CONFIG.copy()
    
    # Override with command line arguments if provided
    if args.model_name:
        config['model_name'] = args.model_name
    if args.output_format:
        config['output_format'] = args.output_format
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.block_size:
        config['block_size'] = args.block_size
    if args.gradient_accumulation_steps:
        config['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    if args.max_iters:
        config['max_iters'] = args.max_iters
    if args.n_layer:
        config['n_layer'] = args.n_layer
    if args.n_head:
        config['n_head'] = args.n_head
    if args.n_embd:
        config['n_embd'] = args.n_embd
    if args.dropout is not None:
        config['dropout'] = args.dropout
    if args.device:
        config['device'] = args.device
    if args.dtype:
        config['dtype'] = args.dtype
    if args.save_interval:
        config['save_interval'] = args.save_interval
    
    # If all args provided, skip interactive mode
    all_args_provided = all([
        args.model_name, args.output_format, args.batch_size, args.block_size,
        args.learning_rate, args.max_iters, args.n_layer, args.n_head, args.n_embd
    ])
    
    if all_args_provided:
        return config
    
    print("\n=== Osmix Text Generation Training ===")
    print("Configure your training parameters (press Enter for defaults):\n")
    
    # Model name
    model_name = input(f"Model name (default: {config['model_name']}): ").strip()
    if model_name:
        config['model_name'] = model_name
    
    # Output format
    print("\nAvailable output formats:")
    formats = ['ckpt', 'pth', 'ggml', 'gguf', 'safetensors', 'onnx']
    for i, fmt in enumerate(formats, 1):
        print(f"{i}. {fmt}")
    format_choice = input(f"\nSelect format (default: {config['output_format']}): ").strip()
    if format_choice:
        try:
            config['output_format'] = formats[int(format_choice) - 1]
        except:
            pass
    
    # Training parameters
    batch_size = input(f"Batch size (default: {config['batch_size']}): ").strip()
    if batch_size:
        config['batch_size'] = int(batch_size)
    
    block_size = input(f"Block size (default: {config['block_size']}): ").strip()
    if block_size:
        config['block_size'] = int(block_size)
    
    learning_rate = input(f"Learning rate (default: {config['learning_rate']}): ").strip()
    if learning_rate:
        config['learning_rate'] = float(learning_rate)
    
    max_iters = input(f"Max iterations (default: {config['max_iters']}): ").strip()
    if max_iters:
        config['max_iters'] = int(max_iters)
    
    # Model architecture
    n_layer = input(f"Number of layers (default: {config['n_layer']}): ").strip()
    if n_layer:
        config['n_layer'] = int(n_layer)
    
    n_head = input(f"Number of attention heads (default: {config['n_head']}): ").strip()
    if n_head:
        config['n_head'] = int(n_head)
    
    n_embd = input(f"Embedding dimension (default: {config['n_embd']}): ").strip()
    if n_embd:
        config['n_embd'] = int(n_embd)
    
    # Save interval
    save_interval = input(f"Save checkpoint every N iterations (default: {config['save_interval']}): ").strip()
    if save_interval:
        config['save_interval'] = int(save_interval)
    
    return config

def save_checkpoint(model, output_path, format_type, model_name, optimizer=None, iter_num=0, best_val_loss=1e9, checkpoint_dir=None, iter_num_suffix=None):
    """Save model checkpoint in specified format"""
    # Determine file extension based on format
    if format_type == 'ckpt':
        ext = 'ckpt'
    elif format_type == 'pth':
        ext = 'pth'
    elif format_type == 'safetensors':
        ext = 'safetensors'
    elif format_type == 'onnx':
        ext = 'onnx'
    else:
        ext = 'pth'  # Default fallback
    
    # Add iteration suffix if provided (for periodic checkpoints)
    if iter_num_suffix is not None:
        base_name = os.path.splitext(output_path)[0]
        if checkpoint_dir:
            checkpoint_path = os.path.join(checkpoint_dir, f"{os.path.basename(base_name)}_iter_{iter_num_suffix}.{ext}")
        else:
            checkpoint_path = f"{base_name}_iter_{iter_num_suffix}.{ext}"
    else:
        checkpoint_path = output_path
    
    if format_type == 'ckpt' or format_type == 'pth':
        checkpoint = {
            'model': model.state_dict(),
            'model_args': model.config.__dict__,
            'model_name': model_name,
            'iter_num': iter_num,
            'best_val_loss': best_val_loss
        }
        if optimizer is not None:
            checkpoint['optimizer'] = optimizer.state_dict()
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to: {checkpoint_path}")
    
    elif format_type == 'safetensors':
        try:
            from safetensors.torch import save_file
            state_dict = model.state_dict()
            save_file(state_dict, checkpoint_path)
            print(f"Checkpoint saved in safetensors format: {checkpoint_path}")
        except ImportError:
            print("safetensors not installed. Installing...")
            os.system("pip install safetensors")
            from safetensors.torch import save_file
            state_dict = model.state_dict()
            save_file(state_dict, output_path)
            print(f"Model saved in safetensors format: {output_path}")
    
    elif format_type == 'onnx':
        try:
            import torch.onnx
            dummy_input = torch.randint(0, model.config.vocab_size, (1, model.config.block_size))
            torch.onnx.export(
                model,
                dummy_input,
                checkpoint_path,
                input_names=['input_ids'],
                output_names=['logits'],
                dynamic_axes={'input_ids': {0: 'batch_size'}, 'logits': {0: 'batch_size'}}
            )
            print(f"Checkpoint saved in ONNX format: {checkpoint_path}")
        except Exception as e:
            print(f"Error saving to ONNX: {e}")
    
    elif format_type in ['ggml', 'gguf']:
        print(f"Format {format_type} requires additional conversion. Use tools like llama.cpp")
        intermediate_path = checkpoint_path.replace(f'.{format_type}', '.pth')
        checkpoint = {
            'model': model.state_dict(),
            'model_args': model.config.__dict__,
            'iter_num': iter_num,
            'best_val_loss': best_val_loss
        }
        if optimizer is not None:
            checkpoint['optimizer'] = optimizer.state_dict()
        torch.save(checkpoint, intermediate_path)
        print(f"Intermediate checkpoint saved to: {intermediate_path}")
    
    return checkpoint_path

def main():
    """Main training function"""
    # Get configuration
    config = get_user_config()
    
    # Create output directory
    output_dir = os.path.join('product', 'models')
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure device and data type
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }
    dtype = dtype_map.get(config['dtype'], torch.bfloat16)
    
    # Create model configuration
    model_config = GPTConfig(
        block_size=config['block_size'],
        vocab_size=50304,  # GPT-2 vocab size
        n_layer=config['n_layer'],
        n_head=config['n_head'],
        n_embd=config['n_embd'],
        dropout=config['dropout'],
        bias=False
    )
    
    # Create model
    print(f"\nCreating model: {config['model_name']}")
    model = GPT(model_config)
    model.to(device)
    
    # Prepare output filenames
    # Main model file
    output_filename = f"{config['model_name']}.{config['output_format']}"
    output_path = os.path.join(output_dir, output_filename)
    
    # Checkpoint directory for periodic saves
    checkpoint_dir = os.path.join(output_dir, config['model_name'] + '_checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"\nTraining Configuration:")
    print(f"  Model: {config['model_name']}")
    print(f"  Format: {config['output_format']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Block Size: {config['block_size']}")
    print(f"  Gradient Accumulation: {config['gradient_accumulation_steps']}")
    print(f"  Learning Rate: {config['learning_rate']}")
    print(f"  Max Iters: {config['max_iters']}")
    print(f"  Layers: {config['n_layer']}")
    print(f"  Heads: {config['n_head']}")
    print(f"  Embedding Dim: {config['n_embd']}")
    print(f"  Device: {device}")
    print(f"  Dtype: {config['dtype']}")
    print(f"\nModel will be saved to: {output_path}")
    print(f"Checkpoints will be saved every {config['save_interval']} iterations to: {checkpoint_dir}")
    
    # Note: Full training loop integration needed
    # During training, call save_checkpoint() every config['save_interval'] iterations:
    # save_checkpoint(model, optimizer, iter_num, best_val_loss, output_path, 
    #                 config['output_format'], config['model_name'], 
    #                 checkpoint_dir, iter_num)
    
    print("\nText Generation training configured successfully!")
    print("Note: Integrate full training loop from train.py for complete training")
    print(f"      Save checkpoints every {config['save_interval']} iterations")

if __name__ == '__main__':
    main()
