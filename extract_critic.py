import argparse
import os
import torch
from accelerate import Accelerator
from critic_code.critic import VLMDoubleCritic
import shlex

def extract_weights(args):
    """
    Loads a trainer checkpoint saved by Accelerator and extracts the raw model weights.
    """
    accelerator = Accelerator()
    device = accelerator.device

    # Initialize the model with the same architecture as during training.
    # The weights will be random initially.
    model = VLMDoubleCritic(
        device=device,
        critic_lm=args.critic_lm,
        cache_dir=args.cache_dir,
        in_dim=1536,  # This should match your model configuration
        out_dim=1     # This should match your model configuration
    )

    # The accelerator needs to prepare the model to know how to load the state.
    model = accelerator.prepare(model)

    # Load the entire training state from the checkpoint directory.
    # This will populate the model with the trained weights.
    try:
        accelerator.load_state(args.checkpoint_path)
        print(f"Successfully loaded checkpoint from {args.checkpoint_path}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Please ensure the checkpoint path is a directory containing the accelerator state.")
        return

    # After loading, the model might be wrapped (e.g., in DDP).
    # Unwrap it to get the original nn.Module.
    unwrapped_model = accelerator.unwrap_model(model)

    # Get the state dictionary from the unwrapped model.
    state_dict = unwrapped_model.state_dict()

    # Ensure the output directory exists.
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save the extracted state dictionary to the specified file.
    # We save on the main process to avoid race conditions.
    if accelerator.is_main_process:
        torch.save(state_dict, args.output_path)
        print(f"Successfully extracted and saved critic weights to {args.output_path}")