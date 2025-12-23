"""
Direct Attention Head Selection for Activation Steering

Loads a pre-specified list of (layer, head) tuples from JSON and evaluates
the steering configuration on a test dataset.

Assumes:
- evaluate_configuration_general exists
- heads are represented as (layer, head) tuples
- objective = accuracy (higher is better)

Author: you + Claude
"""

import argparse
import json
import os
import random
import numpy as np
import pandas as pd
import torch

from utils.ut_processing_utils import (
    load_model,
    layer_head_to_flattened_idx,
    get_attention_activations,
    get_fold_indices,
    extract_final_answer_dataset,
    select_device
)
from utils.ut_intervention_utils import (
    get_com_directions,
    lt_modulated_vector_add,
    get_interventions_dict_variable_alpha
)
from utils.ut_run_llms import run_llama_intervention_batch
from utils.ut_evaluation_utils import (
    evaluate_configuration_general,
    append_to_log_file,
    extract_answer_compare_gt,
    save_results
)

# ---------------------------
# Utilities
# ---------------------------

def load_heads_from_json(json_path):
    """
    Load list of (layer, head) tuples from JSON file.
    
    Expected format:
    [
        [12, 6],
        [12, 23],
        [12, 25],
        [13, 16],
        [13, 17]
    ]
    
    Returns list of tuples: [(12, 6), (12, 23), ...]
    """
    print(f"Loading heads from: {json_path}")
    with open(json_path, 'r') as f:
        heads_list = json.load(f)
    
    # Convert to tuples if they're lists
    heads = [tuple(h) if isinstance(h, list) else h for h in heads_list]
    
    print(f"Loaded {len(heads)} heads:")
    for h in heads:
        print(f"  {h}")
    
    return heads


def load_and_preprocess_initial_data(data_path, tokenizer, model, args):
    """Load and preprocess data for computing steering directions."""
    print(f"Loading initial data from: {data_path}")
    df = pd.read_json(data_path)
    df = get_attention_activations(df, tokenizer, model)
    print(f"Initial data loaded and preprocessed. Shape: {df.shape}")
    return df


def get_directions(model, df, args=None, id_column="data_id", column="activations"):
    """Compute center-of-mass directions for steering."""
    fold_index = 0
    index_dic = {}
    separated_activations = []
    separated_labels = []
    data_ids_order = []
    
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    if hasattr(model.config, "head_dim"):
        head_dim = model.config.head_dim
    else:
        head_dim = model.config.hidden_size // num_heads
    
    for data_id in df[id_column].unique():
        example_indexes = df[df[id_column] == data_id].index
        index_dic[data_id] = list(example_indexes)
        
        temp_activations = df[df[id_column] == data_id][column]
        activations = np.array([list(sample.values()) for sample in temp_activations.values])
        
        number_examples = len(temp_activations)
        example_activations = np.reshape(activations, (number_examples, num_layers, num_heads, head_dim))
        example_labels = [1 if label == True else 0 for label in df[df[id_column] == data_id]['correct'].values]
        
        separated_activations.append(example_activations)
        separated_labels.append(example_labels)
        data_ids_order.append(data_id)
    
    train_set_idxs, val_set_idxs, test_idxs = get_fold_indices(fold_index, args, data_ids_order)
    com_directions = get_com_directions(num_layers, num_heads, train_set_idxs, val_set_idxs, separated_activations, separated_labels)
    
    return com_directions, separated_activations


def prepare_evaluation_dataset(dataset_path, args):
    """Load and prepare evaluation dataset."""
    print(f"Loading evaluation dataset from: {dataset_path}")
    eval_df = pd.read_json(dataset_path)
    eval_df.reset_index(drop=True, inplace=True)
    
    # Apply consistency factor if needed
    if args.consistency_factor > 1 and 'data_id' in eval_df.columns:
        unique_data_ids = eval_df.data_id.unique()
        if len(unique_data_ids) > 0:
            indexes = [eval_df[eval_df['data_id'] == data_id].index[0] for data_id in unique_data_ids]
            repeated_indexes = indexes * args.consistency_factor
            eval_df = eval_df.loc[repeated_indexes].reset_index(drop=True)
    
    print(f"Evaluation dataset prepared. Shape: {eval_df.shape}")
    return eval_df


def evaluate_heads(
    heads,
    tokenizer,
    model,
    initial_df,
    eval_df,
    args,
):
    """
    Evaluate a specific head configuration.
    
    Args:
        heads: List of (layer, head) tuples
        tokenizer, model: Model and tokenizer
        initial_df: DataFrame for computing directions
        eval_df: DataFrame for evaluation
        args: Arguments
    
    Returns:
        accuracy (float)
    """
    if len(heads) == 0:
        print("Warning: No heads specified")
        return 0.0
    
    # Set alphas
    alphas = [args.alpha] * len(heads)
    if args.proportional_alpha:
        alphas = [100 / len(heads)] * len(heads)
    
    print(f"\nEvaluating configuration with {len(heads)} heads:")
    for i, h in enumerate(heads):
        print(f"  {h} (alpha={alphas[i]:.3f})")
    
    # Run evaluation
    results_raw = evaluate_configuration_general(
        args,
        tokenizer,
        model,
        heads,
        alphas,
        initial_df,
        external_test_set=eval_df,
    )
    
    # Process results
    processed = extract_answer_compare_gt(args, results_raw)
    df = pd.DataFrame(processed)
    
    accuracy = df.correct.mean()
    print(f"Accuracy: {accuracy:.4f}")
    
    return accuracy, processed


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pre-specified attention heads for activation steering"
    )
    
    # Required arguments
    parser.add_argument(
        "--heads_json",
        type=str,
        required=True,
        help="Path to JSON file containing list of [layer, head] pairs"
    )
    parser.add_argument(
        "--initial_dataset_path",
        type=str,
        required=True,
        help="Path to JSON dataset for calculating steering directions"
    )
    parser.add_argument(
        "--evaluation_dataset_path",
        type=str,
        required=True,
        help="Path to JSON dataset for evaluation"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Model name or path"
    )
    
    # Steering arguments
    parser.add_argument(
        "--alpha",
        type=float,
        default=3.0,
        help="Steering strength (alpha)"
    )
    parser.add_argument(
        "--proportional_alpha",
        action="store_true",
        default=False,
        help="If true, alpha = 1/n_heads"
    )
    
    # Data processing arguments
    parser.add_argument(
        "--num_fold",
        type=int,
        default=1,
        help="Number of folds for cross-validation"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0,
        help="Ratio of validation set size to development set size"
    )
    parser.add_argument(
        "--use_center_of_mass",
        action="store_true",
        default=True,
        help="Use center of mass for directions"
    )
    parser.add_argument(
        "--use_random_dir",
        action="store_true",
        default=False,
        help="Use random directions"
    )
    parser.add_argument(
        "--id_column",
        type=str,
        default="data_id",
        help="Column name for data IDs"
    )
    parser.add_argument(
        "--activations_column",
        type=str,
        default="activations",
        help="Column name for activations"
    )
    
    # Generation arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for running interventions"
    )
    parser.add_argument(
        "--consistency_factor",
        type=int,
        default=1,
        help="Consistency factor for preparing test set"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=600,
        help="Max new tokens for generation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for generation"
    )
    
    # Dataset-specific arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="survival_instinct",
        help="Dataset name hint"
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="ab_cot",
        help="Prompt type hint"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_json",
        default="direct_heads_results.json",
        help="Path to save results JSON"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save results"
    )
    
    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # -----------------------
    # Reproducibility
    # -----------------------
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # -----------------------
    # Load heads configuration
    # -----------------------
    heads = load_heads_from_json(args.heads_json)
    
    if len(heads) == 0:
        raise ValueError("No heads found in JSON file")
    
    # -----------------------
    # Load model + data
    # -----------------------
    print("\nLoading model...")
    device = select_device(min_vram_gb=20)
    tokenizer, model = load_model(args.model_name, device="auto")
    
    try:
        args.num_heads = model.config.num_attention_heads
    except AttributeError:
        args.num_heads = model.config.text_config.num_attention_heads
    
    args.suffix = ""
    
    print("\nLoading datasets...")
    initial_df = load_and_preprocess_initial_data(
        args.initial_dataset_path,
        tokenizer,
        model,
        args
    )
    
    final_eval_df = prepare_evaluation_dataset(
        args.evaluation_dataset_path,
        args
    )
    
    # -----------------------
    # Evaluate configuration
    # -----------------------
    print("\n" + "="*50)
    print("EVALUATING HEAD CONFIGURATION")
    print("="*50)
    
    accuracy, detailed_results = evaluate_heads(
        heads,
        tokenizer,
        model,
        initial_df,
        final_eval_df,
        args,
    )
    
    # -----------------------
    # Print results
    # -----------------------
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Number of heads: {len(heads)}")
    print(f"Alpha: {args.alpha}")
    if args.proportional_alpha:
        print(f"Proportional alpha: {1/len(heads):.4f} per head")
    print("\nHeads:")
    for h in heads:
        print(f"  Layer {h[0]}, Head {h[1]}")
    
    # -----------------------
    # Save results
    # -----------------------
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {
        "accuracy": float(accuracy),
        "heads": heads,
        "num_heads": len(heads),
        "alpha": args.alpha,
        "proportional_alpha": args.proportional_alpha,
        "model_name": args.model_name,
        "initial_dataset": args.initial_dataset_path,
        "evaluation_dataset": args.evaluation_dataset_path,
        "detailed_results": detailed_results,
    }

    print(results)
    
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {args.output_json}")


if __name__ == "__main__":
    main()