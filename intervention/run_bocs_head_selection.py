"""
BOCS-based Attention Head Selection for Activation Steering

Replaces exhaustive / heuristic head sweeps with
Bayesian Optimization of Combinatorial Structures (BOCS-SA).

Assumes:
- evaluate_configuration_general exists
- heads are represented as (layer, head) tuples
- objective = accuracy (higher is better)

Author: you + ChatGPT
"""

import argparse
import json
import os
import random
import numpy as np
import pandas as pd
import torch

from BOCS import BOCS
from sample_models import sample_models

from utils.ut_processing_utils import (
    load_model,
    layer_head_to_flattened_idx,
    get_attention_activations,
    get_fold_indices,
    extract_final_answer_dataset,
    select_device
)
from utils.ut_intervention_utils import get_com_directions, lt_modulated_vector_add, get_interventions_dict_variable_alpha
from utils.ut_run_llms import run_llama_intervention_batch

from utils.ut_evaluation_utils import evaluate_configuration_general, append_to_log_file, extract_answer_compare_gt, save_results

# ---------------------------
# Utilities
# ---------------------------

def binary_to_heads(x, head_list):
    """Convert binary vector to list of (layer, head) tuples."""
    return [head_list[i] for i in range(len(x)) if x[i] == 1]

def load_and_preprocess_initial_data(data_path, tokenizer, model, args):
    print(f"Loading initial data from: {data_path}")
    df = pd.read_json(data_path)
    # The notebook uses get_attention_activations which also computes 'complete_inputs'
    # We'll assume get_activations_bau is a function known to get_attention_activations
    # or handle it inside get_attention_activations if it's a fixed part of its logic.
    # For now, passing None and assuming get_attention_activations can handle it or has a default.
    df = get_attention_activations(df, tokenizer, model) # Adapt get_activations_bau_func as needed
    print(f"Initial data loaded and preprocessed. Shape: {df.shape}")
    print(df['complete_inputs'].values[0])
    #print(df['activations'].values[0])
    return df

def get_directions(model, df, args=None, id_column = "data_id", column = "activations"):
    fold_index = 0
    #### Purpose of this script is to process attention head activations from a dataframe
    #### So for every unique data_id, it separates the attention head activations into different batches and labels
    index_dic = {}

    ## List of lists with attention head activations for each data_id examples
    separated_activations = []

    ## List of lists with labels for each data_id examples
    separated_labels = []
    data_ids_order = []

    num_layers = model.config.num_hidden_layers 
    num_heads = model.config.num_attention_heads

    if hasattr(model.config, "head_dim"):
        head_dim = model.config.head_dim
    else:
        head_dim = model.config.hidden_size // num_heads

    for data_id in df[id_column].unique():

        ## Necessary? --> used later when expanding train idxs and used for verbose logging
        example_indexes = df[df[id_column] == data_id].index
        ## Gives indexes for samples in the whole dataset
        index_dic[data_id] = list(example_indexes)
        ## Example: {'304_a': [0, 2], '304_b': [1, 3], '294_a': [4, 6], '294_b': [5, 7]} --> Dataset with 4 unique ids with 2 examples each
        
        temp_activations = df[df[id_column] == data_id][column]
        activations = np.array([list(sample.values()) for sample in temp_activations.values]) # [num_examples, num_layers x num_heads, head_dim]
        
        ## Number of example for the current data_id
        number_examples = len(temp_activations)
        
        ## split into attention heads
        example_activations = np.reshape(activations, (number_examples, num_layers, num_heads, head_dim))
        example_labels =[1 if label==True else 0 for label in df[df[id_column] == data_id]['correct'].values]
        
        separated_activations.append(example_activations)
        separated_labels.append(example_labels)
        
        data_ids_order.append(data_id)

    train_set_idxs, val_set_idxs, test_idxs = get_fold_indices(fold_index, args, data_ids_order)
    com_directions = get_com_directions(num_layers, num_heads, train_set_idxs, val_set_idxs, separated_activations, separated_labels)
    return com_directions, separated_activations


def prepare_evaluation_dataset(dataset_path, args):
    print(f"Loading evaluation dataset from: {dataset_path}")
    eval_df = pd.read_json(dataset_path)
    eval_df.reset_index(drop=True, inplace=True)
    
    # Apply consistency factor if needed, similar to prepare_test_set in notebook
    if args.consistency_factor > 1 and 'data_id' in eval_df.columns:
        unique_data_ids = eval_df.data_id.unique()
        if len(unique_data_ids) > 0:
            indexes = [eval_df[eval_df['data_id'] == data_id].index[0] for data_id in unique_data_ids]
            repeated_indexes = indexes * args.consistency_factor
            eval_df = eval_df.loc[repeated_indexes].reset_index(drop=True)

    print(f"Evaluation dataset prepared. Shape: {eval_df.shape}")
    return eval_df

def evaluate_x(
    x,
    head_list,
    tokenizer,
    model,
    initial_df,
    eval_df,
    args,
):
    """
    Black-box objective f(x):
    returns scalar accuracy
    """
    heads = binary_to_heads(x, head_list)

    if len(heads) == 0:
        return 0.0

    alphas = [args.alpha] * len(heads)

    if args.proportional_alpha:
        alphas = [1 / len(heads)] * len(heads)

    results_raw = evaluate_configuration_general(
        args,
        tokenizer,
        model,
        heads,
        alphas,
        initial_df,
        external_test_set=eval_df,
    )

    processed = extract_answer_compare_gt(args, results_raw)
    df = pd.DataFrame(processed)

    print(f"accuracy: {df.correct.mean()}")

    return df.correct.mean()


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--initial_dataset_path", type=str, required=True, help="Path to the JSON dataset for calculating initial directions and sweep evaluation.")
    parser.add_argument("--evaluation_dataset_path", type=str, required=True, help="Path to the JSON dataset for final evaluation with the best head.")
    parser.add_argument("--layers", required=True,
                        help="Comma-separated layer indices, e.g. 8,9,10")
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--eval_budget", type=int, default=28)
    parser.add_argument("--n_init", type=int, default=10)
    parser.add_argument("--lambda_l1", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", default="bocs_results.json")
    parser.add_argument("--num_fold", type=int, default=1, help="Number of folds for cross-validation.")
    parser.add_argument("--val_ratio", type=float, default=0, help="Ratio of validation set size to development set size.")
    parser.add_argument("--use_center_of_mass", action="store_true", default=True, help="Use center of mass for directions.")
    parser.add_argument("--use_random_dir", action="store_true", default=False, help="Use random directions.")
    parser.add_argument("--id_column", type=str, default="data_id", help="Column name for data IDs in dataframes.")
    parser.add_argument("--activations_column", type=str, default="activations", help="Column name for activations in dataframes.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for running interventions.")
    parser.add_argument("--consistency_factor", type=int, default=1, help="Consistency factor for preparing test set.")
    parser.add_argument("--dataset_name", type=str, default="survival_instinct", help="Dataset name hint for extract_final_answer_dataset.")
    parser.add_argument("--prompt_type", type=str, default="ab_cot", help="Prompt type hint for extract_final_answer_dataset.")
    parser.add_argument("--max_new_tokens", type=int, default=600, help="Max new tokens for generation.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for generation during final evaluation.")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results.")
    parser.add_argument("--proportional_alpha", type=bool, default=False, help="If true, alpha = 1/n heads")
    
    args = parser.parse_args()

    # -----------------------
    # Reproducibility
    # -----------------------
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # -----------------------
    # Load model + data
    # -----------------------
    device = select_device(min_vram_gb=20)
    tokenizer, model = load_model(args.model_name, device="auto")

    try: 
        args.num_heads = model.config.num_attention_heads # Set num_heads in args
    except AttributeError:
        args.num_heads = model.config.text_config.num_attention_heads  # Set num_heads in args

    #eval_df = pd.read_json(args.dataset_path)
    args.suffix = ""
    initial_df = load_and_preprocess_initial_data(args.initial_dataset_path, tokenizer, model, args)

    final_eval_df = prepare_evaluation_dataset(args.evaluation_dataset_path, args)
    final_eval_df = final_eval_df.iloc[0:100]

    layers = [int(l) for l in args.layers.split(",")]
    num_heads = model.config.num_attention_heads

    # -----------------------
    # Define combinatorial space
    # -----------------------
    head_list = [(l, h) for l in layers for h in range(num_heads)]
    d = len(head_list)

    print(f"[BOCS] Optimizing over {d} heads")

    # -----------------------
    # Initial random designs
    # -----------------------
    #X_init = sample_models(args.n_init, d)
    def sample_binary_models(n_models, n_vars, p=0.05):
      """
      Sample sparse binary vectors without integer overflow.
      p = expected fraction of active heads
      """
      return (np.random.rand(n_models, n_vars) < p).astype(int)

    X_init = sample_binary_models(args.n_init, d, p=0.05)

    y_init = np.zeros(args.n_init)
    for i in range(args.n_init):
        y_init[i] = evaluate_x(
            X_init[i],
            head_list,
            tokenizer,
            model,
            initial_df,
            final_eval_df,
            args,
        )
        print(f"[init {i}] acc={y_init[i]:.4f}")

    # -----------------------
    # BOCS inputs
    # -----------------------
    inputs = {
        "n_vars": d,
        "evalBudget": args.eval_budget,
        "n_init": args.n_init,
        "lambda": args.lambda_l1,
        "x_vals": X_init,
        "y_vals": y_init,
        "model": lambda X: np.array([
            evaluate_x(x, head_list, tokenizer, model, initial_df, final_eval_df, args)
            for x in X
        ]),
        "penalty": lambda X: args.lambda_l1 * X.sum(axis=1),
    }

    # -----------------------
    # Run BOCS-SA
    # -----------------------
    print("[BOCS] Starting optimization")
    BOCS_model, BOCS_obj = BOCS(
        inputs.copy(),
        order=2,
        acquisitionFn="SA",
    )

    best_idx = np.argmax(BOCS_obj)
    best_x = inputs["x_vals"][best_idx]
    best_heads = binary_to_heads(best_x, head_list)
    best_score = BOCS_obj[best_idx]

    print("\n=== BEST CONFIGURATION ===")
    print("Accuracy:", best_score)
    print("Heads:")
    for h in best_heads:
        print(" ", h)

    # -----------------------
    # Save results
    # -----------------------
    out = {
        "accuracy": float(best_score),
        "heads": best_heads,
        "binary_vector": best_x.tolist(),
        "layers": layers,
        "alpha": args.alpha,
        "lambda_l1": args.lambda_l1,
        "eval_budget": args.eval_budget,
    }

    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[BOCS] Saved results to {args.output_json}")


if __name__ == "__main__":
    main()
