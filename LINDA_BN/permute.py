import numpy as np
import pyAgrum as gum
import torch
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal



def generate_permutation_for_numerical(input_data: torch.tensor, num_samples_per_feature, variance = 0.5, ):
    '''
    [input_data]: Normalised data. should be a 1-D tensor.
    --------------------------
    Return: all permutations.
    '''

    all_permutations = []

    input_backup = input_data.clone()

    max_range = torch.clip(input_data + variance , - 1, 1)
    min_range = torch.clip(input_data - variance, -1, 1)

    for i in range(input_data.size(-1)):
        input_to_permute = input_backup.clone()
        input_to_permute = input_to_permute.unsqueeze(0).repeat(num_samples_per_feature, 1)
        input_to_permute[:,i] = torch.zeros(num_samples_per_feature).uniform_(min_range[i], max_range[i])
        all_permutations.extend(list(torch.chunk(input_to_permute, num_samples_per_feature, dim=0)))
    
    ########## append the original data ##########
    all_permutations.append(input_backup.unsqueeze(0)) 

    return all_permutations

def generate_permutation_for_numerical_all_dim(input_data: torch.tensor, num_samples, variance = 0.5, ):
    '''
    [input_data]: Normalised data. should be a 1-D tensor.
    --------------------------
    Return: all permutations.
    '''
    max_range = torch.clip(input_data + variance , -0.999999, 1).float()
    min_range = torch.clip(input_data - variance, -1, 0.999999).float()
    dist = Uniform(min_range, max_range)
    return dist.sample((num_samples,))

def generate_permutations_for_normerical_all_dim_normal_dist(input_data: torch.tensor, num_samples, variance = 0.5):
    dist = Normal(input_data, torch.full_like(input_data, variance))
    return dist.sample((num_samples,))

def generate_permutation_for_trace(trace: np.array, vocab_size: int, last_n_stages_to_permute: int = None):
    # For each stage (activity), we replace it by another.
    # But we still maintain the same length of the trace.
    '''
    Basically, we see each activity in the trace as a feature.
    'Stage1, Stage2, Stage 3, Stage4 ....  Destination'

    Permute on last few stages
    '''

    all_permutations = []

    if (last_n_stages_to_permute is None):
        last_n_stages_to_permute = len(trace)

    trace_backup = trace.copy()

    max_index = len(trace) - 1

    all_permutations.append(trace.tolist())

    for idx_last in range(last_n_stages_to_permute):
        idx = max_index - idx_last
        trace_to_permute = trace_backup.copy()
        all_permutations.extend([replaceIndex(trace=trace_to_permute, index=idx, value=v_i).tolist(
        ) for v_i in range(vocab_size) if v_i != trace[idx]])

    return all_permutations


def replaceIndex(trace: np.array, index: int, value: int) -> np.array:
    trace[index] = value
    return trace


def exploring():
    gum.MarkovBlanket
    pass