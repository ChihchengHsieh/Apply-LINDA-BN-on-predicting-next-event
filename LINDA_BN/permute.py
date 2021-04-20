import numpy as np

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
