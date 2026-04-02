import math
from numbers import Number
from typing import Any

import numpy as np
from scipy.stats import spearmanr


def _dcg(scores):
    return sum([
        score / math.log2(i + 1)
        for i, score in enumerate(scores, start=1)
    ])

def standard_ndcg_scoring(answer: list[Any], ground_truth: list[Any], k: int) -> list[int]:
    """
    Returns the relevance scores for the elements in *answer* based on their position in *ground_truth*.
     If an element from *answer* is missing in *ground_truth*, it is assigned a relevance score of 0; otherwise, the relevance score is computed as k - position_in_ground_truth.
    :param answer: the elements for which to compute the relevance scores
    :param ground_truth: sorted ground truth elements
    :param k: consider only the top-k elements for both *answer* and *ground_truth*.
    :return:
    """
    return [k - ground_truth.index(elem) if elem in ground_truth[:k] else 0 for elem in answer[:k]]


def ndcg_k(relevance_scores: list[Number], ground_truth_scores: list[Number]=None, k: int=None):
    """
    NDCG@k. It computes the Normalized Discounted Cumulative Gain for *relevance_scores* and *ground_truth_scores*,
    then normalizes the result.
    All scores must be non-negative.
    :param relevance_scores: the relevance scores, used as they are.
    :param ground_truth_scores: scores for the ideal ranking. Defaults to [k, k-1, ..., 1] if not specified.
    :param k: consider only the top-k elements. If not specified, uses the full length of *relevance_scores*.
    :return: NDCG value.
    """
    if k is None:
        k = len(relevance_scores)

    llm = relevance_scores[:k]
    gt = ground_truth_scores[:k] if ground_truth_scores is not None else [k - i for i in range(k)]
    dcg_llm = _dcg(llm)
    dcg_ideal = _dcg(gt)

    return dcg_llm / dcg_ideal


def precision_at_k(llm_ranking, ground_truth, k):
    """
    Precision@k = (# correctly predicted in top-k) / (# predicted in top-k)
    ground_truth: list of true top customers (sorted by similarity)
    llm_ranking: list of predicted top customers (sorted by similarity)
    """
    gt_k = set(ground_truth[:k])
    llm_k = llm_ranking[:k]

    if len(llm_k) == 0:
        return 0.0

    tp = sum(1 for c in llm_k if c in gt_k)
    return tp / len(llm_k)


def hallucination_rate(llm_ranking: list[Any], ground_truth: list[Any]) -> float:
    """
    % of hallucinated entries in llm_ranking i.e. entries that are not in ground_truth
    """
    length = len(llm_ranking)

    misses = sum(1 for c in llm_ranking if c not in ground_truth)

    return misses / length if length > 0 else 0.0


def mare_k(llm_ranking: list[Any], ground_truth: list[Any], k: int=None) -> float:
    """
    Mean Absolute Rank Error @ k. Computes the distance between the predicted and true ranks and computes its mean.
    If an element from *llm_ranking* is missing in *ground_truth*, it is penalized with *k* or *len(ground_truth)* (see below).

    :param llm_ranking: predicted ranking list from LLM
    :param ground_truth: ranking list sorted by true similarity
    :param k: consider only the top-k elements for both *llm_ranking* and *ground_truth*.
        If not specified, uses the full length for both; in this case, if an element from *llm_ranking* is still missing
         in *ground_truth*, it is penalized with *len(ground_truth)*.
    :return: MARE@k.
    """
    if k is not None:
        llm = llm_ranking[:k]
        gt = ground_truth[:k]
    else:
        llm = llm_ranking
        gt = ground_truth
    penality = len(ground_truth)
    # Position maps
    llm_pos_position_map = {c: i for i, c in enumerate(llm)}
    gt_position_map = {c: i for i, c in enumerate(gt)}

    errors = [
        abs(gt_position_map[c] - llm_pos_position_map[c]) if c in gt_position_map else penality
        for c in llm
    ]

    return sum(errors) / len(errors)


def spearman_rho_k(llm_ranking: list[Any], ground_truth: list[Any], k: int = None) -> float:
    """
    Spearman rank correlation (ρ) @ k.
    It computes the spearman ranking correlation between *llm_ranking* and *ground_truth* rankings. It penalizes missing
    elements in *llm_ranking* with *k* or *len(ground_truth)* (see below).

    :param llm_ranking: predicted ranking list from the LLM
    :param ground_truth: ranking list sorted by true similarity
    :param k: consider only the top-k elements for both *llm_ranking* and *ground_truth*.
        If not specified, uses the full length of both; in this case, if an element from *llm_ranking* is still missing
            in *ground_truth*, it is penalized with *len(ground_truth)*.
    """
    if k is None:
        llm = llm_ranking
        gt = ground_truth
    else:
        llm = llm_ranking[:k]
        gt = ground_truth[:k]
    penality = len(ground_truth)

    if len(llm_ranking) < 2:
        return 0.0  # cannot compute correlation with <2 points
    # Position lookup
    gt_position_map = {c: i for i, c in enumerate(gt)}
    # Build rank position vectors
    llm_order = [gt_position_map[c] if c in gt else penality
                 for c in llm]
    gt_order = range(len(llm_order))

    # check for constant list
    if all(x == llm_order[0] for x in llm_order):
        return 0.0

    rho, _ = spearmanr(gt_order, llm_order)
    if np.isnan(rho): # constant list, but should not happen due to previous check
        return 0.0
    return float(rho)


def kendall_tau_k(llm_ranking: list[Any], ground_truth: list[Any], k: int = None) -> float:
    """
    Kendall's Tau @ k.
    It computes the kendall tau ranking correlation between *llm_ranking* and *ground_truth* rankings. It penalizes missing
    elements in *llm_ranking* with *k* or *len(ground_truth)* (see below).

    :param llm_ranking: predicted ranking list from the LLM
    :param ground_truth: ranking list sorted by true similarity
    :param k: consider only the top-k elements for both *llm_ranking* and *ground_truth*.
        If not specified, uses the full length of both; in this case, if an element from *llm_ranking* is still missing
            in *ground_truth*, it is penalized with *len(ground_truth)*.
    """
    if k is None:
        llm = llm_ranking
        gt = ground_truth
    else:
        llm = llm_ranking[:k]
        gt = ground_truth[:k]
    penality = len(ground_truth)

    if len(llm_ranking) < 2:
        return 0.0  # cannot compute correlation with <2 points
    # Position lookup
    gt_position_map = {c: i for i, c in enumerate(gt)}
    # Build rank position vectors
    llm_order = [gt_position_map[c] if c in gt else penality
                 for c in llm]
    gt_order = range(len(llm_order))

    from scipy.stats import kendalltau
    tau, _ = kendalltau(gt_order, llm_order)
    if np.isnan(tau):
        return 0.0
    return float(tau)
