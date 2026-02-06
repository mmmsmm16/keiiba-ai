import numpy as np
import pandas as pd
from numba import jit

# Use Numba for speed if possible, or numpy vectorization
# PL Sampling:
# P(i is 1st) = exp(s_i) / sum(exp(s_j))
# P(k is 2nd | i is 1st) = exp(s_k) / sum_{j!=i}(exp(s_j))
# Efficient sampling: Gumbel-Max trick
# s_i + Gumbel(0,1) -> argsort gives ranking.
# Gumbel(0,1) = -log(-log(Uniform(0,1)))

def sample_rankings(strengths, n_samples=20000, seed=42):
    """
    Generate sampled rankings using Gumbel-Max trick (Plackett-Luce equivalent).
    
    Args:
        strengths (np.array): Log-strengths (logit of win prob). Shape (n_horses,)
        n_samples (int): Number of MC samples
        seed (int): Random seed
        
    Returns:
        np.array: Rankings matrix (n_samples, n_horses). 
                  Each row is a permutation of horse INDICES (0 to n-1).
                  Column 0 is the index of the 1st place horse, Col 1 is 2nd, etc.
    """
    rng = np.random.default_rng(seed)
    n_horses = len(strengths)
    
    # Gumbel noise
    # G = -log(-log(U))
    u = rng.random((n_samples, n_horses))
    gumbel = -np.log(-np.log(u + 1e-10) + 1e-10)
    
    # Perturbed scores
    # Score = Strength + Gumbel
    # Sort descending
    perturbed_scores = strengths + gumbel
    
    # Argsort to get rankings (indices of horses)
    # argsort gives ascending, so we take [::-1] or use -perturbed
    rankings = np.argsort(-perturbed_scores, axis=1)
    
    return rankings

def estimate_p_win(rankings, n_horses):
    """
    Estimate P(Win) from rankings.
    """
    n_samples = rankings.shape[0]
    first_place_counts = np.bincount(rankings[:, 0], minlength=n_horses)
    return first_place_counts / n_samples

def estimate_p_place(rankings, n_horses, n_places=3):
    """
    Estimate P(Place) (within top n_places).
    """
    n_samples = rankings.shape[0]
    # Count appearances in first n_places columns
    counts = np.zeros(n_horses)
    for i in range(n_places):
        if i >= rankings.shape[1]: break
        c = np.bincount(rankings[:, i], minlength=n_horses)
        counts += c
    return counts / n_samples

def estimate_p_umaren(rankings, n_horses):
    """
    Estimate P(Umaren) (1st-2nd in any order).
    Returns matrix (n_horses, n_horses) where M[i,j] = P({i,j} is 1st-2nd).
    M[i,j] = M[j,i]. Diagonal is 0.
    """
    n_samples = rankings.shape[0]
    if rankings.shape[1] < 2:
        return np.zeros((n_horses, n_horses))
        
    firsts = rankings[:, 0]
    seconds = rankings[:, 1]
    
    # We want to count pairs {i, j}
    # Create unique pair ID? Or 2D histogram.
    # For N=18, N^2 = 324. Fast enough to iterate samples or use advanced indexing.
    
    # Vectorized approach:
    # Count (first, second) tuples
    # Use flattened index: first * n_horses + second
    dataset = firsts * n_horses + seconds
    counts_flat = np.bincount(dataset, minlength=n_horses*n_horses)
    probs_exact = counts_flat.reshape(n_horses, n_horses) / n_samples
    
    # Umaren is order agnostic: P({i,j}) = P(i then j) + P(j then i)
    # p_umaren[i, j] = probs_exact[i, j] + probs_exact[j, i]
    p_umaren = probs_exact + probs_exact.T
    return p_umaren

def estimate_p_wakuren(rankings, horse_to_frame, n_frames=8):
    """
    Estimate P(Wakuren) (1st-2nd frames in any order).
    
    Args:
        rankings: (n_samples, n_horses) indices
        horse_to_frame: array-like of shape (n_horses,), mapping horse_idx -> frame_number (1-8)
        n_frames: max frame number (usually 8)
        
    Returns:
        matrix (n_frames+1, n_frames+1) (using 1-based indexing for convenience, 0 unused)
    """
    n_samples = rankings.shape[0]
    if rankings.shape[1] < 2:
        return np.zeros((n_frames+1, n_frames+1))
        
    first_indices = rankings[:, 0]
    second_indices = rankings[:, 1]
    
    # Map to frames
    # Assuming horse_to_frame is array/list where index is horse_idx
    h2f = np.array(horse_to_frame)
    f1 = h2f[first_indices]
    f2 = h2f[second_indices]
    
    # Count pairs (f1, f2)
    # Frames are 1-8.
    # Flat index: f1 * (n_frames+1) + f2
    dim = n_frames + 1
    dataset = f1 * dim + f2
    counts = np.bincount(dataset, minlength=dim*dim)
    probs_exact = counts.reshape(dim, dim) / n_samples
    
    # Wakuren logic:
    # If f1 != f2: P({f1, f2}) = P(f1, f2) + P(f2, f1)
    # If f1 == f2: P({f1, f1}) = P(f1, f1) (already captured)
    # Note: JRA Wakuren usually requires 5+ horses running? If small field, Wakuren might not exist.
    # But probability wise, defined if 1st and 2nd horses have defined frames.
    
    p_wakuren = np.zeros((dim, dim))
    for i in range(1, dim):
        for j in range(1, dim):
            if i == j:
                p_wakuren[i, j] = probs_exact[i, j]
            elif i < j:
                p_wakuren[i, j] = probs_exact[i, j] + probs_exact[j, i]
                p_wakuren[j, i] = p_wakuren[i, j] # Symmetric for ease
                
    return p_wakuren

if __name__ == "__main__":
    # Simple sanity check
    s = np.array([1.0, 0.5, 0.0]) # exp(1)=2.7, exp(0.5)=1.6, exp(0)=1
    r = sample_rankings(s, n_samples=10000)
    print("Win Probs:", estimate_p_win(r, 3))
    print("Exacta 0-1:", np.mean((r[:,0]==0) & (r[:,1]==1)))
