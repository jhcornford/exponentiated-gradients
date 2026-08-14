import numpy as np

def generate_noisy_r_of_k_data(n_datapoints, n, k, r,
                               p_rel=0.5, p_irrel="same", seed=None, shuffle=False,
                               noise_probs=None, w_star=None, neg_wstar=False,
                               verbose=False):
    """
    Generate binary data for the noisy "r of k" task: y = 1 iff <w*, x> >= r,
    where x is a binary vector of n dimensions and w* is supported on k << n
    "relevant" coordinates. Set r=1 for the classic "1 of k" problem.

    X's first k columns are drawn Bernoulli(p_rel) and the remaining n-k
    Bernoulli(p_irrel); the default w* marks exactly that leading block as
    relevant. <w*,x> ~ Binomial(k, p_rel), so classes are roughly balanced when
    r = k*p_rel (the release config: k=100, p_rel=0.5, r=50).

    Args:
        n_datapoints : # of samples to draw.
        n            : # of input dimensions.
        k            : # of task-relevant dimensions (k <= n). Should be even:
                       neg_wstar needs k/2 to be a whole number.
        r            : threshold on <w*, x> for y=1. Note this is >=, not >.
        p_rel        : Bernoulli prob. for the k relevant columns of X.
        p_irrel      : Bernoulli prob. for the n-k irrelevant columns.
                       Currently only the string "same" is accepted, which
                       reuses p_rel; the guard below calls .lower() on this, so
                       passing a float raises AttributeError. Leaving it at
                       "same" makes all n columns i.i.d., which is what
                       `shuffle` relies on (see below).
        seed         : if given, seeds numpy's global RNG.
        shuffle      : cosmetically permute a freshly drawn w* and the columns
                       of X together, so the relevant dimensions aren't a
                       contiguous leading block. Ignored when w_star is
                       supplied. Only distribution-preserving while
                       p_irrel == p_rel, since otherwise the columns are not
                       exchangeable and permuting w* alone would move the
                       signal onto coordinates drawn at the wrong rate.
        noise_probs  : 3-vector of probabilities over {-1, 0, +1}, added to
                       <w*, x> before thresholding to give label noise. None
                       for clean labels.
        w_star       : supply a target vector to reuse across splits (train.py
                       draws it on train and pins val/test to it, so all three
                       share one target function). A supplied w* is used
                       verbatim -- never re-drawn, never shuffled. Because X's
                       relevant block is always its first k columns, a supplied
                       w* whose support lies elsewhere (e.g. one returned by a
                       shuffle=True call) only matches the data-generating
                       distribution while p_irrel == p_rel.
        neg_wstar    : if True, flip half of w*'s k nonzero entries to -1, so
                       <w*, x> is mean-zero. Ignored when w_star is supplied.
        verbose      : print draw probabilities, w* accuracy under noise, and
                       the realised class balance.

    Returns:
        X      : (n_datapoints, n) float array of 0.0/1.0.
        y      : (n_datapoints,) int array of 0/1 labels.
        w_star : (n,) float array -- the target vector actually used.
    """
    if seed is not None: np.random.seed(seed)
    assert k <= n
    if k % 2: print("WARNING: Intended to run with k being even for data balance ")
    if p_irrel.lower() == "same": p_irrel=p_rel
        
    # Generate data
    if verbose: print(f"Drawing data with probability {p_rel}")
    X_k = np.random.binomial(n=1, p=p_rel,   size=(n_datapoints,k))
    X_n = np.random.binomial(n=1, p=p_irrel, size=(n_datapoints,n-k))
    X = np.concatenate([X_k, X_n], axis=1)
    # Generate "target" vector, w_star
    if w_star is None:
        w_star = np.concatenate([np.ones(k), np.zeros(n-k)], axis=0)
        if neg_wstar:
            neg_inds = np.random.choice(range(k), size=int(k/2), replace =False)
            if verbose: print(int(k/2), neg_inds)
            w_star[neg_inds] =  w_star[neg_inds]*-1
        if verbose: print("w_star: ", w_star[:20])

        if shuffle:
            # Cosmetic only: while p_irrel == p_rel every column of X is i.i.d.
            # Bernoulli(p), so which coordinates carry the signal is arbitrary.
            # Permuting w_star and X's columns together just relabels the axes --
            # the learning problem is identical, the relevant dims simply stop
            # being a contiguous leading block when you plot the weights.
            shuffle_idxs = np.random.permutation(n)
            w_star = w_star[shuffle_idxs]
            X = X[:,shuffle_idxs]

    # Note the shuffle above sits inside `w_star is None`: a supplied w_star is
    # used verbatim, so train/val/test all share one target function. Since X's
    # first k columns are always the p_rel block, a supplied w_star supported
    # elsewhere (e.g. one returned by a shuffle=True call) would put the signal
    # on columns drawn at p_irrel -- harmless while p_irrel == p_rel, wrong if not.

    if noise_probs is None:
        if verbose: print(" drawing data without noise")
        y = np.matmul(X,w_star) >= r
    else:
        y_true = np.matmul(X,w_star) >= r
        noise_vec = np.random.choice([-1,0,1],size=X.shape[0],p=noise_probs)
        y = (np.matmul(X,w_star) + noise_vec) >=r
        w_star_acc = sum(y_true == y) / X.shape[0]
        # in future maybe work set noise based on chosen error rate
        if verbose:
            print(np.mean(np.matmul(X,w_star)))
            print(f" adding noise {-1,0,1} with prob. {noise_probs}")
            print("w_star_acc is", w_star_acc)
            print(f"% of positive y_true drawn: {y_true.sum()/len(y_true)}")
            print(f"% of positive y drawn: {y.sum()/len(y)}")

    X = X.astype(float)
    y = y.astype(int)
    return X, y, w_star
