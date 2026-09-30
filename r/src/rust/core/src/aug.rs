use std::cell::RefCell;

// Reused across calls instead of allocating fresh on every invocation: this
// DP runs once per structural-move proposal and once per tessellation per
// sweep, so the two O(2^d) buffers are the hottest allocation in the DART
// path. Each OS thread (chains run on scoped threads, see `threads.rs`) gets
// its own buffers, so this stays race-free under parallel chains.
thread_local! {
    static LOG_F: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    static SUM_W: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    static LN_DENOM: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    static LN_S: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

/// Accumulate one more log-term into a running logsumexp without
/// rescanning the terms seen so far: the standard online rescaling, kept
/// out of line so the DP's inner loop reads as the recurrence it is.
#[inline]
fn logsumexp_add(max_lp: &mut f64, sum_exp: &mut f64, lp: f64) {
    if lp > *max_lp {
        if max_lp.is_finite() {
            *sum_exp *= crate::maths::exp(*max_lp - lp);
        }
        *max_lp = lp;
        *sum_exp += 1.0;
    } else {
        *sum_exp += crate::maths::exp(lp - *max_lp);
    }
}

/// Fills `log_f`/`sum_w` with the DP tables over all `2^d_len` subsets of
/// `dims`; shared by `log_unordered_subset_prior` and
/// `sample_order_given_unordered_set` so the recurrence is written once.
///
/// `ln(s_j)` depends only on `j` and `ln(1 - sum_w[prev])` only on `prev`,
/// so both are precomputed once per mask/covariate (`O(2^d)` and `O(d)`
/// logs) rather than recomputed on every one of the `O(d * 2^d)` DP
/// transitions; the max and sum-exp passes are folded into one online
/// logsumexp. Together these cut the transcendental calls on the DP's hot
/// loop roughly fourfold.
fn build_unordered_subset_dp(
    dims: &[usize],
    weights: &[f64],
    log_f: &mut Vec<f64>,
    sum_w: &mut Vec<f64>,
    ln_denom: &mut Vec<f64>,
    ln_s: &mut Vec<f64>,
) {
    let d_len = dims.len();
    let nmask = 1usize << d_len;

    log_f.clear();
    log_f.resize(nmask, f64::NEG_INFINITY);
    sum_w.clear();
    sum_w.resize(nmask, 0.0);
    ln_denom.clear();
    ln_denom.resize(nmask, 0.0);

    ln_s.clear();
    ln_s.extend(dims.iter().map(|&d| crate::maths::ln(f64::max(weights[d], 1e-16))));

    log_f[0] = 0.0;
    ln_denom[0] = crate::maths::ln(f64::max(1.0 - sum_w[0], 1e-16));

    for mask in 1..nmask {
        let bit_index = mask.trailing_zeros() as usize;
        let prev = mask ^ (1 << bit_index);
        sum_w[mask] = sum_w[prev] + f64::max(weights[dims[bit_index]], 1e-16);
        ln_denom[mask] = crate::maths::ln(f64::max(1.0 - sum_w[mask], 1e-16));
    }

    for mask in 1..nmask {
        let mut max_lp = f64::NEG_INFINITY;
        let mut sum_exp = 0.0;

        for j in 0..d_len {
            let bit = 1 << j;
            if (mask & bit) == 0 {
                continue;
            }
            let prev = mask ^ bit;
            let lp = log_f[prev] + ln_s[j] - ln_denom[prev];
            logsumexp_add(&mut max_lp, &mut sum_exp, lp);
        }
        log_f[mask] = max_lp + crate::maths::ln(sum_exp);
    }
}

/// The Model B unordered subset prior: log F(D) where
/// F(D) = sum over orderings h of D: product_t s_{h_t} / (1 - sum_{r<t} s_{h_r}).
/// This is the marginal probability of selecting the unordered set D
/// under without-replacement sequential selection with weights s.
/// Mirrors C++ `log_unordered_dimension_prior` exactly.
/// O(d * 2^d), fine for d <= 25.
pub(crate) fn log_unordered_subset_prior(dims: &[usize], weights: &[f64]) -> f64 {
    let d_len = dims.len();
    if d_len == 0 {
        return 0.0;
    }
    if d_len > 25 {
        panic!("Exact unordered Model B prior is too expensive for d > 25.");
    }

    LOG_F.with(|log_f_cell| {
        SUM_W.with(|sum_w_cell| {
            LN_DENOM.with(|ln_denom_cell| {
                LN_S.with(|ln_s_cell| {
                    let mut log_f = log_f_cell.borrow_mut();
                    let mut sum_w = sum_w_cell.borrow_mut();
                    let mut ln_denom = ln_denom_cell.borrow_mut();
                    let mut ln_s = ln_s_cell.borrow_mut();
                    build_unordered_subset_dp(
                        dims, weights, &mut log_f, &mut sum_w, &mut ln_denom, &mut ln_s,
                    );
                    log_f[(1 << d_len) - 1]
                })
            })
        })
    })
}

#[derive(Default)]
pub(crate) struct OrderWorkspace {
    order: Vec<usize>,
    probabilities: Vec<f64>,
    candidate_bits: Vec<usize>,
}

pub(crate) fn sample_order_given_unordered_set<'a>(
    dims: &[usize],
    weights: &[f64],
    rng: &mut crate::rng::Rng,
    workspace: &'a mut OrderWorkspace,
) -> &'a [usize] {
    let d_len = dims.len();
    if d_len == 0 {
        workspace.order.clear();
        return &workspace.order;
    }
    if d_len > 25 {
        panic!("Exact latent-order sampling is too expensive for d > 25.");
    }
    if d_len == 1 {
        let _ = crate::rng::uniform(rng);
        workspace.order.clear();
        workspace.order.push(dims[0]);
        return &workspace.order;
    }

    LOG_F.with(|log_f_cell| {
        SUM_W.with(|sum_w_cell| {
            LN_DENOM.with(|ln_denom_cell| {
                LN_S.with(|ln_s_cell| {
                    let mut log_f = log_f_cell.borrow_mut();
                    let mut sum_w = sum_w_cell.borrow_mut();
                    let mut ln_denom = ln_denom_cell.borrow_mut();
                    let mut ln_s = ln_s_cell.borrow_mut();
                    build_unordered_subset_dp(
                        dims, weights, &mut log_f, &mut sum_w, &mut ln_denom, &mut ln_s,
                    );

                    workspace.order.resize(d_len, 0);
                    let mut mask = (1usize << d_len) - 1;

                    for pos in (0..d_len).rev() {
                        workspace.probabilities.clear();
                        workspace.candidate_bits.clear();
                        let mut max_lp = f64::NEG_INFINITY;
                        let mut sum_exp = 0.0;

                        for j in 0..d_len {
                            let bit = 1 << j;
                            if (mask & bit) == 0 {
                                continue;
                            }
                            let prev = mask ^ bit;
                            let lp = log_f[prev] + ln_s[j] - ln_denom[prev];
                            logsumexp_add(&mut max_lp, &mut sum_exp, lp);
                            workspace.probabilities.push(lp);
                            workspace.candidate_bits.push(j);
                        }
                        for p in &mut workspace.probabilities {
                            *p = crate::maths::exp(*p - max_lp);
                        }

                        let mut u = crate::rng::uniform(rng) * sum_exp;
                        let mut selected_idx = 0;
                        for (i, &p) in workspace.probabilities.iter().enumerate() {
                            u -= p;
                            if u <= 0.0 {
                                selected_idx = i;
                                break;
                            }
                        }
                        if selected_idx >= workspace.candidate_bits.len() {
                            selected_idx = workspace.candidate_bits.len() - 1;
                        }

                        let selected_bit = workspace.candidate_bits[selected_idx];
                        workspace.order[pos] = dims[selected_bit];
                        mask ^= 1 << selected_bit;
                    }

                    workspace.order.as_slice()
                })
            })
        })
    })
}


pub(crate) fn rmultinom_into(
    n: u64,
    probs: &[f64],
    rng: &mut crate::rng::Rng,
    counts: &mut Vec<u64>,
) {
    counts.clear();
    counts.resize(probs.len(), 0);
    if probs.len() <= 1 {
        if let Some(count) = counts.first_mut() {
            *count = n;
        }
        return;
    }
    let mut sum_p = 0.0;
    for &p in probs {
        sum_p += p;
    }
    if sum_p <= 0.0 {
        return;
    }

    // Draw the multinomial through sequential binomials. This has the same
    // distribution as n categorical draws, but avoids work proportional to n
    // when adaptive DART produces a large augmented failure count.
    let mut remaining = n;
    let mut remaining_probability = sum_p;
    for (index, &probability) in probs.iter().enumerate().take(probs.len() - 1) {
        if remaining == 0 {
            break;
        }
        let conditional_probability = (probability / remaining_probability).clamp(0.0, 1.0);
        counts[index] = crate::rng::binomial(remaining, conditional_probability, rng);
        remaining -= counts[index];
        remaining_probability -= probability;
    }
    if let Some(last) = counts.last_mut() {
        *last = remaining;
    }
}
