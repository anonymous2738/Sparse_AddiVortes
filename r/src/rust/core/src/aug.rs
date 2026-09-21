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

    let nmask = 1 << d_len;
    let mut log_f = vec![f64::NEG_INFINITY; nmask];
    let mut sum_w = vec![0.0_f64; nmask];

    log_f[0] = 0.0;

    for mask in 1..nmask {
        let bit_index = mask.trailing_zeros() as usize;
        let prev = mask ^ (1 << bit_index);
        sum_w[mask] = sum_w[prev] + f64::max(weights[dims[bit_index]], 1e-16);
    }

    for mask in 1..nmask {
        let mut max_lp = f64::NEG_INFINITY;

        for j in 0..d_len {
            let bit = 1 << j;
            if (mask & bit) == 0 {
                continue;
            }
            let prev = mask ^ bit;
            let s_j = f64::max(weights[dims[j]], 1e-16);
            let denom = f64::max(1.0 - sum_w[prev], 1e-16);
            let lp = log_f[prev] + crate::maths::ln(s_j) - crate::maths::ln(denom);
            if lp > max_lp {
                max_lp = lp;
            }
        }

        let mut sum_exp = 0.0;
        for j in 0..d_len {
            let bit = 1 << j;
            if (mask & bit) == 0 {
                continue;
            }
            let prev = mask ^ bit;
            let s_j = f64::max(weights[dims[j]], 1e-16);
            let denom = f64::max(1.0 - sum_w[prev], 1e-16);
            let lp = log_f[prev] + crate::maths::ln(s_j) - crate::maths::ln(denom);
            sum_exp += crate::maths::exp(lp - max_lp);
        }
        log_f[mask] = max_lp + crate::maths::ln(sum_exp);
    }

    log_f[nmask - 1]
}

pub(crate) fn sample_order_given_unordered_set(dims: &[usize], weights: &[f64], rng: &mut crate::rng::Rng) -> Vec<usize> {
    let d_len = dims.len();
    if d_len == 0 {
        return Vec::new();
    }
    if d_len > 25 {
        panic!("Exact latent-order sampling is too expensive for d > 25.");
    }

    let nmask = 1 << d_len;
    let mut log_f = vec![f64::NEG_INFINITY; nmask];
    let mut sum_w = vec![0.0_f64; nmask];

    log_f[0] = 0.0;

    for mask in 1..nmask {
        let bit_index = mask.trailing_zeros() as usize;
        let prev = mask ^ (1 << bit_index);
        sum_w[mask] = sum_w[prev] + f64::max(weights[dims[bit_index]], 1e-16);
    }

    for mask in 1..nmask {
        let mut max_lp = f64::NEG_INFINITY;

        for j in 0..d_len {
            let bit = 1 << j;
            if (mask & bit) == 0 {
                continue;
            }
            let prev = mask ^ bit;
            let s_j = f64::max(weights[dims[j]], 1e-16);
            let denom = f64::max(1.0 - sum_w[prev], 1e-16);
            let lp = log_f[prev] + crate::maths::ln(s_j) - crate::maths::ln(denom);
            if lp > max_lp {
                max_lp = lp;
            }
        }

        let mut sum_exp = 0.0;
        for j in 0..d_len {
            let bit = 1 << j;
            if (mask & bit) == 0 {
                continue;
            }
            let prev = mask ^ bit;
            let s_j = f64::max(weights[dims[j]], 1e-16);
            let denom = f64::max(1.0 - sum_w[prev], 1e-16);
            let lp = log_f[prev] + crate::maths::ln(s_j) - crate::maths::ln(denom);
            sum_exp += crate::maths::exp(lp - max_lp);
        }
        log_f[mask] = max_lp + crate::maths::ln(sum_exp);
    }

    let mut order = vec![0; d_len];
    let mut mask = nmask - 1;
    let mut log_probs = Vec::with_capacity(d_len);
    let mut candidate_bits = Vec::with_capacity(d_len);
    let mut probs = Vec::with_capacity(d_len);

    for pos in (0..d_len).rev() {
        log_probs.clear();
        candidate_bits.clear();
        let mut max_lp = f64::NEG_INFINITY;

        for j in 0..d_len {
            let bit = 1 << j;
            if (mask & bit) == 0 {
                continue;
            }
            let prev = mask ^ bit;
            let s_j = f64::max(weights[dims[j]], 1e-16);
            let denom = f64::max(1.0 - sum_w[prev], 1e-16);
            let lp = log_f[prev] + crate::maths::ln(s_j) - crate::maths::ln(denom);

            log_probs.push(lp);
            candidate_bits.push(j);
            if lp > max_lp {
                max_lp = lp;
            }
        }

        let mut sum_exp = 0.0;
        probs.clear();
        for &lp in &log_probs {
            let p = crate::maths::exp(lp - max_lp);
            probs.push(p);
            sum_exp += p;
        }

        let mut u = crate::rng::uniform(rng) * sum_exp;
        let mut selected_idx = 0;
        for (i, &p) in probs.iter().enumerate() {
            u -= p;
            if u <= 0.0 {
                selected_idx = i;
                break;
            }
        }
        if selected_idx >= candidate_bits.len() {
            selected_idx = candidate_bits.len() - 1;
        }

        let selected_bit = candidate_bits[selected_idx];
        order[pos] = dims[selected_bit];
        mask ^= 1 << selected_bit;
    }

    order
}

pub(crate) fn rmultinom(n: u64, probs: &[f64], rng: &mut crate::rng::Rng) -> Vec<u64> {
    let mut counts = vec![0; probs.len()];
    if probs.len() <= 1 {
        if let Some(count) = counts.first_mut() {
            *count = n;
        }
        return counts;
    }
    let mut sum_p = 0.0;
    for &p in probs {
        sum_p += p;
    }
    if sum_p <= 0.0 {
        return counts;
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
    counts
}
