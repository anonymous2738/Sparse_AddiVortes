//! The chain's random number generator: ChaCha8 keyed from a `u64` seed
//! through splitmix64, and the draws the sampler makes from it.

use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};
use rand_distr::Distribution;

/// The chain RNG.
pub(crate) type Rng = ChaCha8Rng;

/// splitmix64 (Steele, Lea and Flood 2014), one output per call.
pub(crate) fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The chain RNG for `seed`: the 256-bit ChaCha8 key is four successive
/// splitmix64 outputs from `seed`.
pub(crate) fn chain_rng(seed: u64) -> Rng {
    let mut state = seed;
    let mut key = [0u8; 32];
    for word in key.chunks_exact_mut(8) {
        word.copy_from_slice(&splitmix64(&mut state).to_le_bytes());
    }
    ChaCha8Rng::from_seed(key)
}

/// The seed of chain `index` (0-based) derived from `seed`: chain 0 is
/// `seed` itself, chain k the k-th splitmix64 output after it.
pub fn chain_seed(seed: u64, index: usize) -> u64 {
    let mut state = seed;
    let mut out = seed;
    for _ in 0..index {
        out = splitmix64(&mut state);
    }
    out
}

/// Uniform on [0, 1) with 53 random bits.
pub(crate) fn uniform(rng: &mut Rng) -> f64 {
    (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// Uniform index in `0..n`, n >= 1.
pub(crate) fn uniform_index(n: usize, rng: &mut Rng) -> usize {
    debug_assert!(n >= 1);
    ((uniform(rng) * n as f64) as usize).min(n - 1)
}

/// Standard normal draw.
pub(crate) fn standard_normal(rng: &mut Rng) -> f64 {
    rand_distr::StandardNormal.sample(rng)
}

/// Gamma(shape, scale) draw, shape > 0, scale > 0.
pub(crate) fn gamma(shape: f64, scale: f64, rng: &mut Rng) -> f64 {
    rand_distr::Gamma::new(shape, scale)
        .expect("shape and scale are positive by construction")
        .sample(rng)
}

/// Binomial(successes, probability) draw.
pub(crate) fn binomial(successes: u64, probability: f64, rng: &mut Rng) -> u64 {
    rand_distr::Binomial::new(successes, probability)
        .expect("successes and probability are valid by construction")
        .sample(rng)
}

/// Inverse-Gaussian(mean, shape) draw, both positive: density
/// sqrt(shape / (2 pi x^3)) exp(-shape (x - mean)^2 / (2 mean^2 x)).
#[cfg(feature = "experimental")]
pub(crate) fn inverse_gaussian(mean: f64, shape: f64, rng: &mut Rng) -> f64 {
    rand_distr::InverseGaussian::new(mean, shape)
        .expect("mean and shape are positive by construction")
        .sample(rng)
}

/// One index drawn from unnormalised log weights, with one uniform.
#[cfg(feature = "experimental")]
pub(crate) fn draw_discrete(log_weights: &[f64], rng: &mut Rng) -> usize {
    let max = log_weights.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v));
    let weights: Vec<f64> = log_weights.iter().map(|&v| libm::exp(v - max)).collect();
    let total: f64 = weights.iter().sum();
    let target = uniform(rng) * total;
    let mut cumulative = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        cumulative += w;
        if target < cumulative {
            return i;
        }
    }
    weights.len() - 1
}

/// Standard normal draw restricted to z >= a (Robert 1995, Statistics and
/// Computing 5, 121-125): plain rejection from N(0, 1) for a < 0.45,
/// otherwise rejection from the exponential with rate
/// (a + sqrt(a^2 + 4)) / 2 shifted to a, whose acceptance rate does not
/// fall as a grows.
pub(crate) fn truncated_standard_normal_above(a: f64, rng: &mut Rng) -> f64 {
    if a < 0.45 {
        loop {
            let z = standard_normal(rng);
            if z >= a {
                return z;
            }
        }
    }
    let alpha = 0.5 * (a + (a * a + 4.0).sqrt());
    loop {
        // 1 - u lies in (0, 1], so the logarithm is finite.
        let z = a - libm::log(1.0 - uniform(rng)) / alpha;
        let rho = libm::exp(-0.5 * (z - alpha) * (z - alpha));
        if uniform(rng) <= rho {
            return z;
        }
    }
}

/// Standard normal draw restricted to a <= z <= b, a < b, both finite
/// (Robert 1995, section 2). An interval covering 0 takes plain
/// rejection from N(0, 1) when it is at least sqrt(2 pi) wide and
/// uniform rejection against the density maximum at 0 otherwise; an
/// interval with a > 0 takes the one-sided exponential proposal
/// restricted to [a, b] when b lies beyond Robert's crossover and
/// uniform rejection against the density maximum at a otherwise; an
/// interval with b < 0 reflects. Every branch is an exact rejection
/// sampler; the crossover sets only the acceptance rate.
#[cfg(feature = "experimental")]
pub(crate) fn truncated_standard_normal_between(a: f64, b: f64, rng: &mut Rng) -> f64 {
    debug_assert!(a.is_finite() && b.is_finite() && a < b);
    if b <= 0.0 {
        return -truncated_standard_normal_between(-b, -a, rng);
    }
    if a <= 0.0 {
        if b - a >= (2.0 * std::f64::consts::PI).sqrt() {
            loop {
                let z = standard_normal(rng);
                if a <= z && z <= b {
                    return z;
                }
            }
        }
        loop {
            let z = a + (b - a) * uniform(rng);
            if uniform(rng) <= libm::exp(-0.5 * z * z) {
                return z;
            }
        }
    }
    let alpha = 0.5 * (a + (a * a + 4.0).sqrt());
    if b > a + libm::exp(0.5 + 0.25 * (a * a - a * (a * a + 4.0).sqrt())) / alpha {
        loop {
            let z = a - libm::log(1.0 - uniform(rng)) / alpha;
            if z <= b && uniform(rng) <= libm::exp(-0.5 * (z - alpha) * (z - alpha)) {
                return z;
            }
        }
    }
    loop {
        let z = a + (b - a) * uniform(rng);
        if uniform(rng) <= libm::exp(0.5 * (a * a - z * z)) {
            return z;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncated_normal_moments_on_both_branches() {
        // E[Z | Z >= a] = phi(a) / (1 - Phi(a)).
        let mean_above = |a: f64| {
            let phi = (-0.5 * a * a).exp() / (2.0 * std::f64::consts::PI).sqrt();
            phi / (0.5 * libm::erfc(a * std::f64::consts::FRAC_1_SQRT_2))
        };
        let mut rng = chain_rng(9);
        let n = 200_000;
        for a in [-1.5, 0.0, 0.44, 0.46, 2.0, 6.0] {
            let mut sum = 0.0;
            for _ in 0..n {
                let z = truncated_standard_normal_above(a, &mut rng);
                assert!(z >= a);
                sum += z;
            }
            let mean = sum / n as f64;
            assert!(
                (mean - mean_above(a)).abs() < 0.01,
                "a = {a}: {mean} vs {}",
                mean_above(a)
            );
        }
    }

    /// The pairs cover every branch: the plain Normal rejection, the
    /// central and the shifted uniform rejection, the restricted
    /// exponential rejection and the reflection.
    #[cfg(feature = "experimental")]
    #[test]
    fn two_sided_truncated_normal_moments_on_every_branch() {
        // E[Z | a <= Z <= b] = (phi(a) - phi(b)) / (Phi(b) - Phi(a)).
        let phi = |z: f64| (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
        let cdf = |z: f64| 0.5 * libm::erfc(-z * std::f64::consts::FRAC_1_SQRT_2);
        let mut rng = chain_rng(13);
        let n = 200_000;
        for (a, b) in [
            (-3.0, 3.0),
            (-0.5, 0.5),
            (-0.4, 1.8),
            (1.0, 1.2),
            (1.0, 8.0),
            (2.0, 2.05),
            (-1.2, -1.0),
            (-8.0, -1.0),
        ] {
            let mut sum = 0.0;
            for _ in 0..n {
                let z = truncated_standard_normal_between(a, b, &mut rng);
                assert!((a..=b).contains(&z), "({a}, {b}): {z}");
                sum += z;
            }
            let mean = sum / n as f64;
            let expected = (phi(a) - phi(b)) / (cdf(b) - cdf(a));
            assert!(
                (mean - expected).abs() < 0.01,
                "({a}, {b}): {mean} vs {expected}"
            );
        }
    }

    #[test]
    fn same_seed_same_stream() {
        let mut a = chain_rng(42);
        let mut b = chain_rng(42);
        let mut c = chain_rng(43);
        let (ua, ub, uc) = (uniform(&mut a), uniform(&mut b), uniform(&mut c));
        assert_eq!(ua, ub);
        assert_ne!(ua, uc);
    }

    #[test]
    fn chain_seeds_are_distinct_and_start_at_the_seed() {
        assert_eq!(chain_seed(7, 0), 7);
        let seeds: Vec<u64> = (0..4).map(|k| chain_seed(7, k)).collect();
        for i in 0..4 {
            for j in 0..i {
                assert_ne!(seeds[i], seeds[j]);
            }
        }
    }

    #[test]
    fn uniform_index_is_in_range() {
        let mut rng = chain_rng(1);
        for _ in 0..1000 {
            assert!(uniform_index(3, &mut rng) < 3);
            assert_eq!(uniform_index(1, &mut rng), 0);
        }
    }

    /// E[X] = mean and Var[X] = mean^3 / shape.
    #[cfg(feature = "experimental")]
    #[test]
    fn inverse_gaussian_moments() {
        let mut rng = chain_rng(23);
        let n = 100_000;
        let (mean, shape) = (1.5, 2.0);
        let (mut sum, mut sum_sq) = (0.0, 0.0);
        for _ in 0..n {
            let x = inverse_gaussian(mean, shape, &mut rng);
            assert!(x > 0.0);
            sum += x;
            sum_sq += x * x;
        }
        let m = sum / n as f64;
        let v = sum_sq / n as f64 - m * m;
        assert!((m - mean).abs() < 0.02, "{m}");
        assert!((v - mean * mean * mean / shape).abs() < 0.05, "{v}");
    }

    #[cfg(feature = "experimental")]
    #[test]
    fn draw_discrete_frequencies_match_the_weights() {
        let mut rng = chain_rng(17);
        // Unnormalised log weights of probabilities 1/6, 2/6, 3/6, offset
        // to exercise the max shift.
        let log_weights = [100.0, 100.0 + libm::log(2.0), 100.0 + libm::log(3.0)];
        let n = 60_000;
        let mut counts = [0usize; 3];
        for _ in 0..n {
            counts[draw_discrete(&log_weights, &mut rng)] += 1;
        }
        for (count, expected) in counts.iter().zip([1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0]) {
            let share = *count as f64 / n as f64;
            assert!((share - expected).abs() < 0.01, "{share} vs {expected}");
        }
    }

    #[test]
    fn gamma_and_normal_moments() {
        let mut rng = chain_rng(5);
        let n = 20_000;
        let mean_g: f64 = (0..n).map(|_| gamma(4.0, 0.5, &mut rng)).sum::<f64>() / n as f64;
        assert!((mean_g - 2.0).abs() < 0.05, "{mean_g}");
        let mean_z: f64 = (0..n).map(|_| standard_normal(&mut rng)).sum::<f64>() / n as f64;
        assert!(mean_z.abs() < 0.03, "{mean_z}");
    }
}
