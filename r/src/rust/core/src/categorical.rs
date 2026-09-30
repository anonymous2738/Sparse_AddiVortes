//! Latent utility and prediction helpers for nominal categorical outcomes.

use crate::rng::{self, Rng};

pub(crate) fn update_utilities(utilities: &mut [f64], mean: &[f64], label: usize, rng: &mut Rng) {
    assert_eq!(utilities.len(), mean.len());
    assert!(label < utilities.len());
    let k = utilities.len();
    let mut order: Vec<usize> = (0..k).collect();
    for index in (1..k).rev() {
        let selected = (rng::uniform(rng) * (index + 1) as f64) as usize;
        order.swap(index, selected.min(index));
    }
    for coordinate in order {
        if coordinate == label {
            let lower = (0..k)
                .filter(|&other| other != label)
                .map(|other| utilities[other])
                .fold(f64::NEG_INFINITY, f64::max);
            utilities[label] =
                mean[label] + rng::truncated_standard_normal_above(lower - mean[label], rng);
        } else {
            utilities[coordinate] = mean[coordinate]
                - rng::truncated_standard_normal_above(mean[coordinate] - utilities[label], rng);
        }
    }
}

pub(crate) fn choice_probabilities(mean: &[f64], simulations: usize, rng: &mut Rng) -> Vec<f64> {
    let mut probabilities = vec![0.0; mean.len()];
    let mut utilities = vec![0.0; mean.len()];
    for _ in 0..simulations {
        for (utility, &location) in utilities.iter_mut().zip(mean) {
            *utility = location + rng::standard_normal(rng);
        }
        let winner = utilities
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index)
            .expect("categorical model needs at least one class");
        probabilities[winner] += 1.0;
    }
    let scale = 1.0 / simulations as f64;
    probabilities
        .iter_mut()
        .for_each(|probability| *probability *= scale);
    probabilities
}
