//! The nominal label-symmetric categorical outcome.

use crate::outcome::{OutcomeModel, RequiredData, Sigma2Mode};
use crate::rng::Rng;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CategoricalOutcome {
    labels: Vec<f64>,
    categories: usize,
}

impl CategoricalOutcome {
    pub(crate) fn new(categories: usize) -> Self {
        Self {
            labels: Vec::new(),
            categories,
        }
    }

    pub(crate) fn categories(&self) -> usize {
        self.categories
    }
    pub(crate) fn labels(&self) -> &[f64] {
        &self.labels
    }
}

impl OutcomeModel for CategoricalOutcome {
    fn required_data(&self) -> RequiredData {
        RequiredData::Categorical
    }
    fn init(&mut self, y: &[f64]) {
        self.labels = y.to_vec();
    }
    fn draw_extra(&mut self, _y: &[f64], _total: &[f64], _precision: &[f64], _rng: &mut Rng) {}
    fn working_response(
        &mut self,
        _total: &[f64],
        _precision: &[f64],
        _y: &mut [f64],
        _rng: &mut Rng,
    ) {
    }
    fn weights(&self) -> Option<&[f64]> {
        None
    }
    fn sigma2_mode(&self) -> Sigma2Mode {
        Sigma2Mode::Fixed(1.0)
    }
    fn predictive_quantile(&self, _mean: f64, _sd: f64, _p: f64) -> Option<f64> {
        None
    }
}

#[cfg(feature = "experimental")]
pub(crate) fn choice_probabilities(mean: &[f64], simulations: usize, rng: &mut Rng) -> Vec<f64> {
    let mut probabilities = vec![0.0; mean.len()];
    let mut utilities = vec![0.0; mean.len()];
    for _ in 0..simulations {
        for (utility, &location) in utilities.iter_mut().zip(mean) {
            *utility = location + crate::rng::standard_normal(rng);
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
