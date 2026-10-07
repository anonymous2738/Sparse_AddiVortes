//! Shared-tessellation ensemble with vector-valued cell outputs.

use crate::data::Data;
use crate::maths;
use crate::moves::{self, Prior};
use crate::rng::{self, Rng};
use crate::tessellation::{Assignment, Tessellation};

#[derive(Debug, Clone)]
pub(crate) struct VectorEnsemble {
    prior: Prior,
    width: usize,
    tessellations: Vec<Tessellation>,
    assignments: Vec<Assignment>,
    values: Vec<Vec<f64>>,
    total: Vec<f64>,
    sigma_mu_sq: f64,
}

impl VectorEnsemble {
    pub(crate) fn from_tessellations(
        prior: Prior,
        x: &Data,
        tessellations: Vec<Tessellation>,
        width: usize,
        sigma_mu_sq: f64,
    ) -> Self {
        assert!(width > 0);
        let assignments = tessellations
            .iter()
            .map(|t| Assignment::full(x, t, &prior.geometry))
            .collect();
        let values = tessellations
            .iter()
            .map(|t| vec![0.0; t.n_cells() * width])
            .collect();
        Self {
            prior,
            width,
            tessellations,
            assignments,
            values,
            total: vec![0.0; x.n_rows() * width],
            sigma_mu_sq,
        }
    }

    pub(crate) fn values(&self) -> &[Vec<f64>] {
        &self.values
    }

    pub(crate) fn tessellations(&self) -> &[Tessellation] {
        &self.tessellations
    }

    pub(crate) fn total(&self) -> &[f64] {
        &self.total
    }

    pub(crate) fn sweep(&mut self, x: &Data, response: &[f64], rng: &mut Rng) {
        let n = x.n_rows();
        assert_eq!(response.len(), n * self.width);
        for j in 0..self.tessellations.len() {
            let current = &self.tessellations[j];
            let cells = &self.assignments[j].cells;
            let b = self.tessellations[j].n_cells();
            for i in 0..n {
                let cell = cells[i];
                for coordinate in 0..self.width {
                    self.total[i * self.width + coordinate] -=
                        self.values[j][cell * self.width + coordinate];
                }
            }
            let mut residuals = vec![0.0; n * self.width];
            for i in 0..n {
                for coordinate in 0..self.width {
                    residuals[i * self.width + coordinate] =
                        response[i * self.width + coordinate]
                            - self.total[i * self.width + coordinate];
                }
            }
            let selected = moves::select(current, &self.prior, rng);
            let mut proposal_tessellation = Tessellation::empty();
            let proposal = moves::propose(
                selected,
                current,
                &self.prior,
                rng,
                &mut proposal_tessellation,
            );
            let mut proposal_assignment = Assignment::default();
            let mut keys = Vec::new();
            self.assignments[j].updated_into(
                x,
                &proposal_tessellation,
                proposal.delta,
                &self.prior.geometry,
                &mut keys,
                &mut proposal_assignment,
            );
            let current_log = self.log_marginal(&self.assignments[j].cells, &residuals, b);
            let proposed_log = self.log_marginal(
                &proposal_assignment.cells,
                &residuals,
                proposal_tessellation.n_cells(),
            );
            let log_alpha = proposed_log - current_log
                + proposal.log_structure_ratio
                + moves::log_subset_prior_ratio(
                    current,
                    &proposal_tessellation,
                    &self.prior,
                    false,
                )
                + moves::log_selection_ratio(
                    selected,
                    current,
                    &proposal_tessellation,
                    &self.prior,
                );
            if maths::ln(rng::uniform(rng)) < log_alpha {
                self.tessellations[j] = proposal_tessellation;
                self.assignments[j] = proposal_assignment;
            }
            let cells = &self.assignments[j].cells;
            let b = self.tessellations[j].n_cells();
            let mut sums = vec![0.0; b * self.width];
            let mut counts = vec![0usize; b];
            for i in 0..n {
                let cell = cells[i];
                counts[cell] += 1;
                for coordinate in 0..self.width {
                    sums[cell * self.width + coordinate] +=
                        residuals[i * self.width + coordinate];
                }
            }
            self.values[j].resize(b * self.width, 0.0);
            for cell in 0..b {
                let denominator = 1.0 + counts[cell] as f64 * self.sigma_mu_sq;
                let variance = self.sigma_mu_sq / denominator;
                for coordinate in 0..self.width {
                    let mean =
                        self.sigma_mu_sq * sums[cell * self.width + coordinate] / denominator;
                    self.values[j][cell * self.width + coordinate] =
                        mean + variance.sqrt() * rng::standard_normal(rng);
                }
            }
            for i in 0..n {
                for coordinate in 0..self.width {
                    let cell = self.assignments[j].cells[i];
                    self.total[i * self.width + coordinate] +=
                        self.values[j][cell * self.width + coordinate];
                }
            }
        }
    }

    fn log_marginal(&self, cells: &[usize], residuals: &[f64], b: usize) -> f64 {
        let mut counts = vec![0usize; b];
        let mut sums = vec![0.0; b * self.width];
        for (i, &cell) in cells.iter().enumerate() {
            counts[cell] += 1;
            for coordinate in 0..self.width {
                sums[cell * self.width + coordinate] +=
                    residuals[i * self.width + coordinate];
            }
        }
        counts
            .iter()
            .enumerate()
            .map(|(cell, &count)| {
                let denominator = 1.0 + count as f64 * self.sigma_mu_sq;
                let fit: f64 = (0..self.width)
                    .map(|coordinate| {
                        let sum = sums[cell * self.width + coordinate];
                        self.sigma_mu_sq * sum * sum / denominator
                    })
                    .sum();
                -(self.width as f64) * 0.5 * maths::ln(denominator) + 0.5 * fit
            })
            .sum()
    }
}
