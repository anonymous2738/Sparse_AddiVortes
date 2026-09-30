//! The outcome-model layer: the two questions the kernel asks per sweep,
//! and the scale mode from which every validity rule is derived.
//!
//! The kernel asks an outcome model two questions each sweep: what is the
//! working response the ensembles fit against, and what per-observation
//! weights enter the precisions. A latent-normal model answers the first
//! and leaves the weights uniform; a scale-mixture model does the reverse.
//! An outcome model answers those two questions and nothing else: it never
//! touches the backfitting loop, the structural moves, the scaling or the
//! draw collection.
//!
//! [`Sigma2Mode`] is the single source of truth for the model's Gaussian
//! scale. Whether a variance ensemble may attach, whether a global sigma^2
//! is drawn, and whether the fit stores sigma^2 draws are derived from it,
//! never declared per model: an ensemble needs a sampled scale to attach
//! to, so it is available exactly when the mode is [`Sigma2Mode::Sampled`].

use crate::rng::Rng;

/// What the outcome model does with the Gaussian scale sigma^2.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Sigma2Mode {
    /// sigma^2 is a parameter with a prior, drawn from its conditional;
    /// a variance ensemble may replace the global draw.
    Sampled,
    /// sigma^2 is fixed at the given value because the scale is not
    /// identified (the probit model fixes 1 on the latent scale).
    Fixed(f64),
    /// The model has no Gaussian scale.
    Absent,
}

impl Sigma2Mode {
    /// Whether a variance ensemble may attach: it multiplies a sampled
    /// scale, so exactly under [`Sigma2Mode::Sampled`].
    pub(crate) fn permits_variance_ensemble(self) -> bool {
        matches!(self, Sigma2Mode::Sampled)
    }

    /// Whether a global sigma^2 is drawn and stored, in the absence of a
    /// variance ensemble.
    pub(crate) fn samples_global_sigma_sq(self) -> bool {
        matches!(self, Sigma2Mode::Sampled)
    }
}

/// The response contract an outcome model imposes, checked at fit. A
/// `Continuous` response is min-max scaled to [-0.5, 0.5] over its
/// training range; a `Binary` response holds labels in {0, 1} and an
/// `Ordinal` response integer category codes, neither scaled, the model
/// working on its latent scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RequiredData {
    Continuous,
    Binary,
    #[cfg(feature = "experimental")]
    Categorical,
    #[cfg(feature = "experimental")]
    Ordinal,
}

/// An observation model behind the mean ensemble.
///
/// The contract: the kernel calls [`draw_extra`](OutcomeModel::draw_extra),
/// then [`working_response`](OutcomeModel::working_response), then reads
/// [`weights`](OutcomeModel::weights), once per sweep and in that order.
/// The scale of the precisions comes from
/// [`sigma2_mode`](OutcomeModel::sigma2_mode): a kernel-side conjugate
/// draw under [`Sigma2Mode::Sampled`], the fixed value under
/// [`Sigma2Mode::Fixed`], the variance ensemble's product when one is
/// attached.
pub(crate) trait OutcomeModel: std::fmt::Debug {
    /// The response contract, checked at fit before any state exists.
    fn required_data(&self) -> RequiredData;

    /// Resolve data-dependent parameters and initial state from the
    /// response on the model's working scale (the offset and the latent
    /// initialisation for the probit model). Called once at construction.
    fn init(&mut self, y: &[f64]);

    /// Draw the model's own parameters beyond the cells and the scale
    /// (cutpoints, scale-mixture weights, a contamination weight); a
    /// no-op for a model without any. `y` is the current working
    /// response, `total` the mean ensemble's total at the training rows
    /// and `precision` the standing per-observation precisions, the
    /// state such a conditional may read; the draw runs before the
    /// working-response refresh, so a parameter drawn with the latents
    /// integrated out composes with the refresh into one joint draw of
    /// the parameter and the latents.
    fn draw_extra(&mut self, y: &[f64], total: &[f64], precision: &[f64], rng: &mut Rng);

    /// Write this sweep's working response into `y`, given the mean
    /// ensemble's current total at the training rows and this sweep's
    /// per-observation precisions (already written for the sweep, so a
    /// latent refresh sees the same noise the backfit will). The identity
    /// for a model whose response is observed; a truncated-normal refresh
    /// for a latent-normal model. Not called under prior-only sampling,
    /// where the precisions are zero.
    fn working_response(&mut self, total: &[f64], precision: &[f64], y: &mut [f64], rng: &mut Rng);

    /// The per-observation weights of this sweep's precisions: `None` for
    /// unit weight on every observation (the latent-normal models), one
    /// weight per observation for a scale mixture.
    fn weights(&self) -> Option<&[f64]>;

    /// What the model does with sigma^2; every scale validity rule is
    /// derived from this value.
    fn sigma2_mode(&self) -> Sigma2Mode;

    /// Quantile `p` of the predictive distribution of one new observation
    /// given the mean `mean` and the scale `sd` at its row; `None` for a
    /// model without a continuous predictive distribution.
    fn predictive_quantile(&self, mean: f64, sd: f64, p: f64) -> Option<f64>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng;

    #[test]
    fn variance_permission_and_sampling_derive_from_the_mode() {
        assert!(Sigma2Mode::Sampled.permits_variance_ensemble());
        assert!(Sigma2Mode::Sampled.samples_global_sigma_sq());
        assert!(!Sigma2Mode::Fixed(1.0).permits_variance_ensemble());
        assert!(!Sigma2Mode::Fixed(1.0).samples_global_sigma_sq());
        assert!(!Sigma2Mode::Absent.permits_variance_ensemble());
        assert!(!Sigma2Mode::Absent.samples_global_sigma_sq());
    }

    /// A minimal observed-response model exercising the contract's call
    /// order.
    #[derive(Debug, Default)]
    struct Identity {
        initialised: bool,
    }

    impl OutcomeModel for Identity {
        fn required_data(&self) -> RequiredData {
            RequiredData::Continuous
        }

        fn init(&mut self, _y: &[f64]) {
            self.initialised = true;
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
            Sigma2Mode::Sampled
        }

        fn predictive_quantile(&self, mean: f64, _sd: f64, p: f64) -> Option<f64> {
            (p == 0.5).then_some(mean)
        }
    }

    #[test]
    fn the_contract_is_callable_in_sweep_order() {
        let mut model = Identity::default();
        let mut rng = rng::chain_rng(1);
        let mut y = vec![0.25, -0.25];
        model.init(&y);
        assert!(model.initialised);
        model.draw_extra(&y, &[0.0, 0.0], &[1.0, 1.0], &mut rng);
        model.working_response(&[0.0, 0.0], &[1.0, 1.0], &mut y, &mut rng);
        assert_eq!(y, vec![0.25, -0.25]);
        assert_eq!(model.weights(), None);
        assert_eq!(model.sigma2_mode(), Sigma2Mode::Sampled);
        assert_eq!(model.required_data(), RequiredData::Continuous);
        assert_ne!(model.required_data(), RequiredData::Binary);
        assert_eq!(model.predictive_quantile(0.3, 1.0, 0.5), Some(0.3));
        assert_eq!(model.predictive_quantile(0.3, 1.0, 0.9), None);
    }
}
