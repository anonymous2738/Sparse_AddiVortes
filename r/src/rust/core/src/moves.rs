//! The six structural Metropolis-Hastings moves on a tessellation (Stone
//! and Gosling 2025, s. 2.3 and Appendix B) and their selection table.
//!
//! Each move supplies a proposal, the structural change for the assignment
//! cache, and ln[prior ratio x within-move proposal ratio]. The selection
//! ratio ln q(reverse | proposed) - ln q(move | current) is computed from the
//! table, so the boundary corrections of Appendix B (+- ln 2) follow from the
//! weight folding rather than being written by hand.

use crate::geometry::{CoordinateLaw, Geometry};
use crate::maths;
use crate::rng::{uniform, uniform_index, Rng};
use crate::tessellation::{Delta, Tessellation};

/// The move kinds, in selection-table order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Move {
    AddCentre,
    RemoveCentre,
    AddDimension,
    RemoveDimension,
    Change,
    Swap,
}

const MOVES: [Move; 6] = [
    Move::AddCentre,
    Move::RemoveCentre,
    Move::AddDimension,
    Move::RemoveDimension,
    Move::Change,
    Move::Swap,
];

/// Selection weights before folding (Stone and Gosling 2025, Appendix B).
const WEIGHTS: [f64; 6] = [0.2, 0.2, 0.2, 0.2, 0.1, 0.1];

/// Read-only model state the moves consult.
#[derive(Debug, Clone)]
pub(crate) struct Prior {
    /// Number of covariates p.
    pub p: usize,
    /// Dimension-count prior parameter omega (theta = omega / p).
    pub omega: f64,
    /// Cell-count prior rate lambda_c.
    pub lambda_c: f64,
    /// The metric of each column.
    pub geometry: Geometry,
    /// The centre-coordinate law of each column.
    pub laws: Vec<CoordinateLaw>,
    /// The weighted inclusion prior; `None` is the uniform prior,
    /// including a weight vector whose entries are all equal.
    #[cfg(feature = "experimental")]
    pub weights: Option<InclusionWeights>,
    /// Momentum-adjusted weights used only to choose dimensions in add and
    /// swap proposals. The structural prior always uses `weights`.
    #[cfg(feature = "experimental")]
    pub proposal_weights: Option<InclusionWeights>,
}

/// A fixed inclusion weight per column, used by the DART variable
/// selection prior.
#[cfg(feature = "experimental")]
#[derive(Debug, Clone)]
pub(crate) struct InclusionWeights {
    pub(crate) w: Vec<f64>,
    cumulative: Vec<f64>,
    /// Precomputed once per construction (every sweep, for DART's sampled
    /// weights) so `unused_weight`/`has_positive_unused` need not rescan
    /// all p columns on every move proposal, which matters when p is in
    /// the hundreds or thousands: sum(w) and the count of strictly
    /// positive entries.
    total: f64,
    positive_count: usize,
}

#[cfg(feature = "experimental")]
impl InclusionWeights {
    /// `None` when the entries are all equal: equal weights are the
    /// uniform prior and take its code path.
    pub(crate) fn new(w: &[f64]) -> Option<Self> {
        if w.iter().all(|&v| v == w[0]) {
            return None;
        }
        Some(Self::sampled(w.to_vec()))
    }

    /// Weights taken as given, without the equal-entries fold: sampled
    /// weight vectors keep their own values.
    pub(crate) fn sampled(w: Vec<f64>) -> Self {
        let mut cumulative = Vec::with_capacity(w.len() + 1);
        cumulative.push(0.0);
        let mut total = 0.0;
        let mut positive_count = 0;
        for &weight in &w {
            total += weight;
            positive_count += usize::from(weight > 0.0);
            cumulative.push(total);
        }
        Self {
            w,
            cumulative,
            total,
            positive_count,
        }
    }

    pub(crate) fn replace(&mut self, weights: &[f64]) {
        debug_assert_eq!(self.w.len(), weights.len());
        self.w.copy_from_slice(weights);
        self.cumulative[0] = 0.0;
        self.total = 0.0;
        self.positive_count = 0;
        for (index, &weight) in weights.iter().enumerate() {
            self.total += weight;
            self.positive_count += usize::from(weight > 0.0);
            self.cumulative[index + 1] = self.total;
        }
    }

    /// The total weight of the columns not in `dims`: sum(w) minus the
    /// weight of the (few, already-known) columns in `dims`, rather than
    /// a fresh scan of every column not in `dims`.
    fn unused_weight(&self, dims: &[usize]) -> f64 {
        let used: f64 = dims.iter().map(|&col| self.w[col]).sum();
        f64::max(self.total - used, 1e-16)
    }

    /// Whether any column outside `dims` has positive weight. If more
    /// columns are positive than `dims` can possibly hold, one of them
    /// must lie outside it (pigeonhole), so the common case (DART weights
    /// are all strictly positive) is answered without a scan at all.
    fn has_positive_unused(&self, dims: &[usize]) -> bool {
        if self.positive_count > dims.len() {
            return true;
        }
        self.w
            .iter()
            .enumerate()
            .any(|(col, &v)| v > 0.0 && !dims.contains(&col))
    }

    /// A column drawn with probability proportional to its weight among
    /// the columns not in `dims`, using the cached weighted CDF; returns the
    /// unused total for the proposal ratio.
    fn draw_unused(&self, dims: &[usize], rng: &mut Rng) -> (usize, f64) {
        let total = self.unused_weight(dims);
        let target = uniform(rng) * total;
        let mut lower = 0;
        let mut upper = self.w.len();
        while lower < upper {
            let middle = lower + (upper - lower) / 2;
            let used_before: f64 = dims
                .iter()
                .filter(|&&dim| dim <= middle)
                .map(|&dim| self.w[dim])
                .sum();
            if self.cumulative[middle + 1] - used_before > target {
                upper = middle;
            } else {
                lower = middle + 1;
            }
        }
        if lower < self.w.len() && self.w[lower] > 0.0 && !dims.contains(&lower) {
            return (lower, total);
        }

        let mut cumulative = 0.0;
        for (col, &weight) in self.w.iter().enumerate() {
            if weight <= 0.0 || dims.contains(&col) {
                continue;
            }
            cumulative += weight;
            if target < cumulative {
                return (col, total);
            }
        }
        let last = self
            .w
            .iter()
            .enumerate()
            .rev()
            .find(|(col, weight)| **weight > 0.0 && !dims.contains(col))
            .map(|(col, _)| col)
            .expect("a positive unused weight");
        (last, total)
    }
}

impl Prior {
    /// p Euclidean columns with centre coordinates N(0, sigma_c^2).
    #[cfg(test)]
    pub(crate) fn euclidean(p: usize, omega: f64, lambda_c: f64, sigma_c: f64) -> Self {
        Self {
            p,
            omega,
            lambda_c,
            geometry: Geometry::euclidean(p),
            laws: vec![
                CoordinateLaw::Normal {
                    mean: 0.0,
                    sd: sigma_c
                };
                p
            ],
            #[cfg(feature = "experimental")]
            weights: None,
            #[cfg(feature = "experimental")]
            proposal_weights: None,
        }
    }

    /// ln P(b) - ln P(b - 1) of the cell-count prior b - 1 ~ Poisson(lambda_c),
    /// at the larger count b >= 2: ln lambda_c - ln(b - 1).
    fn log_cell_count_ratio(&self, b: usize) -> f64 {
        maths::ln(self.lambda_c) - maths::ln((b - 1) as f64)
    }

    /// ln P(d) - ln P(d - 1) of the dimension-count prior
    /// d - 1 ~ Binomial(p - 1, theta), theta = omega / p, at the larger count
    /// d >= 2: ln(p - d + 1) - ln(d - 1) + ln theta - ln(1 - theta). The
    /// p - 1 trials make the uniform-subset factor 1 / C(p, d) cancel the
    /// uniform covariate pick, so the ratio below is complete.
    fn log_dim_count_ratio(&self, d: usize) -> f64 {
        let p = self.p as f64;
        let d = d as f64;
        let theta = self.omega / p;
        maths::ln(p - d + 1.0) - maths::ln(d - 1.0) + maths::ln(theta) - maths::ln(1.0 - theta)
    }

    /// One centre coordinate on column `col` from its law.
    pub(crate) fn coordinate(&self, col: usize, rng: &mut Rng) -> f64 {
        self.laws[col].draw(rng)
    }

    /// The single covariate of an initial tessellation, with one
    /// uniform: uniform over columns, or the inclusion weights.
    pub(crate) fn initial_dim(&self, rng: &mut Rng) -> usize {
        #[cfg(feature = "experimental")]
        if let Some(weights) = &self.weights {
            return weights.draw_unused(&[], rng).0;
        }
        uniform_index(self.p, rng)
    }

    /// Whether `m` may be proposed from dims `dims` as far as the
    /// inclusion prior is concerned.
    fn inclusion_permits(&self, m: Move, dims: &[usize]) -> bool {
        #[cfg(feature = "experimental")]
        if let Some(weights) = &self.proposal_weights {
            if matches!(m, Move::AddDimension | Move::Swap) {
                return weights.has_positive_unused(dims);
            }
        }
        let _ = (m, dims);
        true
    }
}

impl Move {
    /// Whether the move can be proposed from `t`.
    fn valid(self, t: &Tessellation, prior: &Prior) -> bool {
        match self {
            Move::AddCentre | Move::Change => true,
            Move::RemoveCentre => t.n_cells() >= 2,
            Move::AddDimension | Move::Swap => {
                t.n_dims() < prior.p && prior.inclusion_permits(self, &t.dims)
            }
            Move::RemoveDimension => t.n_dims() >= 2,
        }
    }

    /// The move whose weight an invalid move folds into (Appendix B).
    fn fold_target(self) -> Option<Move> {
        match self {
            Move::RemoveCentre => Some(Move::AddCentre),
            Move::RemoveDimension => Some(Move::AddDimension),
            Move::AddDimension => Some(Move::RemoveDimension),
            Move::Swap => Some(Move::Change),
            Move::AddCentre | Move::Change => None,
        }
    }

    fn reverse(self) -> Move {
        match self {
            Move::AddCentre => Move::RemoveCentre,
            Move::RemoveCentre => Move::AddCentre,
            Move::AddDimension => Move::RemoveDimension,
            Move::RemoveDimension => Move::AddDimension,
            Move::Change => Move::Change,
            Move::Swap => Move::Swap,
        }
    }

    fn index(self) -> usize {
        MOVES.iter().position(|&m| m == self).expect("listed")
    }
}

/// Selection probabilities in state `t`: invalid moves fold their weight
/// into their partner, then the valid mass is normalised.
pub(crate) fn selection_probs(t: &Tessellation, prior: &Prior) -> [f64; 6] {
    // AddDimension and Swap share the same "some unused column has positive
    // weight" check, so compute it once instead of twice.
    let dimension_move_ok = t.n_dims() < prior.p && prior.inclusion_permits(Move::AddDimension, &t.dims);
    let valid = MOVES.map(|m| match m {
        Move::AddDimension | Move::Swap => dimension_move_ok,
        _ => m.valid(t, prior),
    });
    let mut probs = [0.0_f64; 6];
    for (i, &m) in MOVES.iter().enumerate() {
        if valid[i] {
            probs[i] += WEIGHTS[i];
        } else if let Some(target) = m.fold_target() {
            let j = target.index();
            if valid[j] {
                probs[j] += WEIGHTS[i];
            }
        }
    }
    let total: f64 = probs.iter().sum();
    for q in &mut probs {
        *q /= total;
    }
    probs
}

/// Draw a move for state `t` with one uniform (ascending cumulative walk).
/// AddCentre and Change are always valid, so a move always exists.
pub(crate) fn select(t: &Tessellation, prior: &Prior, rng: &mut Rng) -> Move {
    let probs = selection_probs(t, prior);
    let target = uniform(rng);
    let mut cumulative = 0.0;
    for (i, &q) in probs.iter().enumerate() {
        cumulative += q;
        if target < cumulative {
            return MOVES[i];
        }
    }
    MOVES[probs
        .iter()
        .rposition(|&q| q > 0.0)
        .expect("some move is valid")]
}

/// The change a proposed tessellation makes to the assignment and
/// ln[structure prior ratio x within-move proposal ratio], excluding the
/// weighted subset prior and the selection ratio. Cell means in the proposal
/// are placeholders; the sampler redraws every mean after accept or reject.
pub(crate) struct Proposal {
    pub delta: Delta,
    pub log_structure_ratio: f64,
}

/// Propose `m` from `t` into `out`, which keeps its capacity across
/// calls. Draw order: the move's picks, then coordinates in centre then
/// dimension order.
pub(crate) fn propose(
    m: Move,
    t: &Tessellation,
    prior: &Prior,
    rng: &mut Rng,
    out: &mut Tessellation,
) -> Proposal {
    let (b, d) = (t.n_cells(), t.n_dims());
    out.clone_from(t);
    out.betas.clear();
    match m {
        // The new centre's coordinates are drawn from their prior, which
        // cancels the proposal density; the reverse uniform pick 1 / (b + 1)
        // cancels against the b + 1 orderings of the enlarged centre set.
        // What remains is the cell-count prior ratio.
        Move::AddCentre => {
            out.centres
                .extend(t.dims.iter().map(|&dim| prior.coordinate(dim, rng)));
            out.mus.push(0.0);
            Proposal {
                delta: Delta::CentreAdded,
                log_structure_ratio: prior.log_cell_count_ratio(b + 1),
            }
        }
        Move::RemoveCentre => {
            let removed = uniform_index(b, rng);
            out.centres.drain(removed * d..(removed + 1) * d);
            out.mus.remove(removed);
            Proposal {
                delta: Delta::CentreRemoved(removed),
                log_structure_ratio: -prior.log_cell_count_ratio(b),
            }
        }
        // The incoming covariate is picked uniformly among the p - d unused
        // ones and each centre gets one coordinate from the prior.
        Move::AddDimension => {
            #[cfg(feature = "experimental")]
            let (incoming, log_structure_ratio) = match &prior.weights {
                Some(weights) => {
                    let proposal_weights = prior.proposal_weights.as_ref().unwrap_or(weights);
                    let (incoming, s_unused_old) = proposal_weights.draw_unused(&t.dims, rng);
                    let s_new = f64::max(proposal_weights.w[incoming], 1e-16);
                    // Proposal ratio: reverse is uniform removal 1/(d+1),
                    // forward is weighted pick s_new / S_unused_old
                    let log_q_ratio = -maths::ln((d + 1) as f64)
                        + maths::ln(s_unused_old)
                        - maths::ln(s_new);
                    (incoming, prior.log_dim_count_ratio(d + 1) + log_q_ratio)
                }
                None => (
                    unused_column(&t.dims, prior.p, uniform_index(prior.p - d, rng)),
                    prior.log_dim_count_ratio(d + 1),
                ),
            };
            #[cfg(not(feature = "experimental"))]
            let (incoming, log_structure_ratio) = (
                unused_column(&t.dims, prior.p, uniform_index(prior.p - d, rng)),
                prior.log_dim_count_ratio(d + 1),
            );
            out.dims.push(incoming);
            out.centres.clear();
            for k in 0..b {
                out.centres.extend_from_slice(t.centre(k));
                out.centres.push(prior.coordinate(incoming, rng));
            }
            Proposal {
                delta: Delta::Full,
                log_structure_ratio,
            }
        }
        Move::RemoveDimension => {
            let removed = uniform_index(d, rng);
            let removed_var = t.dims[removed];
            out.dims.remove(removed);
            out.centres.clear();
            for k in 0..b {
                for (j, &c) in t.centre(k).iter().enumerate() {
                    if j != removed {
                        out.centres.push(c);
                    }
                }
            }
            Proposal {
                delta: Delta::Full,
                log_structure_ratio: {
                    #[cfg(feature = "experimental")]
                    match &prior.weights {
                        Some(weights) => {
                            let proposal_weights = prior.proposal_weights.as_ref().unwrap_or(weights);
                            let s_old = f64::max(proposal_weights.w[removed_var], 1e-16);
                            let s_unused_new = proposal_weights.unused_weight(&out.dims);
                            // Proposal ratio: forward is uniform removal 1/d,
                            // reverse is weighted pick s_old / S_unused_new
                            let log_q_ratio = maths::ln(s_old)
                                - maths::ln(s_unused_new)
                                + maths::ln(d as f64);
                            -prior.log_dim_count_ratio(d) + log_q_ratio
                        }
                        None => -prior.log_dim_count_ratio(d),
                    }
                    #[cfg(not(feature = "experimental"))]
                    -prior.log_dim_count_ratio(d)
                },
            }
        }
        // Counts unchanged; the pick is 1 / b both ways and the old and new
        // coordinate priors cancel the proposal densities.
        Move::Change => {
            let cell = uniform_index(b, rng);
            for (c, &dim) in out.centres[cell * d..(cell + 1) * d]
                .iter_mut()
                .zip(&t.dims)
            {
                *c = prior.coordinate(dim, rng);
            }
            Proposal {
                delta: Delta::CentreMoved(cell),
                log_structure_ratio: 0.0,
            }
        }
        // Outgoing pick 1 / d, incoming pick 1 / (p - d), mirrored by the
        // reverse; the subset prior is uniform.
        Move::Swap => {
            let removed = uniform_index(d, rng);
            #[cfg(feature = "experimental")]
            let (incoming, log_structure_ratio) = match &prior.weights {
                Some(weights) => {
                    let proposal_weights = prior.proposal_weights.as_ref().unwrap_or(weights);
                    let (incoming, s_unused_old) = proposal_weights.draw_unused(&t.dims, rng);
                    let old_var = t.dims[removed];
                    let s_old = f64::max(proposal_weights.w[old_var], 1e-16);
                    let s_new = f64::max(proposal_weights.w[incoming], 1e-16);
                    // S_unused_new = S_unused_old - s_new + s_old
                    let s_unused_new = s_unused_old - s_new + s_old;
                    // Proposal ratio: forward picks incoming with s_new/S_unused_old,
                    // reverse picks old_var with s_old/S_unused_new
                    let log_q_ratio = maths::ln(s_old)
                        - maths::ln(s_unused_new)
                        - maths::ln(s_new)
                        + maths::ln(s_unused_old);
                    (incoming, log_q_ratio)
                }
                None => (
                    unused_column(&t.dims, prior.p, uniform_index(prior.p - d, rng)),
                    0.0,
                ),
            };
            #[cfg(not(feature = "experimental"))]
            let (incoming, log_structure_ratio) = (
                unused_column(&t.dims, prior.p, uniform_index(prior.p - d, rng)),
                0.0,
            );
            out.dims[removed] = incoming;
            for k in 0..b {
                out.centres[k * d + removed] = prior.coordinate(incoming, rng);
            }
            Proposal {
                delta: Delta::Full,
                log_structure_ratio,
            }
        }
    }
}

/// The `index`-th column in ascending order among those not in `dims`.
fn unused_column(dims: &[usize], p: usize, index: usize) -> usize {
    (0..p)
        .filter(|c| !dims.contains(c))
        .nth(index)
        .expect("an unused column")
}

/// [`propose`] into a fresh tessellation.
#[cfg(test)]
pub(crate) fn propose_owned(
    m: Move,
    t: &Tessellation,
    prior: &Prior,
    rng: &mut Rng,
) -> (Tessellation, Proposal) {
    let mut out = Tessellation::empty();
    let proposal = propose(m, t, prior, rng, &mut out);
    (out, proposal)
}

/// The weighted subset-prior log ratio for a structural proposal.
/// `approximate` selects the cheap delayed-acceptance screen; the accepted
/// proposal must still be corrected with the exact ratio.
#[cfg(feature = "experimental")]
pub(crate) fn log_subset_prior_ratio(
    current: &Tessellation,
    proposed: &Tessellation,
    prior: &Prior,
    approximate: bool,
) -> f64 {
    let Some(weights) = &prior.weights else {
        return 0.0;
    };
    if current.dims() == proposed.dims() {
        return 0.0;
    }
    if approximate {
        crate::aug::log_unordered_subset_prior_approx(proposed.dims(), &weights.w)
            - crate::aug::log_unordered_subset_prior_approx(current.dims(), &weights.w)
    } else {
        crate::aug::log_unordered_subset_prior_ratio(
            current.dims(),
            proposed.dims(),
            &weights.w,
        )
    }
}

/// ln q(reverse | proposed) - ln q(move | current).
pub(crate) fn log_selection_ratio(
    m: Move,
    current: &Tessellation,
    proposed: &Tessellation,
    prior: &Prior,
) -> f64 {
    let forward = selection_probs(current, prior)[m.index()];
    let reverse = selection_probs(proposed, prior)[m.reverse().index()];
    maths::ln(reverse) - maths::ln(forward)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::chain_rng;

    fn prior(p: usize) -> Prior {
        Prior::euclidean(p, 3.0_f64.min(p as f64), 5.0, 0.8)
    }

    #[cfg(feature = "experimental")]
    mod weighted {
        use super::*;

        #[test]
        fn equal_weights_are_the_uniform_prior() {
            assert!(InclusionWeights::new(&[0.5, 0.5, 0.5]).is_none());
            assert!(InclusionWeights::new(&[0.5, 0.5, 0.4]).is_some());
        }

        #[test]
        fn the_symmetric_polynomials_are_computed() {
            let w = InclusionWeights::new(&[1.0, 2.0, 3.0]).unwrap();
            assert_eq!(w.e, vec![1.0, 6.0, 11.0, 6.0]);
        }

        #[test]
        fn draws_avoid_zero_weights_and_active_columns() {
            let w = InclusionWeights::new(&[1.0, 0.0, 2.0]).unwrap();
            let mut rng = chain_rng(5);
            for _ in 0..200 {
                assert_ne!(w.draw_unused(&[], &mut rng).0, 1);
                assert_eq!(w.draw_unused(&[2], &mut rng).0, 0);
            }
            assert!(w.has_positive_unused(&[0]));
            assert!(!w.has_positive_unused(&[0, 2]));
            assert_eq!(w.unused_weight(&[1]), 3.0);
        }

        #[test]
        fn a_zero_weight_blocks_add_and_swap() {
            let mut prior = prior(2);
            prior.weights = InclusionWeights::new(&[1.0, 0.0]);
            let t = tess(1, vec![0]);
            let probs = selection_probs(&t, &prior);
            assert_eq!(probs[Move::AddDimension.index()], 0.0);
            assert_eq!(probs[Move::Swap.index()], 0.0);
        }
    }

    fn tess(b: usize, dims: Vec<usize>) -> Tessellation {
        let d = dims.len();
        Tessellation {
            centres: vec![0.0; b * d],
            dims,
            mus: vec![0.0; b],
            betas: Vec::new(),
            tau: None,
        }
    }

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
    }

    #[test]
    fn folded_table_matches_appendix_b() {
        let pr = prior(10);
        // Interior state: the published weights.
        assert_eq!(
            selection_probs(&tess(3, vec![0, 1]), &pr),
            [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]
        );
        // One cell: RemoveCentre folds into AddCentre.
        let one_cell = selection_probs(&tess(1, vec![0, 1]), &pr);
        close(one_cell[0], 0.4);
        close(one_cell[1], 0.0);
        // One dimension: RemoveDimension folds into AddDimension.
        let one_dim = selection_probs(&tess(3, vec![0]), &pr);
        close(one_dim[2], 0.4);
        close(one_dim[3], 0.0);
        // All dimensions: AddDimension folds into RemoveDimension and Swap
        // into Change.
        let full = selection_probs(&tess(3, (0..10).collect()), &pr);
        close(full[2], 0.0);
        close(full[3], 0.4);
        close(full[4], 0.2);
        close(full[5], 0.0);
    }

    #[test]
    fn selection_ratio_gives_the_boundary_corrections() {
        let pr = prior(10);
        let ln2 = std::f64::consts::LN_2;
        // AddCentre from one cell to two: ln(0.2 / 0.4).
        close(
            log_selection_ratio(
                Move::AddCentre,
                &tess(1, vec![0, 1]),
                &tess(2, vec![0, 1]),
                &pr,
            ),
            -ln2,
        );
        // RemoveCentre from two cells to one: ln(0.4 / 0.2).
        close(
            log_selection_ratio(
                Move::RemoveCentre,
                &tess(2, vec![0, 1]),
                &tess(1, vec![0, 1]),
                &pr,
            ),
            ln2,
        );
        // AddDimension from one dimension to two: -ln 2; to the last
        // dimension: +ln 2.
        close(
            log_selection_ratio(
                Move::AddDimension,
                &tess(2, vec![0]),
                &tess(2, vec![0, 1]),
                &pr,
            ),
            -ln2,
        );
        let all: Vec<usize> = (0..10).collect();
        close(
            log_selection_ratio(
                Move::AddDimension,
                &tess(2, all[..9].to_vec()),
                &tess(2, all.clone()),
                &pr,
            ),
            ln2,
        );
        // Change and Swap in the interior: 0.
        close(
            log_selection_ratio(
                Move::Change,
                &tess(2, vec![0, 1]),
                &tess(2, vec![0, 1]),
                &pr,
            ),
            0.0,
        );
    }

    #[test]
    fn count_prior_ratios_match_the_closed_forms() {
        let pr = prior(10);
        close(pr.log_cell_count_ratio(4), (5.0_f64 / 3.0).ln());
        // d - 1 ~ Binomial(9, 0.3): P(d = 3) / P(d = 2) = (9 - 2 + 1) / 2 * 0.3 / 0.7.
        close(pr.log_dim_count_ratio(3), (8.0 / 2.0 * 0.3 / 0.7_f64).ln());
        // At omega = p the ratio is +infinity: the dimension count saturates.
        let saturated = Prior::euclidean(2, 2.0, pr.lambda_c, 0.8);
        assert_eq!(saturated.log_dim_count_ratio(2), f64::INFINITY);
    }

    #[test]
    fn proposals_have_the_declared_shape() {
        let pr = prior(5);
        let mut rng = chain_rng(3);
        let t = tess(3, vec![1, 4]);
        for _ in 0..50 {
            let m = select(&t, &pr, &mut rng);
            let (new, _) = propose_owned(m, &t, &pr, &mut rng);
            assert_eq!(new.centres.len(), new.n_cells() * new.n_dims());
            match m {
                Move::AddCentre => assert_eq!(new.n_cells(), 4),
                Move::RemoveCentre => assert_eq!(new.n_cells(), 2),
                Move::AddDimension => assert_eq!(new.n_dims(), 3),
                Move::RemoveDimension => assert_eq!(new.n_dims(), 1),
                Move::Change => assert_eq!((new.n_cells(), new.n_dims()), (3, 2)),
                Move::Swap => {
                    assert_eq!(new.n_dims(), 2);
                    assert_ne!(new.dims, t.dims);
                }
            }
            let mut dims = new.dims.clone();
            dims.sort_unstable();
            dims.dedup();
            assert_eq!(dims.len(), new.n_dims());
            assert!(dims.iter().all(|&c| c < 5));
        }
    }

    #[test]
    fn add_and_remove_structure_ratios_are_negatives() {
        let pr = prior(6);
        let mut rng = chain_rng(9);
        let t = tess(3, vec![0, 2]);
        let (added, add) = propose_owned(Move::AddCentre, &t, &pr, &mut rng);
        let (_, remove) = propose_owned(Move::RemoveCentre, &added, &pr, &mut rng);
        close(add.log_structure_ratio, -remove.log_structure_ratio);
        let (added, add) = propose_owned(Move::AddDimension, &t, &pr, &mut rng);
        let (_, remove) = propose_owned(Move::RemoveDimension, &added, &pr, &mut rng);
        close(add.log_structure_ratio, -remove.log_structure_ratio);
    }
}
