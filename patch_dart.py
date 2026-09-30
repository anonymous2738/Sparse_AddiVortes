import re

with open("r/R/params.R", "r") as f:
    text = f.read()
text = text.replace(
    "dirichlet_warmup = 1000) {",
    "dirichlet_warmup = 1000,\n                           momentum_settle = 0, momentum_window = 1) {"
)
text = text.replace(
    "check_whole_number(dirichlet_warmup, min = 0)",
    "check_whole_number(dirichlet_warmup, min = 0)\n  check_whole_number(momentum_settle, min = 0)\n  check_whole_number(momentum_window, min = 1)"
)
text = text.replace(
    "dirichlet_warmup = as.integer(dirichlet_warmup)",
    "dirichlet_warmup = as.integer(dirichlet_warmup),\n    momentum_settle = as.integer(momentum_settle),\n    momentum_window = as.integer(momentum_window)"
)
with open("r/R/params.R", "w") as f:
    f.write(text)

with open("r/src/rust/core/src/ensemble.rs", "r") as f:
    text = f.read()
text = text.replace(
    "if iter > dart.dirichlet_warmup && iter <= dart.burn_in && moves_a_dimension {",
    "if iter > dart.dirichlet_warmup && iter <= dart.burn_in.saturating_sub(dart.momentum_settle) && moves_a_dimension {"
)
with open("r/src/rust/core/src/ensemble.rs", "w") as f:
    f.write(text)

with open("r/src/rust/core/src/sampler.rs", "r") as f:
    text = f.read()
text = text.replace(
    "pub(crate) burn_in: usize,\n    counts: Vec<f64>,",
    "pub(crate) burn_in: usize,\n    pub(crate) momentum_settle: usize,\n    pub(crate) momentum_window: usize,\n    pub(crate) momentum_sum: Vec<f64>,\n    counts: Vec<f64>,"
)
text = text.replace(
    "fn draw(alpha: f64, a_alpha: f64, b_alpha: f64, adapt_boost: f64, adapt_penalty: f64, momentum_decay: f64, kappa: f64, tau: f64, dirichlet_warmup: usize, burn_in: usize, p: usize) -> Self {",
    "fn draw(alpha: f64, a_alpha: f64, b_alpha: f64, adapt_boost: f64, adapt_penalty: f64, momentum_decay: f64, kappa: f64, tau: f64, dirichlet_warmup: usize, momentum_settle: usize, momentum_window: usize, burn_in: usize, p: usize) -> Self {"
)
text = text.replace(
    "dirichlet_warmup,\n            burn_in,\n            counts: vec![0.0; p],",
    "dirichlet_warmup,\n            burn_in,\n            momentum_settle,\n            momentum_window,\n            momentum_sum: vec![0.0; p],\n            counts: vec![0.0; p],"
)

text = text.replace(
    "crate::config::Inclusion::Dart { alpha, a_alpha, b_alpha, adapt_boost, adapt_penalty, momentum_decay, kappa, tau, dirichlet_warmup } => {",
    "crate::config::Inclusion::Dart { alpha, a_alpha, b_alpha, adapt_boost, adapt_penalty, momentum_decay, kappa, tau, dirichlet_warmup, momentum_settle, momentum_window } => {"
)
text = text.replace(
    "Some(Dart::draw(*alpha, *a_alpha, *b_alpha, *adapt_boost, *adapt_penalty, *momentum_decay, *kappa, *tau, *dirichlet_warmup, config.general_params.burn_in, p))",
    "Some(Dart::draw(*alpha, *a_alpha, *b_alpha, *adapt_boost, *adapt_penalty, *momentum_decay, *kappa, *tau, *dirichlet_warmup, *momentum_settle, *momentum_window, config.general_params.burn_in, p))"
)

old_update = """        if self.iter <= dart.burn_in {
            for (momentum, tilt) in dart.momentum.iter_mut().zip(&mut dart.proposal_tilt) {
                *momentum *= dart.momentum_decay;
                *tilt = maths::exp(momentum.clamp(-30.0, 30.0));
            }
        }"""
new_update = """        let t_adapt = dart.burn_in.saturating_sub(dart.momentum_settle);
        if self.iter <= t_adapt {
            let in_window = self.iter > t_adapt.saturating_sub(dart.momentum_window);
            for (i, (momentum, tilt)) in dart.momentum.iter_mut().zip(&mut dart.proposal_tilt).enumerate() {
                *momentum *= dart.momentum_decay;
                *tilt = maths::exp(momentum.clamp(-30.0, 30.0));
                if in_window {
                    dart.momentum_sum[i] += *momentum;
                }
            }
            if self.iter == t_adapt && dart.momentum_window > 0 {
                for (i, (momentum, tilt)) in dart.momentum.iter_mut().zip(&mut dart.proposal_tilt).enumerate() {
                    let m_star = dart.momentum_sum[i] / (dart.momentum_window as f64);
                    *momentum = m_star.clamp(-30.0, 30.0);
                    *tilt = maths::exp(*momentum);
                }
            }
        }"""
text = text.replace(old_update, new_update)

with open("r/src/rust/core/src/sampler.rs", "w") as f:
    f.write(text)
