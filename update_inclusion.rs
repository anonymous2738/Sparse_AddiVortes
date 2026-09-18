    #[cfg(feature = "experimental")]
    fn update_inclusion(&mut self) {
        let Some(dart) = &mut self.dart else {
            return;
        };
        let p = dart.s.len();
        let mut counts = vec![0.0_f64; p];
        let mut dim_lists: Vec<Vec<usize>> = Vec::new();

        let mut tally = |tessellations: &[crate::tessellation::Tessellation]| {
            for t in tessellations {
                let dims = t.dims().to_vec();
                for &dim in &dims {
                    counts[dim] += 1.0;
                }
                dim_lists.push(dims);
            }
        };
        tally(self.mean.tessellations());
        if let Some(variance) = &self.variance {
            tally(variance.tessellations());
        }

        if self.iter > dart.dirichlet_warmup {
            for dims in dim_lists {
                let d_len = dims.len();
                if d_len == 0 { continue; }
                
                let latent_order = crate::aug::sample_order_given_unordered_set(&dims, &dart.s, &mut self.rng);
                
                for k in 1..d_len {
                    let mut sum_inel = 0.0;
                    for e in 0..k {
                        sum_inel += dart.s[latent_order[e]];
                    }
                    let prob_success = f64::max(1e-5, f64::min(1.0 - 1e-10, 1.0 - sum_inel));
                    
                    let u = crate::rng::uniform(&mut self.rng);
                    let raw_failures = crate::maths::ln(u) / crate::maths::ln(1.0 - prob_success);
                    let failures = (raw_failures.floor() as u64).min(1000);
                    
                    if failures > 0 {
                        if k == 1 {
                            counts[latent_order[0]] += failures as f64;
                        } else {
                            let mut probs = vec![0.0; k];
                            for e in 0..k {
                                probs[e] = dart.s[latent_order[e]];
                            }
                            let alloc = crate::aug::rmultinom(failures, &probs, &mut self.rng);
                            for e in 0..k {
                                counts[latent_order[e]] += alloc[e] as f64;
                            }
                        }
                    }
                }
            }
        }

        let total_counts: f64 = counts.iter().sum();
        let log_density: Vec<f64> = dart
            .grid_theta
            .iter()
            .zip(&dart.log_grid_prior)
            .map(|(&theta, &log_prior)| {
                let mut log_lik = crate::maths::lgamma(theta) - crate::maths::lgamma(theta + total_counts);
                let c = theta / p as f64;
                for &cnt in &counts {
                    if cnt > 0.0 {
                        log_lik += crate::maths::lgamma(c + cnt) - crate::maths::lgamma(c);
                    }
                }
                log_prior + log_lik
            })
            .collect();
        dart.theta = dart.grid_theta[crate::rng::draw_discrete(&log_density, &mut self.rng)];

        let mut shapes: Vec<f64> = Vec::with_capacity(p);
        let base_prior_shape = dart.theta / p as f64;
        let decay = dart.momentum_decay;
        for (i, &m) in counts.iter().enumerate() {
            let base_shape = base_prior_shape + m;
            let safe_momentum = f64::max(dart.momentum[i], -0.95 * base_shape);
            shapes.push(f64::max(base_shape + safe_momentum, 1e-10));
            dart.momentum[i] *= decay;
        }
        let proposed = draw_dirichlet(&shapes, &mut self.rng);
        
        dart.s = proposed;
        let weights = crate::moves::InclusionWeights::sampled(dart.s.clone());
        self.mean.set_inclusion_weights(weights.clone());
        if let Some(variance) = &mut self.variance {
            variance.set_inclusion_weights(weights);
        }
    }
