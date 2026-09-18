fn main() {
    let p = 50;
    let mut counts = vec![0.0_f64; p];
    counts[0] = 300.0; counts[1] = 200.0; counts[2] = 200.0; counts[3] = 200.0; counts[4] = 100.0;
    for i in 5..p { counts[i] = 20.0; } // Some uniform counts on noise!
    
    let total_counts: f64 = counts.iter().sum();
    
    fn lgamma(x: f64) -> f64 {
        let pi = std::f64::consts::PI;
        x.ln() * (x - 0.5) - x + 0.5 * (2.0 * pi).ln()
    }
    
    let thetas = [0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0];
    for &theta in &thetas {
        let mut log_lik = lgamma(theta) - lgamma(theta + total_counts);
        let c = theta / p as f64;
        for &cnt in &counts {
            if cnt > 0.0 {
                log_lik += lgamma(c + cnt) - lgamma(c);
            }
        }
        let log_prior = -0.5 * theta.ln() - theta; // a_alpha=0.5, b_alpha=1.0
        println!("theta = {:.2}, log_lik = {:.2}, log_prior = {:.2}, total = {:.2}", theta, log_lik, log_prior, log_lik + log_prior);
    }
}
