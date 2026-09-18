fn main() {
    let mut rng = rand::thread_rng();
    use rand::Rng;
    for &prob_success in &[0.01, 0.5, 0.99] {
        let mut sum = 0;
        let n = 100000;
        for _ in 0..n {
            let u: f64 = rng.gen();
            let raw_failures = u.ln() / (1.0 - prob_success).ln();
            let failures = raw_failures.floor() as u64;
            sum += failures;
        }
        println!("prob_success = {}, E[X] = {}", prob_success, sum as f64 / n as f64);
    }
}
