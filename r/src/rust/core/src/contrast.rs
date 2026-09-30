//! Contrast-space utilities for label-symmetric categorical models.

#[cfg(feature = "experimental")]
pub(crate) fn normalized_helmert(k: usize) -> Vec<f64> {
    assert!(k >= 2, "a categorical model needs at least two classes");
    let q = k - 1;
    let mut h = vec![0.0; k * q];
    for column in 0..q {
        let width = (column + 1) as f64;
        let norm = (width * (width + 1.0)).sqrt();
        for row in 0..=column {
            h[row * q + column] = 1.0 / norm;
        }
        h[(column + 1) * q + column] = -width / norm;
    }
    h
}

#[cfg(feature = "experimental")]
pub(crate) fn project(h: &[f64], k: usize, utilities: &[f64]) -> Vec<f64> {
    assert_eq!(utilities.len(), k);
    let q = k - 1;
    (0..q)
        .map(|column| {
            (0..k)
                .map(|row| h[row * q + column] * utilities[row])
                .sum()
        })
        .collect()
}

#[cfg(feature = "experimental")]
pub(crate) fn lift(h: &[f64], k: usize, contrast: &[f64]) -> Vec<f64> {
    assert_eq!(contrast.len(), k - 1);
    let q = k - 1;
    (0..k)
        .map(|row| {
            (0..q)
                .map(|column| h[row * q + column] * contrast[column])
                .sum()
        })
        .collect()
}
