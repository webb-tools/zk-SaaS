use ark_ff::FftField;
use ark_poly::domain::DomainCoeff;
use ark_std::{cfg_chunks, UniformRand};
use rand::thread_rng;
use secret_sharing::pss::PackedSharingParams;

pub fn pack_vec<F: FftField, T: DomainCoeff<F> + UniformRand>(
    secrets: &Vec<T>,
    pp: &PackedSharingParams<F>,
) -> Vec<Vec<T>> {
    debug_assert_eq!(secrets.len() % pp.l, 0, "Mismatch of size in pack_vec");

    let rng = &mut thread_rng();

    // pack shares
    cfg_chunks!(secrets, pp.l)
        .map(|x| pp.pack(x.to_vec(), rng))
        .collect::<Vec<_>>()
}

pub fn transpose<T: Clone>(matrix: Vec<Vec<T>>) -> Vec<Vec<T>> {
    assert!(!matrix.is_empty());
    let cols = matrix[0].len();
    let rows = matrix.len();

    let mut result: Vec<Vec<T>> = vec![vec![matrix[0][0].clone(); rows]; cols];

    for (c, column) in result.iter_mut().enumerate().take(cols) {
        for (r, row) in matrix.iter().enumerate().take(rows) {
            column[r] = row[c].clone();
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose() {
        let matrix = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];

        let expected = vec![vec![1, 4, 7], vec![2, 5, 8], vec![3, 6, 9]];

        assert_eq!(transpose(matrix), expected);
    }
}
