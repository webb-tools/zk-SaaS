use ark_poly::{domain::DomainCoeff, EvaluationDomain, Radix2EvaluationDomain};

use ark_ff::FftField;
use ark_std::{rand::Rng, UniformRand};

use crate::utils::lagrange_interpolate;

/// Packed Secret Sharing Parameters
///
/// Configures the parameters for packed secret sharing. It assumes that the number of parties is `4l`,
/// the corrupting threshold is `l`, and checks that the number of parties (n) equals to `2(t + l)`.
/// Reconstruction of multiplied shares is not possible if >1 party drops out in this configuration
/// Some possible configurations `(t, l, n) - #dropouts tolerated = n - (2(t+l-1) + 1)`:
/// 1. (1, 2, 8) - 3 (ROBUST)
/// 2. (1, 3, 8) - 1 (FAST)
/// 3. (2, 2, 8) - 1 (PRIVATE) [currently implemented]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PackedSharingParams<F>
where
    F: FftField,
{
    /// Corrupting threshold
    pub t: usize,
    /// Packing factor
    pub l: usize,
    /// Number of parties
    pub n: usize,
    /// Share domain
    pub share: Radix2EvaluationDomain<F>,
    /// Secrets domain
    pub secret: Radix2EvaluationDomain<F>,
    /// Secrets2 domain
    pub secret2: Radix2EvaluationDomain<F>,
}

impl<F: FftField> PackedSharingParams<F> {
    /// Creates a new instance of PackedSharingParams with the given packing factor
    #[allow(unused)]
    pub fn new(l: usize) -> Self {
        let n = l * 4;
        let t = l;
        debug_assert_eq!(n, 2 * (t + l));

        let share = Radix2EvaluationDomain::<F>::new(n).unwrap();
        let secret = Radix2EvaluationDomain::<F>::new(l + t)
            .unwrap()
            .get_coset(F::GENERATOR)
            .unwrap();
        let secret2 = Radix2EvaluationDomain::<F>::new(2 * (l + t))
            .unwrap()
            .get_coset(F::GENERATOR)
            .unwrap();

        debug_assert_eq!(share.size(), n);
        debug_assert_eq!(secret.size(), l + t);
        debug_assert_eq!(secret2.size(), 2 * (l + t));

        PackedSharingParams {
            t,
            l,
            n,
            share,
            secret,
            secret2,
        }
    }

    /// Deterministically packs secrets into shares
    #[allow(unused)]
    pub fn det_pack<T: DomainCoeff<F> + UniformRand>(
        &self,
        secrets: Vec<T>,
    ) -> Vec<T> {
        debug_assert!(secrets.len() == self.l, "Secrets length mismatch");

        let mut result = secrets;

        // Resize the secrets with t zeros
        result.append(&mut vec![T::zero(); self.t]);

        // interpolating on secrets domain
        self.secret.ifft_in_place(&mut result);

        // evaluate on share domain
        self.share.fft_in_place(&mut result);

        result
    }

    /// Packs secrets into shares
    #[allow(unused)]
    pub fn pack<T: DomainCoeff<F> + UniformRand>(
        &self,
        secrets: Vec<T>,
        rng: &mut impl Rng,
    ) -> Vec<T> {
        debug_assert!(secrets.len() == self.l, "Secrets length mismatch");

        let mut result = secrets;

        // Resize the secrets with t random points
        let rand_points = (0..self.t).map(|_| T::rand(rng)).collect::<Vec<T>>();
        result.extend_from_slice(&rand_points);

        // interpolating on secrets domain
        self.secret.ifft_in_place(&mut result);

        #[cfg(debug_assertions)]
        {
            // assert that all but first l+t elements are zero
            for item in result.iter().skip(self.l + self.t) {
                debug_assert!(
                    item.is_zero(),
                    "Pack found non zero coefficient: {:?}",
                    result
                );
            }
        }

        // evaluate on share domain
        self.share.fft_in_place(&mut result);

        result
    }

    /// Unpacks shares of degree t+l into secrets
    #[allow(unused)]
    pub fn unpack<T: DomainCoeff<F>>(&self, shares: Vec<T>) -> Vec<T> {
        let mut result = shares;

        // interpolating on share domain
        self.share.ifft_in_place(&mut result);

        // evaluate on secrets domain
        self.secret.fft_in_place(&mut result);

        // truncate to remove the randomness
        result.truncate(self.l);

        result
    }

    /// Unpacks shares of degree 2(t+l) into secrets
    #[allow(unused)]
    pub fn unpack2<T: DomainCoeff<F>>(&self, shares: Vec<T>) -> Vec<T> {
        let mut result = shares;

        // interpolating on share domain
        self.share.ifft_in_place(&mut result);

        // assert that all but first 2(t+l)+1 elements are zero
        #[cfg(debug_assertions)]
        {
            for item in result.iter().skip(2 * (self.l + self.t) - 1) {
                debug_assert!(
                    item.is_zero(),
                    "Unpack2 found non zero coefficient: {:?}",
                    result
                );
            }
        }

        // evaluate on secrets domain
        self.secret2.fft_in_place(&mut result);

        // drop alternate elements from shares array and only iterate till 2l as the rest of it is randomness
        result = result[0..2 * self.l].iter().step_by(2).copied().collect();

        result
    }

    /// Runs lagrange interpolation to unpack the secrets. Can be used when some shares are missing.
    /// todo: can be optimized by compting secrets directly instead of first interpolating the polynomial
    pub fn lagrange_unpack<T: DomainCoeff<F>>(
        &self,
        shares: &[T],
        parties: &[u32],
    ) -> Vec<T> {
        // first generate lagrange coefficients for the parties specified
        // these are the lagrange polynomials corresponding to the share domain, evaluated at the secret domain
        // code ported from https://github.com/facebook/winterfell/blob/a450b818f7ec70e7d40628c789845a93d6e0c030/math/src/polynom/mod.rs#L626
        // Note: ordering of polynomial coefficients is largest power -> smaller power

        debug_assert!(
            shares.len() == parties.len(),
            "Shares and parties length mismatch"
        );

        debug_assert!(
            parties.len() > 2 * (self.t + self.l - 1),
            "Not enough shares to reconstruct"
        );

        let mut xs = Vec::new();
        let share_elements = self.share.elements().collect::<Vec<F>>();
        for i in 0..parties.len() {
            xs.push(share_elements[parties[i] as usize]);
        }

        let mut result = lagrange_interpolate(&xs, shares);

        // evaluate on secrets domain
        self.secret2.fft_in_place(&mut result);

        // drop alternate elements from shares array and only iterate till 2l as the rest of it is randomness
        result = result[0..2 * self.l].iter().step_by(2).copied().collect();

        result
    }
}

// Tests
#[cfg(test)]
mod tests {
    use crate::utils::eval;

    use super::*;
    use ark_bls12_377::Fr as F;
    use ark_std::UniformRand;
    use PackedSharingParams;

    const L: usize = 2;
    const N: usize = L * 4;
    const T: usize = L;

    #[test]
    fn test_initialize() {
        let pp = PackedSharingParams::<F>::new(L);
        assert_eq!(pp.t, L);
        assert_eq!(pp.l, L);
        assert_eq!(pp.n, N);
        assert_eq!(pp.share.size(), N);
        assert_eq!(pp.secret.size(), L + T);
        assert_eq!(pp.secret2.size(), 2 * (L + T));
    }

    #[test]
    fn test_packing() {
        let pp = PackedSharingParams::<F>::new(L);

        let rng = &mut ark_std::test_rng();
        let secrets: [F; L] = UniformRand::rand(rng);
        let secrets = secrets.to_vec();

        let expected = secrets.clone();

        let shares = pp.pack(secrets, rng);
        let secrets = pp.unpack(shares.clone());

        // using only a subset of shares here
        let lagrange_secrets = pp.lagrange_unpack(
            &shares[0..pp.n - pp.t + 1],
            &(0..(pp.n - pp.t + 1) as u32)
                .collect::<Vec<u32>>()
                .as_slice(),
        );

        assert_eq!(expected, secrets);
        assert_eq!(expected, lagrange_secrets);
    }

    #[test]
    fn test_det_packing() {
        let pp = PackedSharingParams::<F>::new(L);

        let rng = &mut ark_std::test_rng();
        let secrets: [F; L] = UniformRand::rand(rng);
        let secrets = secrets.to_vec();

        let expected = secrets.clone();

        let shares = pp.det_pack(secrets);
        let secrets = pp.unpack(shares);

        assert_eq!(expected, secrets);
    }

    #[test]
    fn test_multiplication() {
        let pp = PackedSharingParams::<F>::new(L);

        let rng = &mut ark_std::test_rng();
        let secrets: [F; L] = UniformRand::rand(rng);
        let secrets = secrets.to_vec();
        let expected: Vec<F> = secrets.iter().map(|x| (*x) * (*x)).collect();

        let shares = pp.pack(secrets, rng);
        let mul_shares: Vec<F> = shares.iter().map(|x| (*x) * (*x)).collect();
        let mul_secrets = pp.unpack2(mul_shares.clone());

        // can tolerate 1 party dropping out
        let lagrange_secrets = pp.lagrange_unpack(
            &mul_shares[0..pp.n - 1].to_vec(),
            &(0..(pp.n - 1) as u32).collect::<Vec<u32>>().as_slice(),
        );

        assert_eq!(expected, mul_secrets);
        assert_eq!(expected, lagrange_secrets);
    }

    #[test]
    fn test_eval_interpolate() {
        let degree = 32u32;
        let rng = &mut ark_std::test_rng();
        let p = (0..degree)
            .map(|_| F::rand(rng))
            .collect::<Vec<F>>();
        let xs = (1..=2*degree).map(|x| F::from(x)).collect::<Vec<F>>();
        let ys = xs.iter().map(|x| eval(&p, *x)).collect::<Vec<F>>();

        let should_be_p = lagrange_interpolate(&xs, &ys);
        assert_eq!(should_be_p, p);
    }
}
