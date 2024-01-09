use ark_ec::CurveGroup;
use ark_ff::UniformRand;
use mpc_net::ser_net::MpcSerNet;
use mpc_net::{MpcNetError, MultiplexedStreamID};
use secret_sharing::pss::PackedSharingParams;

/// Masks used in dmsm
/// Note that this only contains one share of the mask
#[derive(Clone)]
pub struct MsmMask<G: CurveGroup> {
    pub in_mask: G,
    pub out_mask: G,
}

impl<G: CurveGroup> MsmMask<G> {
    pub fn new(in_mask: G, out_mask: G) -> Self {
        Self { in_mask, out_mask }
    }

    /// Samples a random MsmMask and returns the shares of n parties
    pub fn sample(
        pp: &PackedSharingParams<G::ScalarField>,
        rng: &mut impl rand::Rng,
    ) -> Vec<Self> {
        let gen = G::generator();
        let mut mask_values = Vec::new();
        for _ in 0..pp.l {
            mask_values.push(G::ScalarField::rand(rng));
        }

        let mask_values: Vec<G> = mask_values.iter().map(|x| gen * x).collect();
        let out_mask_value = -(mask_values.iter().sum::<G>());

        let in_mask_shares = pp.pack(mask_values, rng);

        // TODO: use regular secret sharing here. Currently using packed secret sharing with repeated secrets.
        // doesn't affect correctness/privacy but would give a little bit of performance
        let out_mask_shares = pp.pack(vec![out_mask_value; pp.l], rng);

        in_mask_shares
            .into_iter()
            .zip(out_mask_shares.iter())
            .map(|(in_mask_share, out_mask_share)| Self::new(in_mask_share, out_mask_share.clone()))
            .collect()
    }
}

pub async fn d_msm<G: CurveGroup, Net: MpcSerNet>(
    bases: &[G::Affine],
    scalars: &[G::ScalarField],
    msm_mask: &MsmMask<G>,
    pp: &PackedSharingParams<G::ScalarField>,
    net: &Net,
    sid: MultiplexedStreamID,
) -> Result<G, MpcNetError> {
    // Using affine is important because we don't want to create an extra vector for converting Projective to Affine.
    // Eventually we do have to convert to Projective but this will be pp.l group elements instead of m()

    // First round of local computation done by parties
    debug_assert_eq!(bases.len(), scalars.len());
    log::debug!("bases: {}, scalars: {}", bases.len(), scalars.len());
    let c_share = G::msm(bases, scalars)?;
    let c_share = c_share + msm_mask.in_mask;
    // Now we do degree reduction -- psstoss
    // Send to king who reduces and sends shamir shares (not packed).
    // Should be randomized. First convert to projective share.
    let n_parties = net.n_parties();
    let king_answer: Option<Vec<G>> = net
        .client_send_or_king_receive_serialized(&c_share, sid, pp.t)
        .await?
        .map(|rs| {
            // TODO: Mask with random values.

            let result = pp.unpack_missing_shares(&rs.shares, &rs.parties);
            let output: G = result.iter().sum();
            vec![output; n_parties]
        });

    let result = net
        .client_receive_or_king_send_serialized(king_answer, sid)
        .await;

    // At the end all parties hold a packed secret sharing of the output
    // Note that the output is just a single group element and it is shared
    // using "repeated" packed secret sharing i.e equivalent to pp.pack(vec![output; pp.l])
    if let Ok(output) = result {
        Ok(output + msm_mask.out_mask)
    } else {
        result
    }
}

#[cfg(test)]
mod tests {
    use ark_ec::bls12::Bls12Config;
    use ark_ec::CurveGroup;
    use ark_ec::Group;
    use ark_ec::VariableBaseMSM;
    use ark_std::UniformRand;
    use ark_std::Zero;
    use secret_sharing::pss::PackedSharingParams;

    use ark_bls12_377::G1Affine;
    use ark_bls12_377::G1Projective as G1P;

    type F = <ark_ec::short_weierstrass::Projective<
        <ark_bls12_377::Config as Bls12Config>::G1Config,
    > as Group>::ScalarField;

    use crate::utils::pack::transpose;

    const L: usize = 2;
    const N: usize = L * 4;
    const M: usize = 1 << 8;

    #[tokio::test]
    async fn pack_unpack_test() {
        let pp = PackedSharingParams::<F>::new(L);
        let rng = &mut ark_std::test_rng();
        let secrets: [G1P; L] = UniformRand::rand(rng);
        let secrets = secrets.to_vec();

        let shares = pp.pack(secrets.clone(), rng);
        let result = pp.unpack(shares);
        assert_eq!(secrets, result);
    }

    #[tokio::test]
    async fn pack_unpack2_test() {
        let pp = PackedSharingParams::<F>::new(L);
        let rng = &mut ark_std::test_rng();

        let gsecrets: [G1P; M] = [G1P::rand(rng); M];
        let gsecrets = gsecrets.to_vec();

        let fsecrets: [F; M] = [F::from(1_u32); M];
        let fsecrets = fsecrets.to_vec();

        ///////////////////////////////////////
        let gsecrets_aff: Vec<G1Affine> =
            gsecrets.iter().map(|s| (*s).into()).collect();
        let expected = G1P::msm(&gsecrets_aff, &fsecrets).unwrap();
        ///////////////////////////////////////
        let gshares: Vec<Vec<G1P>> = gsecrets
            .chunks(L)
            .map(|s| pp.pack(s.to_vec(), rng))
            .collect();

        let fshares: Vec<Vec<F>> = fsecrets
            .chunks(L)
            .map(|s| pp.pack(s.to_vec(), rng))
            .collect();

        let gshares = transpose(gshares);
        let fshares = transpose(fshares);

        let mut result = vec![G1P::zero(); pp.n];

        for i in 0..N {
            let temp_aff: Vec<
                <ark_ec::short_weierstrass::Projective<
                    <ark_bls12_377::Config as Bls12Config>::G1Config,
                > as CurveGroup>::Affine,
            > = gshares[i].iter().map(|s| (*s).into()).collect();
            result[i] = G1P::msm(&temp_aff, &fshares[i]).unwrap();
        }
        let result: G1P = pp.unpack2(result).iter().sum();
        assert_eq!(expected, result);
    }
}
