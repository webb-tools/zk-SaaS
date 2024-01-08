use ark_ec::CurveGroup;
use mpc_net::ser_net::MpcSerNet;
use mpc_net::{MpcNetError, MultiplexedStreamID};
use secret_sharing::pss::PackedSharingParams;

pub async fn d_msm<G: CurveGroup, Net: MpcSerNet>(
    bases: &[G::Affine],
    scalars: &[G::ScalarField],
    in_mask: G,
    out_mask: G,
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
    let c_share = c_share + in_mask;
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

    if let Ok(output) = result {
        Ok(output + out_mask)
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
