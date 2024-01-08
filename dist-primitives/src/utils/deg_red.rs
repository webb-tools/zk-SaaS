use ark_ff::FftField;
use ark_poly::domain::DomainCoeff;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::UniformRand;
use mpc_net::ser_net::MpcSerNet;
use mpc_net::{MpcNetError, MultiplexedStreamID};
use secret_sharing::pss::PackedSharingParams;

use super::pack::transpose;

/// Reduces the degree of a poylnomial with the help of king
pub async fn deg_red<
    F: FftField,
    T: DomainCoeff<F> + CanonicalSerialize + CanonicalDeserialize + UniformRand,
    Net: MpcSerNet,
>(
    x_share: Vec<T>,
    in_mask: Vec<T>,
    out_mask: Vec<T>,
    pp: &PackedSharingParams<F>,
    net: &Net,
    sid: MultiplexedStreamID,
) -> Result<Vec<T>, MpcNetError> {
    
    debug_assert_eq!(x_share.len(), in_mask.len());
    debug_assert_eq!(x_share.len(), out_mask.len());

    let x_mask = x_share.into_iter().zip(in_mask.into_iter()).map(|(x, m)| x + m).collect();
    let received_shares = net
        .client_send_or_king_receive_serialized(&x_mask, sid, pp.t)
        .await?;

    let king_answer: Option<Vec<Vec<T>>> = received_shares.map(|rs| {
        let mut x_shares = transpose(rs.shares);

        for x_share in &mut x_shares {
            let xi: Vec<T> = pp.unpack_missing_shares(x_share, &rs.parties);
            *x_share = pp.pack(xi, &mut rand::thread_rng());
        }
        transpose(x_shares)
    });

    let result = net.client_receive_or_king_send_serialized(king_answer, sid)
        .await;

    if let Ok(x_share) = result {
        Ok(x_share
            .into_iter()
            .zip(out_mask.into_iter())
            .map(|(x, m)| x + m)
            .collect()
        )
    } else {
        result
    }
}

#[cfg(test)]
mod tests {
    use ark_bls12_377::Fr as F;
    use ark_std::UniformRand;
    use mpc_net::ser_net::ReceivedShares;
    use mpc_net::MpcNet;
    use mpc_net::{LocalTestNet, MultiplexedStreamID};
    use secret_sharing::pss::PackedSharingParams;

    use crate::utils::{deg_red::deg_red, pack::transpose};
    const L: usize = 4;

    #[tokio::test]
    async fn test_deg_red() {
        let pp = PackedSharingParams::<F>::new(L);
        let rng = &mut ark_std::test_rng();
        let network = LocalTestNet::new_local_testnet(pp.n).await.unwrap();
        let secrets: [F; L] = UniformRand::rand(rng);
        let secrets = secrets.to_vec();
        let expected: Vec<F> = secrets.iter().map(|x| (*x) * (*x)).collect();

        let shares = pp.pack(secrets, rng);
        let mul_shares: Vec<F> = shares.iter().map(|x| (*x) * (*x)).collect();

        let mut mask_values = Vec::new();
        for _ in 0..pp.l {
            mask_values.push(F::rand(rng));
        }
        let in_masks = pp.pack(mask_values.clone(), rng);
        // negate every value of mask_values
        let out_masks = pp.pack(mask_values.into_iter().map(|x| -x).collect(), rng);

        let rs: ReceivedShares<Vec<F>> = network
            .simulate_lossy_network_round(
                (mul_shares, in_masks, out_masks, pp),
                |net, (mul_shares, in_masks, out_masks, pp)| async move {
                    let idx = net.party_id() as usize;
                    let mul_share = mul_shares[idx].clone();
                    deg_red(
                        vec![mul_share],
                        vec![in_masks[idx]],
                        vec![out_masks[idx]],
                        &pp,
                        &net,
                        MultiplexedStreamID::One,
                    )
                    .await
                    .unwrap()
                },
            )
            .await;

        let shares = transpose(rs.shares);
        let computed = if rs.parties.len() == pp.n {
            shares
                .into_iter()
                .flat_map(|x| pp.unpack(x))
                .collect::<Vec<_>>()
        } else {
            println!("Using lagrange unpack");
            shares
                .into_iter()
                .flat_map(|x| pp.lagrange_unpack(&x, &rs.parties))
                .collect::<Vec<_>>()
        };

        assert_eq!(computed, expected);
    }
}
