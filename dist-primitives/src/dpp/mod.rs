// Evalauting a distributed version of partial products
// Given x1, x2, .., xn, output x1, x1*x2, x1*x2*x3, .., x1*x2*..*xn

use crate::utils::{
    deg_red::deg_red,
    pack::{pack_vec, transpose},
};
use ark_ff::{FftField, Field, PrimeField};
use mpc_net::ser_net::MpcSerNet;
use mpc_net::{MpcNetError, MultiplexedStreamID};
use secret_sharing::pss::PackedSharingParams;

// Given pre-processed randomness [s], [s^-1]
// Partial products of [num] and [den] are computed
pub async fn d_pp<F: FftField + PrimeField + Field, Net: MpcSerNet>(
    num: Vec<F>,
    den: Vec<F>,
    pp: &PackedSharingParams<F>,
    net: &Net,
    sid: MultiplexedStreamID,
) -> Result<Vec<F>, MpcNetError> {
    // using some dummy randomness
    let s = F::from(1_u32);
    let sinv = s.inverse().unwrap();

    // multiply all entries of px by of s
    let num_rand = num.iter().map(|&x| x * s).collect::<Vec<_>>();
    let mut den_rand = den.iter().map(|&x| x * s).collect::<Vec<_>>();

    let mut numden_rand = num_rand;
    numden_rand.append(&mut den_rand);

    // Along with degree reduction
    // King recovers secrets, computes partial products and repacks
    let received_shares = net
        .client_send_or_king_receive_serialized(&numden_rand, sid, pp.t)
        .await?;

    let king_answer: Option<Vec<Vec<F>>> = received_shares.map(|rs| {
        // nx(m/l) -> (m/l)xn
        debug_assert_eq!(rs.shares.len(), pp.n, "Mismatch of size in d_pp");
        let numden_shares = transpose(rs.shares);

        // Unpack the secrets
        // (m/l)xn -> m
        // iterate over pxss_shares, unpack to get a vector and append all the vectors
        let mut numden: Vec<F> = numden_shares
            .into_iter()
            .flat_map(|x| pp.unpack_missing_shares(&x, &rs.parties))
            .collect();

        for i in 0..numden.len() / 2 {
            let den = numden[i + numden.len() / 2].inverse().unwrap();
            numden[i] *= den;
        }

        numden.truncate(numden.len() / 2);

        // Compute the partial products across pxss
        for i in 1..numden.len() {
            let last = numden[i - 1];
            numden[i] *= last;
        }

        // Pack the secrets
        // m -> (m/l)xn
        // (m/l)xl -> (m/l)xn
        let pp_numden_shares = pack_vec(&numden, pp);
        drop(numden);

        // send shares to parties
        // (m/l)xn -> nx(m/l)
        transpose(pp_numden_shares)
    });

    let mut pp_numden_rand = net
        .client_receive_or_king_send_serialized(king_answer, sid)
        .await?;

    // Finally, remove the ranomness in the partial products
    // multiply all entries of pp_pxss by of s
    // do degree reduction
    // todo: replace in_mask and out_mask
    pp_numden_rand.iter_mut().for_each(|x| *x *= sinv);
    deg_red(pp_numden_rand, vec![F::zero(); pp.l], vec![F::zero(); pp.l], pp, net, sid).await //packed shares of partial products
}
