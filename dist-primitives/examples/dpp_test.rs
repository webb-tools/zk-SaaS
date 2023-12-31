use ark_bls12_377::Fr;
use ark_ff::{FftField, PrimeField};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use dist_primitives::{
    dpp::d_pp,
    utils::pack::{pack_vec, transpose},
};
use mpc_net::ser_net::MpcSerNet;
use mpc_net::{LocalTestNet as Net, MpcNet, MultiplexedStreamID};
use secret_sharing::pss::PackedSharingParams;

pub async fn d_pp_test<F: FftField + PrimeField, Net: MpcNet>(
    pp: &PackedSharingParams<F>,
    dom: &Radix2EvaluationDomain<F>,
    net: &Net,
) {
    let px: Option<Vec<Vec<F>>>;
    if net.is_king() {
        let mut x: Vec<F> = Vec::new();
        for i in 0..dom.size() {
            x.push(F::from((i + 1) as u64));
        }
        px = Some(transpose(pack_vec(&x, pp)))
    } else {
        px = None;
    }

    let px_share = net
        .client_receive_or_king_send_serialized(px, MultiplexedStreamID::One)
        .await
        .unwrap();

    let pp_px_share = d_pp(
        px_share.clone(),
        px_share.clone(),
        pp,
        net,
        MultiplexedStreamID::One,
    )
    .await
    .unwrap();

    // Send to king who reconstructs and checks the answer
    net.client_send_or_king_receive_serialized(
        &pp_px_share,
        MultiplexedStreamID::One,
        pp.t,
    )
    .await
    .unwrap()
    .map(|rs| {
        let pp_px_shares = transpose(rs.shares);

        let pp_px: Vec<F> = pp_px_shares
            .into_iter()
            .flat_map(|x| pp.unpack(x))
            .collect();

        debug_assert_eq!(vec![F::one(); dom.size()], pp_px);
    });
}

#[tokio::main]
async fn main() {
    env_logger::builder().format_timestamp(None).init();

    let network = Net::new_local_testnet(8).await.unwrap();
    network
        .simulate_network_round((), |net, _| async move {
            let pp = PackedSharingParams::<Fr>::new(2);
            let cd = Radix2EvaluationDomain::<Fr>::new(1 << 5).unwrap();
            d_pp_test::<Fr, _>(&pp, &cd, &net).await;
        })
        .await;
}
