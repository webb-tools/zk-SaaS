use ark_bls12_377::Fr;
use ark_ff::{FftField, PrimeField};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use ark_std::One;
use dist_primitives::{
    dpp::d_pp,
    utils::{
        deg_red::DegRedMask,
        pack::{pack_vec, transpose},
    },
};
use mpc_net::ser_net::MpcSerNet;
use mpc_net::{LocalTestNet as Net, MpcNet, MultiplexedStreamID};
use secret_sharing::pss::PackedSharingParams;

pub async fn d_pp_test<F: FftField + PrimeField, Net: MpcNet>(
    px_share: &Vec<F>,
    degred_mask: &DegRedMask<F, F>,
    pp: &PackedSharingParams<F>,
    dom: &Radix2EvaluationDomain<F>,
    net: &Net,
) {
    let pp_px_share = d_pp(
        px_share.clone(),
        px_share.clone(),
        &degred_mask,
        // DegRedMask::new(vec![F::from(1u32); dom.size()/pp.l], vec![-F::from(1u32); dom.size()/pp.l]),
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

    let pp = PackedSharingParams::<Fr>::new(2);
    let dom = Radix2EvaluationDomain::<Fr>::new(1 << 5).unwrap();
    let mut x: Vec<Fr> = Vec::new();
    for i in 0..dom.size() {
        x.push(Fr::from((i + 1) as u64));
    }

    let px = transpose(pack_vec(&x, &pp));
    let degred_masks = DegRedMask::<Fr, Fr>::sample(
        &pp,
        Fr::one(),
        dom.size() / pp.l,
        &mut ark_std::test_rng(),
    );

    network
        .simulate_network_round(
            (px, degred_masks, pp, dom),
            |net, (px, degred_masks, pp, dom)| async move {
                let idx = net.party_id() as usize;
                d_pp_test::<Fr, _>(
                    &px[idx],
                    &degred_masks[idx],
                    &pp,
                    &dom,
                    &net,
                )
                .await;
            },
        )
        .await;
}
