use ark_bls12_377::Fr;
use ark_ec::CurveGroup;
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use ark_std::{UniformRand, Zero};
use dist_primitives::dmsm::{d_msm, MsmMask};
use mpc_net::{LocalTestNet as Net, MpcNet, MultiplexedStreamID};
use secret_sharing::pss::PackedSharingParams;

pub async fn d_msm_test<G: CurveGroup, Net: MpcNet>(
    pp: &PackedSharingParams<G::ScalarField>,
    dom: &Radix2EvaluationDomain<G::ScalarField>,
    net: &Net,
) {
    // let m = pp.l*4;
    let mbyl: usize = dom.size() / pp.l;
    println!("m: {}, mbyl: {}", dom.size(), mbyl);

    let rng = &mut ark_std::test_rng();

    let mut y_share: Vec<G::ScalarField> =
        vec![G::ScalarField::zero(); dom.size()];
    let mut x_share: Vec<G> = vec![G::zero(); dom.size()];

    for i in 0..dom.size() {
        y_share[i] = G::ScalarField::rand(rng);
        x_share[i] = G::rand(rng);
    }

    let x_share_aff: Vec<G::Affine> =
        x_share.iter().map(|s| (*s).into()).collect();

    let msm_mask = MsmMask::<G>::new(G::zero(), G::zero());
    d_msm::<G, _>(
        &x_share_aff,
        &y_share,
        &msm_mask,
        pp,
        net,
        MultiplexedStreamID::One,
    )
    .await
    .unwrap();
}

#[tokio::main]
async fn main() {
    env_logger::builder().format_timestamp(None).init();
    let network = Net::new_local_testnet(8).await.unwrap();

    network
        .simulate_network_round((), |net, _| async move {
            let pp = PackedSharingParams::<Fr>::new(2);
            for i in 10..14 {
                let dom = Radix2EvaluationDomain::<Fr>::new(1 << i).unwrap();
                println!("domain size: {}", dom.size());
                d_msm_test::<ark_bls12_377::G1Projective, _>(&pp, &dom, &net)
                    .await;
            }
        })
        .await;
}
