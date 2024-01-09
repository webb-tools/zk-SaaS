use ark_bls12_377::Fr;
use ark_bls12_377::G1Projective as G;
use ark_ec::CurveGroup;
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use ark_std::UniformRand;
use dist_primitives::dmsm::{d_msm, MsmMask};
use dist_primitives::utils::pack::transpose;
use mpc_net::ser_net::MpcSerNet;
use mpc_net::{LocalTestNet as Net, MpcNet, MultiplexedStreamID};
use rand::thread_rng;
use secret_sharing::pss::PackedSharingParams;

pub async fn d_msm_test<G: CurveGroup, Net: MpcNet>(
    x_pub: &Vec<G>,
    y_pub: &Vec<G::ScalarField>,
    x_share: &Vec<G>,
    y_share: &Vec<G::ScalarField>,
    msm_mask: &MsmMask<G>,
    pp: &PackedSharingParams<G::ScalarField>,
    net: &Net,
) {
    let x_pub_aff: Vec<G::Affine> = x_pub.iter().map(|s| (*s).into()).collect();
    let x_share_aff: Vec<G::Affine> =
        x_share.iter().map(|s| (*s).into()).collect();

    // Will be comparing against this in the end
    let should_be_output =
        G::msm(x_pub_aff.as_slice(), y_pub.as_slice()).unwrap();

    // fix the check here. output should actually be shares
    let output = d_msm::<G, Net>(
        &x_share_aff,
        &y_share,
        &msm_mask,
        pp,
        net,
        MultiplexedStreamID::One,
    )
    .await
    .unwrap();

    net.client_send_or_king_receive_serialized(
        &output,
        MultiplexedStreamID::One,
        pp.t,
    )
    .await
    .unwrap()
    .map(|rs| {
        let result = pp.unpack_missing_shares(&rs.shares, &rs.parties);
        assert_eq!(should_be_output, result[0]);
    });
}

#[tokio::main]
async fn main() {
    env_logger::builder().format_timestamp(None).init();
    let network = Net::new_local_testnet(8).await.unwrap();

    let pp = PackedSharingParams::<Fr>::new(2);
    let dom = Radix2EvaluationDomain::<Fr>::new(1 << 8).unwrap();

    let rng = &mut thread_rng();

    let mut y_pub: Vec<Fr> = Vec::new();
    let mut x_pub: Vec<G> = Vec::new();

    for _ in 0..dom.size() {
        y_pub.push(Fr::rand(rng));
        x_pub.push(G::rand(rng));
    }

    let x_shares: Vec<Vec<G>> = x_pub
        .chunks(pp.l)
        .map(|s| pp.pack(s.to_vec(), rng))
        .collect();
    let x_shares = transpose(x_shares);

    let y_shares: Vec<Vec<Fr>> = y_pub
        .chunks(pp.l)
        .map(|s| pp.pack(s.to_vec(), rng))
        .collect();
    let y_shares = transpose(y_shares);

    let msm_masks = MsmMask::sample(&pp, rng);

    network
        .simulate_network_round((x_pub, y_pub, x_shares, y_shares, msm_masks, pp), |net, (x_pub, y_pub, x_shares, y_shares, msm_masks, pp)| async move {
            let idx = net.party_id() as usize;
            d_msm_test::<ark_bls12_377::G1Projective, _>(&x_pub, &y_pub, &x_shares[idx], &y_shares[idx], &msm_masks[idx], &pp, &net).await;
        })
        .await;
}
