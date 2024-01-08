#![allow(non_snake_case, clippy::too_many_arguments)]

use ark_ec::pairing::Pairing;
use ark_ff::Zero;
use dist_primitives::dmsm::d_msm;
use mpc_net::{MpcNet, MpcNetError, MultiplexedStreamID};
use secret_sharing::pss::PackedSharingParams;

/// A = L.(N)^r.∏{i∈[0,m]}(S_i)^a_i
#[derive(Debug, Clone, Copy)]
pub struct A<'a, E: Pairing> {
    pub L: E::G1Affine,
    pub N: E::G1Affine,
    pub r: E::ScalarField,
    pub pp: &'a PackedSharingParams<E::ScalarField>,
    pub S: &'a [E::G1Affine],
    pub a: &'a [E::ScalarField],
}

impl<'a, E: Pairing> A<'a, E> {
    /// Computes A
    pub async fn compute<Net: MpcNet>(
        self,
        net: &Net,
        sid: MultiplexedStreamID,
    ) -> Result<E::G1, MpcNetError> {
        // We use variables (L, N, (S_i){i∈[0,m]}) to denote elements in G1
        // We also assume that all the servers computing the proof
        // get L, N in the clear and only receive packed shares of the remaining elements
        // specifically, each server gets a share of (S_i)^a_i for i∈[0,Q-1].
        //
        // Given packed shares of S_i and au_i terms, the servers can use πMSM (dmsm) to compute ∏{i∈[0,Q−1]}(S_i)^a_i.
        // Since the output of MSM are regular shares, they can then be combined with L, N and regular shares
        // of r to get regular shares of A.

        // Calculate (N)^r
        let v0 = self.N * self.r;
        // Calculate L.(N)^r
        let v1 = self.L + v0;

        // Calculate ∏{i∈[0,m]}(S_i)^a_i using dmsm
        let prod = d_msm::<E::G1, _>(
            self.S,
            self.a,
            E::G1::zero(),
            E::G1::zero(),
            self.pp,
            net,
            sid,
        )
        .await?;

        let A = v1 + prod;

        Ok(A)
    }
}

/// B = Z.(K)^s.∏{i∈[0,m]}(V_i)^a_i
#[derive(Debug, Clone, Copy)]
pub struct B<'a, E: Pairing> {
    pub Z: E::G2Affine,
    pub K: E::G2Affine,
    pub s: E::ScalarField,
    pub pp: &'a PackedSharingParams<E::ScalarField>,
    pub V: &'a [E::G2Affine],
    pub a: &'a [E::ScalarField],
}

impl<'a, E: Pairing> B<'a, E> {
    /// Computes B
    pub async fn compute<Net: MpcNet>(
        self,
        net: &Net,
        sid: MultiplexedStreamID,
    ) -> Result<E::G2, MpcNetError> {
        // We use variables (Z, K, (V_i){i∈[0,m]}) to denote elements in G2
        // We also assume that all the servers computing the proof
        // get Z, K in the clear and only receive packed shares of the remaining elements
        // specifically, each server gets a share of (V_i)^a_i for i∈[0,Q-1].
        //
        // Given packed shares of V_i and av_i terms, the servers can use πMSM (dmsm) to compute ∏{i∈[0,Q−1]}(V_i)^a_i.
        // Since the output of MSM are regular shares, they can then be combined with Z, K and regular shares
        // of s to get regular shares of A.
        // Calculate (K)^s
        let v0 = self.K * self.s;
        // Calculate Z.(K)^s
        let v1 = self.Z + v0;
        // Calculate ∏{i∈[0,m]}(V_i)^a_i using dmsm
        let prod = d_msm::<E::G2, _>(
            self.V,
            self.a,
            E::G2::zero(),
            E::G2::zero(),
            self.pp,
            net,
            sid,
        )
        .await?;

        let B = v1 + prod;

        Ok(B)
    }
}

/// C = (∏{i∈[l+1,m]}(W_i)^a_i)(∏{i∈[0,Q−2]}(U_i)^h_i).A^s.M^r.(∏{i∈[0,m]}(H_i)^a_i)^r
#[derive(Debug, Clone, Copy)]
pub struct C<'a, E: Pairing> {
    pub A: E::G1,
    pub M: E::G1Affine,
    pub s: E::ScalarField,
    pub r: E::ScalarField,
    pub pp: &'a PackedSharingParams<E::ScalarField>,
    pub W: &'a [E::G1Affine],
    pub U: &'a [E::G1Affine],
    pub H: &'a [E::G1Affine],
    pub a: &'a [E::ScalarField],
    pub ax: &'a [E::ScalarField],
    pub h: &'a [E::ScalarField],
}

impl<'a, E: Pairing> C<'a, E> {
    /// Computes C
    pub async fn compute<Net: MpcNet>(
        self,
        net: &Net,
    ) -> Result<E::G1, MpcNetError> {
        // We use variables (A, M, ∏{i∈[l+1,m]}(W_i)^a_i, ∏{i∈[0,Q−2]}(U_i)h_i, ∏{i∈[0,m]}(H_i)^a_i)
        // to denote elements in G1. We also assume that all the servers computing the proof
        // get A, M, s, r and h in the clear and only receive packed shares of the remaining elements.

        const CHANNEL0: MultiplexedStreamID = MultiplexedStreamID::Zero;
        const CHANNEL1: MultiplexedStreamID = MultiplexedStreamID::One;
        const CHANNEL2: MultiplexedStreamID = MultiplexedStreamID::Two;

        // todo: replace in_mask and out_mask

        // Calculate ∏{i∈[l+1,m]}(W_i)^a_i using dmsm
        let w = d_msm::<E::G1, _>(
            self.W,
            self.ax,
            E::G1::zero(),
            E::G1::zero(),
            self.pp,
            net,
            CHANNEL0,
        );
        // Calculate ∏{i∈[0,Q−2]}(U_i)^h_i using dmsm
        let u = d_msm::<E::G1, _>(
            self.U,
            self.h,
            E::G1::zero(),
            E::G1::zero(),
            self.pp,
            net,
            CHANNEL1,
        );
        // Calculate ∏{i∈[0,m]}(H_i)^a_i using dmsm
        let h = d_msm::<E::G1, _>(
            self.H,
            self.a,
            E::G1::zero(),
            E::G1::zero(),
            self.pp,
            net,
            CHANNEL2,
        );

        let (w, u, h) = tokio::try_join!(w, u, h)?;

        // Calculate A^s
        let v0 = self.A * self.s;
        // Calculate M^r
        let v1 = self.M * self.r;
        // Calculate (∏{i∈[0,m]}(H_i)^a_i)^r
        let v2 = h * self.r;
        // finally compute C
        let C = w + u + v0 + v1 + v2;
        Ok(C)
    }
}
