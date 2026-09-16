// The smoothed-prediction pass of the extension walk, kept in its own translation unit.
//
// This pass instantiates brokenline::prepareBrokenLineData, matrixC_uBandLower, circleBorderLower and
// the band solve for N = 3..12. Compiled alongside Kernel_BLFit, those extra instantiations change the
// compiler's inlining and FMA-contraction context for the track fit itself, and its float-accumulated
// results move at the ULP level without any change to its arithmetic. Keeping this file separate leaves
// BrokenLineFitKernels.h / BrokenLineFit.dev.cc generating exactly the code they generate without it.
// For the same reason the three container-build kernels below are renamed copies of the twin-refit
// trio in BrokenLineFit.dev.cc rather than a shared header: ~90 lines of pure bookkeeping (CSR offsets,
// container init, radius sort). If they ever diverge the symptom is a wrong anchor node.
#include <cstdlib>
#include "Utilities/Cadna/interface/CadnaEigenTypes.h"

#include "BrokenLineFitKernels.h"
#include "CAExtensionKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // ---------------------------------------------------------------------------------------------
  // Merger-side twin refit support: build a SequentialContainer over the MERGED track CSR so the
  // refitExtended machinery can be reused on the twin-united winners. off[0]=0, off[t+1] =
  // the merged track's cumulative hit-end (tracks.hitOffsets()); content ALIASES the merged
  // trackHits id column (bit30 OT tags preserved) so no hit copy is needed.
  // ---------------------------------------------------------------------------------------------
  class Kernel_extPredFillOffsets {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  caStructures::SequentialContainerOffsets* __restrict__ off,
                                  ::reco::TrackSoAConstView tracks,
                                  uint32_t nTracks) const {
      for (uint32_t t : cms::alpakatools::uniform_elements(acc, nTracks)) {
        if (t == 0)
          off[0] = 0;
        off[t + 1] = tracks[t].hitOffsets();  // cumulative CSR end of merged track t
      }
    }
  };

  // Wire the container header to its (already-filled) offsets + aliased content, WITHOUT zeroing
  // the offsets (unlike launchZero). Single grid thread.
  class Kernel_extPredInitContainer {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, caStructures::SequentialContainerView view) const {
      if (alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0] == 0) {
        view.assoc->psws = 0;
        view.assoc->initStorage(view);
      }
    }
  };

  // Trajectory-order and physically dedup the refit input of each united winner. Two union
  // pathologies break the GBL fit (which assumes the inside-out, one-measurement-per-crossing lists
  // the CA provides), showing up as huge chi2 / NaN covariance:
  //   (1) ORDER: Kernel_filterTracksScatter appends the loser's non-shared hits after the winner's, so
  //       the chain zig-zags in radius. Stable insertion sort by 3D radius x^2+y^2+z^2, which is
  //       monotonic along an outgoing near-prompt trajectory in both barrel and endcap (transverse r
  //       alone scrambles same-ring disk hits).
  //   (2) SAME PHYSICAL HIT, TWO IDS: one arm attached a raw OT rechit (bit30-tagged id), the other
  //       carries the same cluster as a merged-SoA stub id (or two stub rows share the lower cluster).
  //       Identical positions give a zero-length GBL segment and a singular normal matrix. The fit's
  //       own dedup walk only merges consecutive merged-pixel pairs, so after sorting a consecutive
  //       pair closer than kDedupDsMin (1 mm, here in 3D) is collapsed when at least one member is a
  //       stub/OT id; the span is compacted and the tail padded with an untagged out-of-range sentinel
  //       (the dedup walk break-terminates on it).
  // content is a copy, so the output SoA hit list is untouched. Only unitedMask[t] >= 0 spans are
  // processed.
  class Kernel_extPredSortHits {
  public:
    caExtension::OTHitsSource otSource_{};

    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  uint32_t* __restrict__ content,
                                  ::reco::TrackSoAConstView tracks,
                                  ::reco::TrackingRecHitConstView hh,
                                  const int32_t* __restrict__ unitedMask,
                                  uint32_t nTracks) const {
      const uint32_t nTot = uint32_t(hh.metadata().size());
      constexpr float kFar = 3.40282347e+38f;
      constexpr uint32_t kSentinel = 0x3FFFFFFFu;  // untagged, >= any real nTot -> walk break
      // Position + class of an id. Returns false for unresolvable ids (guards; sorted last).
      // isPixel = merged-SoA non-stub hit (the only class the walk itself dedups).
      auto posOf = [&](uint32_t id, float& x, float& y, float& z, bool& isPixel) -> bool {
        if (caOTHitTag::isOTId(id)) {
          const uint32_t o = caOTHitTag::otIdx(id);
          if (o >= otSource_.nOTHits)
            return false;
          x = otSource_.otHits[o].xGlobal();
          y = otSource_.otHits[o].yGlobal();
          z = otSource_.otHits[o].zGlobal();
          isPixel = false;
          return true;
        }
        if (id >= nTot)
          return false;
        x = hh[id].xGlobal();
        y = hh[id].yGlobal();
        z = hh[id].zGlobal();
        isPixel = !::reco::isStub(hh, int32_t(id));
        return true;
      };
      auto keyOf = [&](uint32_t id) -> float {
        float x, y, z;
        bool p;
        if (!posOf(id, x, y, z, p))
          return kFar;
        return x * x + y * y + z * z;
      };
      for (uint32_t t : cms::alpakatools::uniform_elements(acc, nTracks)) {
        if (unitedMask[t] < 0)
          continue;
        const uint32_t b = (t == 0) ? 0u : tracks[t - 1].hitOffsets();
        const uint32_t e = tracks[t].hitOffsets();
        // (1) stable insertion sort by 3D radius (spans <= the twin-merge hit cap, ~32)
        for (uint32_t i = b + 1; i < e; ++i) {
          const uint32_t v = content[i];
          const float kv = keyOf(v);
          uint32_t j = i;
          while (j > b && keyOf(content[j - 1]) > kv) {
            content[j] = content[j - 1];
            --j;
          }
          content[j] = v;
        }
        // (2) collapse same-physical-hit duplicates (vs the previous KEPT entry), compact, pad.
        uint32_t w = b;
        float px = 0.f, py = 0.f, pz = 0.f;
        bool pPix = false, havePrev = false;
        for (uint32_t i = b; i < e; ++i) {
          const uint32_t id = content[i];
          float x, y, z;
          bool isPix;
          const bool ok = posOf(id, x, y, z, isPix);
          if (ok && havePrev && !(isPix && pPix)) {
            const float dx = x - px, dy = y - py, dz = z - pz;
            if (dx * dx + dy * dy + dz * dz < caFitHitSel::kDedupDsMin2)
              continue;  // same physical measurement through a second id: drop
          }
          content[w++] = id;
          if (ok) {
            px = x;
            py = y;
            pz = z;
            pPix = isPix;
            havePrev = true;
          }
        }
        for (uint32_t i = w; i < e; ++i)
          content[i] = kSentinel;
      }
    }
  };

  // ================================================================================================
  // The smoothed-prediction functional, computed once per extension-walk host.
  //
  // The walk needs an honest prediction covariance at a layer it has not yet reached. Propagating the
  // fit's perigee 5x5 there comes out about a factor two too narrow in pull: the broken-line fit
  // interpolates, and its measurements outweigh its kinks by orders of magnitude, so no single-anchor
  // read-out is the smoother's prediction at the target. The correct object augments the band with a
  // virtual node at the target and gives node n-1 the kink degree of freedom the fit omits
  // (varBeta(n-1) == 0). Because the virtual node is unmeasured and that kink is unconstrained, the
  // marginal A^-1 on (u_{n-2}, u_{n-1}, dkappa) is unchanged by the augmentation, so the whole object
  // is that 3x3 marginal, obtained from the already factorised band by two unit-column solves plus a
  // 6-element Schur combination (Kleinwort's sparse inverse) -- the machinery circleFit already runs
  // for its (0, mref, n) block. No dense inverse, no second linearisation.
  //
  // Published per host: C_loc = T A^-1 T^T on (u [cm], u' [rad], kappa [1/cm]) at the last fitted
  // node, with T = [[0,1,0], [-1/h, 1/h, -h/2], [0,0,1]]; that node's transverse arc in the walk's own
  // convention and its global (r, z), the anchor of every road integral; and the last fitted gap's
  // exit-direction kink variance theta_T^2 (1 - m1^2/m2), the Brownian-bridge conditional variance of
  // the direction leaving a chord pinned at both ends, which the fit cannot see because that gap's
  // kink has no DOF. The walk forms V_u(ds) = g^T C_loc g with g = [1, ds, -ds^2/2], identically the
  // V_out = g^T A^-1 g of the bordered band.
  // ---------------------------------------------------------------------------------------------
  // The per-lane body of the coefficient pass, run by the fused all-bins launch through the out-of-line
  // trampoline below. Everything it reads is addressed relative to the base pointers it is handed, so a
  // caller can pass a per-bin slice of a larger allocation; the layout inside the slice is lane-strided.
  template <int N, uint32_t Stride>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void extPredCoeffLane(Acc1D const& acc,
                                                       uint32_t local_idx,
                                                       const double bField,
                                                       ::reco::TrackSoAConstView tracks,
                                                       caStructures::tindex_type const* __restrict__ ptkids,
                                                       double_st* __restrict__ phits,
                                                       float_st* __restrict__ phits_ge,
                                                       double_st* __restrict__ pfast_fit,
                                                       double_st* __restrict__ pscratch,
                                                       caExtension::ExtPredCoeff* __restrict__ out,
                                                       const float* __restrict__ rhoMap_,
                                                       const float* __restrict__ bMap_,
                                                       const bool fitCorrections_) {
    constexpr u_int n = N;
    constexpr auto invalidTkId = std::numeric_limits<typename caStructures::tindex_type>::max();
    const auto tkid = ptkids[local_idx];
    if (tkid == invalidTkId)
      return;  // the compacted list is dense; the tail is the invalid sentinel

    riemannFit::Map3xNdS<N, Stride> hits(phits + local_idx);
    riemannFit::Map6xNfS<N, Stride> hits_ge(phits_ge + local_idx);
    riemannFit::Map4dS<Stride> fast_fit(pfast_fit + local_idx);
    brokenline::PreparedBrokenLineDataMap<N, Stride> data(pscratch + local_idx);
    brokenline::LegacyFitWorkspaceMap<N, Stride> fitWs(
        pscratch + local_idx + std::size_t(brokenline::kPreparedDataDoubles<N>) * std::size_t(Stride));

    // The field this pass solves its band in. Kernel_BLFit publishes q/pT as (geometric
    // curvature)/bFieldEff, with bFieldEff = blEffectiveBField over the fitted hits; the band extracted
    // here is the covariance of that same trajectory and the anchor arc below inverts the same
    // conversion. Solving in the origin scalar instead would leave the two a factor B_eff/B(0,0) apart,
    // i.e. the forward field correction re-applied with the wrong sign to every road built from this
    // payload. Same helper, same hits, same gate as the fit; fitCorrections off (or a null map) gives
    // the scalar.
    const double bFieldEff = fitCorrections_ ? blEffectiveBField(acc, hits, int(N), fast_fit, bField, bMap_) : bField;
    // elossGaps records the running material column from hit 0 into fitWs.gapXX0(g) inside the gap
    // loop this call already runs (one add + one store per gap); it is the only extra input the
    // energy-loss road centre needs. Gated on fitCorrections_ exactly as Kernel_BLFit gates it.
    brokenline::prepareBrokenLineData(
        acc, hits, fast_fit, bFieldEff, rhoMap_, data, fitWs, fitCorrections_, /*elossGaps=*/fitCorrections_);

    // Taken from the circle fit: the fit's kink variable is the TRANSVERSE-frame angle, and everything
    // the walk consumes downstream is in that same frame. The alternative -- a 3-D scattering angle
    // multiplied by the transverse arc -- mixes frames.
    const double_st qCharge = data.qCharge;
    const double_st slope = -qCharge / fast_fit(3);
    data.varBeta *= 1. + riemannFit::sqr(slope);

    // As in the circle fit: the track-orthogonal projection of each hit's 2x2 xy error.
    auto& weightsVec = fitWs.circleWeights;
    {
      riemannFit::Matrix2d vMat;
      riemannFit::Matrix2d rotMat;
      for (u_int i = 0; i < n; i++) {
        vMat(0, 0) = hits_ge.col(i)[0];
        vMat(0, 1) = vMat(1, 0) = hits_ge.col(i)[1];
        vMat(1, 1) = hits_ge.col(i)[2];
        rotMat = brokenline::rotationMatrix(acc, -data.radii(0, i) / data.radii(1, i));
        weightsVec(i) = 1. / ((rotMat * vMat * rotMat.transpose())(1, 1));
      }
    }

    caExtension::ExtPredCoeff o{};
    o.valid = 0.f;
    const int ls = fitWs.laneStride;
    double_st* const Mb = fitWs.bandMb;
    double_st* const cbor = fitWs.bandCbor;
    double_st* const wcol = fitWs.bandW;
    double_st* const msc = fitWs.bandMs;
    brokenline::matrixC_uBandLower(weightsVec, data.sTransverse, data.varBeta, Mb, ls);
    fitWs.bandCorner[0] = brokenline::circleBorderLower(data.sTransverse, data.varBeta, cbor, ls);
    generalBrokenLine::bandFactor<2>(Mb, n, ls);
    for (u_int i = 0; i < n; i++)
      wcol[i * ls] = cbor[i * ls];
    generalBrokenLine::bandSolveInPlace<2>(Mb, wcol, n, ls);
    double_st schur = fitWs.bandCorner[0];
    for (u_int i = 0; i < n; i++)
      schur -= cbor[i * ls] * wcol[i * ls];

    if (schur > 0. && alpaka::math::isfinite(acc, schur)) {
      const u_int ia = n - 2;  // node n-2
      const u_int ib = n - 1;  // node n-1
      const double_st invS = 1. / schur;
      for (u_int i = 0; i < n; i++)
        msc[i * ls] = 0.;
      msc[ia * ls] = 1.;
      generalBrokenLine::bandSolveInPlace<2>(Mb, msc, n, ls);  // M^-1 e_{n-2}
      const double_st miaa = msc[ia * ls], miba = msc[ib * ls];
      for (u_int i = 0; i < n; i++)
        msc[i * ls] = 0.;
      msc[ib * ls] = 1.;
      generalBrokenLine::bandSolveInPlace<2>(Mb, msc, n, ls);  // M^-1 e_{n-1}
      const double_st miab = msc[ia * ls], mibb = msc[ib * ls];
      const double_st wa = wcol[ia * ls], wb = wcol[ib * ls];
      // A^-1(g,h) = M^-1(g,h) + w(g)w(h)/schur ; A^-1(g,n) = -w(g)/schur ; A^-1(n,n) = 1/schur
      const double_st aUU = miaa + wa * wa * invS;
      // M^-1 is symmetric; the two unit-column solves give the same off-diagonal twice, so average
      // them rather than pick one (the difference is pure round-off and the symmetrised value is the
      // one the T transform below assumes).
      const double_st aUV = 0.5 * (miab + miba) + wa * wb * invS;
      const double_st aVV = mibb + wb * wb * invS;
      const double_st aUK = -wa * invS;
      const double_st aVK = -wb * invS;
      const double_st aKK = invS;

      // T (u_{n-2}, u_{n-1}, dk) -> (u, u', kappa) at node n-1, p2b_local.loc_from.
      const double_st h = data.sTransverse(ib) - data.sTransverse(ia);
      if (alpaka::math::abs(acc, h) > 1e-6 && aUU > 0. && aVV > 0. && aKK > 0.) {
        const double_st t10 = -1. / h, t11 = 1. / h, t12 = -h / 2.;
        // row0 = (0,1,0) ; row1 = (t10,t11,t12) ; row2 = (0,0,1)
        const double_st c00 = aVV;
        const double_st c01 = t10 * aUV + t11 * aVV + t12 * aVK;
        const double_st c02 = aVK;
        const double_st c11 = t10 * (t10 * aUU + t11 * aUV + t12 * aUK) + t11 * (t10 * aUV + t11 * aVV + t12 * aVK) +
                           t12 * (t10 * aUK + t11 * aVK + t12 * aKK);
        const double_st c12 = t10 * aUK + t11 * aVK + t12 * aKK;
        const double_st c22 = aKK;
        if (c00 > 0. && c11 > 0. && c22 > 0. && alpaka::math::isfinite(acc, c11)) {
          o.c00 = float(c00);
          o.c01 = float(c01);
          o.c02 = float(c02);
          o.c11 = float(c11);
          o.c12 = float(c12);
          o.c22 = float(c22);
          // The last FITTED gap's exit-direction variance. varBeta(n-2) is the projected kink of the
          // gap that ARRIVES at node n-2; the SAME projected law on the LAST gap's material is
          // varBeta(n-2) * f(matXX0(n-2)) / f(matXX0(n-3)) with f(x) = x (1 + 0.038 ln x)^2, and the
          // Brownian-bridge factor (1 - m1^2/m2) conditions it on the chord being pinned at both hits.
          // m1/m2 come from the same material march the fit ran.
          double_st kink = 0.;
          if constexpr (N >= 4) {
            auto fxx = [&](double_st x) {
              const double_st xs = x > 1e-12 ? x : static_cast<double_st>(1e-12);
              const double_st lg = 1. + 0.038 * alpaka::math::log(acc, xs);
              return xs * lg * lg;
            };
            const double_st mPrev = data.matXX0(n - 3);
            const double_st mLast = data.matXX0(n - 2);
            const double_st vbPrev = data.varBeta(n - 2);
            if (mPrev > 1e-9 && mLast > 1e-9 && vbPrev > 0.)
              kink = vbPrev * fxx(mLast) / fxx(mPrev);
          }
          double_st d1 = 0., w1 = 0.;
          const double_st rA = alpaka::math::sqrt(acc, hits(0, ia) * hits(0, ia) + hits(1, ia) * hits(1, ia));
          const double_st rB = alpaka::math::sqrt(acc, hits(0, ib) * hits(0, ib) + hits(1, ib) * hits(1, ib));
          double_st bridge = 0.25;  // the uniform-material limit of 1 - m1^2/m2
          if (rhoMap_ != nullptr) {
            const double_st Wg = brokenline::segmentXX0Moments(acc, rhoMap_, rA, hits(2, ia), rB, hits(2, ib), d1, w1);
            // S1 = w1 W d1, S2 = w1 W d1^2 => 1 - m1^2/m2 = 1 - (S1/W)^2/(S2/W) = 1 - w1
            if (Wg > 0. && w1 > 0. && w1 <= 1.)
              bridge = 1. - w1;
          }
          o.qgapCoef = float(alpaka::math::max(acc, static_cast<double>(kink * bridge), 0.));

          // The anchor, in the WALK's own arc convention: the walk measures every arc from the
          // helix's alphaOrigin (its PCA), so the payload must publish node n-1's arc in THAT frame,
          // not in the band's. Rebuild the walk's helix from the published state and take the arc of
          // the last fitted hit on it -- the same closed form predictOn* inverts.
          const float phi0 = tracks[tkid].state()(0);
          const float tip = tracks[tkid].state()(1);
          const float invPt = tracks[tkid].state()(2);
          const float bf = float(bFieldEff);
          const float rhoH = (invPt != 0.f) ? 1.f / (invPt * bf) : 1e9f;
          const float sp = alpaka::math::sin(acc, phi0);
          const float cp = alpaka::math::cos(acc, phi0);
          const float xc = tip * sp + rhoH * sp;
          const float yc = -tip * cp - rhoH * cp;
          const float alphaOrigin = alpaka::math::atan2(acc, -yc, -xc);
          const float alphaH = alpaka::math::atan2(acc, float(hits(1, ib)) - yc, float(hits(0, ib)) - xc);
          float dAlpha = alphaOrigin - alphaH;
          constexpr float kPi = 3.14159265f;
          if (dAlpha > kPi)
            dAlpha -= 2.f * kPi;
          if (dAlpha < -kPi)
            dAlpha += 2.f * kPi;
          o.anchorS = rhoH * dAlpha;
          o.anchorR = float(rB);
          o.anchorZ = float(hits(2, ib));
          // Published so the WALK inverts the same conversion the fit applied (see ExtPredCoeff).
          o.bFieldEff = float(bFieldEff);

          // ---- The ionization energy-loss road centre (see ExtPredCoeff's eloss block) ----------
          // Same scalar, same Landau law and the same O(N) march the fast fit runs inside circleFit,
          // on the same hits, in the same field, behind the same gate. It yields (elossU, elossUp),
          // the fit's deterministic offset and its arc derivative at the walk's anchor (node n-1),
          // and (elossDkAnchor, elossK), the curvature excess accumulated there plus its growth rate
          // per unit material column. Downstream everything is a second-order polynomial in the arc
          // with those four coefficients. The expansion is about the anchor because the walk measures
          // every arc from anchorS and C_loc is referenced there too; it is exact,
          // u(a + ds) = u(a) + u'(a) ds - int_0^ds (ds-s') dkappa(s') ds', so moving the reference
          // loses nothing.
          if (fitCorrections_) {
            const double_st xTot = data.innerXX0 + fitWs.gapXX0(int(N) - 2);  // beamline -> node n-1 [X/X0]
            const double_st pTot =
                alpaka::math::sqrt(acc, riemannFit::sqr(bFieldEff * fast_fit(2)) * (1. + riemannFit::sqr(slope)));
            if (xTot > 0. && pTot > 0. && fast_fit(2) > 0.) {
              // kappa_0 * (dE/dX)_eff / p, with (dE/dX)_eff = dE(X_tot)/X_tot: the linearisation is
              // anchored on the total column exactly as in Kernel_BLFit, so the two agree by
              // construction on the span the fit measured.
              const double_st kEl = (generalBrokenLine::elossTypicalColumn(acc, pTot, xTot) / xTot) / (pTot * fast_fit(2));
              // The fit's march (BrokenLine.h circleFit): mid-gap column quadrature, u(0) = u'(0) = 0
              // at node 0. gapXX0 is only read here, so no in-place slot juggling is needed.
              double_st xPrev = data.innerXX0;                   // column at node 0
              double_st xCur = data.innerXX0 + fitWs.gapXX0(0);  // column at node 1
              double_st du = 0., dup = 0.;
              for (u_int k = 0; k + 1 < n; ++k) {
                const double_st dsg = data.sTransverse(k + 1) - data.sTransverse(k);
                const double_st dk = kEl * 0.5 * (xPrev + xCur);  // mid-gap curvature increment [1/cm]
                du += dup * dsg - 0.5 * dk * dsg * dsg;
                dup -= dk * dsg;
                xPrev = xCur;
                if (k + 2 < n)
                  xCur = data.innerXX0 + fitWs.gapXX0(k + 1);
              }
              if (alpaka::math::isfinite(acc, du) && alpaka::math::isfinite(acc, dup) &&
                  alpaka::math::isfinite(acc, kEl)) {
                o.elossK = float(kEl);
                o.elossDkAnchor = float(kEl * xTot);
                o.elossU = float(du);
                o.elossUp = float(dup);
              }
            }
          }
          if (alpaka::math::isfinite(acc, o.anchorS) && o.anchorR > 0.f)
            o.valid = 1.f;
        }
      }
    }
    out[tkid] = o;
  }

  // Out-of-line trampoline for the fused launch. Inlining all ten compile-time-N bodies into one entry
  // point makes the compiler lay their frames out side by side instead of overlapping them, roughly
  // tripling the per-thread stack, and every byte of stack is reserved for every resident thread.
  // Behind a call each body is compiled once with no stack of its own, and the entry kernel keeps only
  // the CUDA ABI save/restore area around the call (the caller preserving its live registers, which is
  // why passing the SoA view by reference does not shrink it).
#if defined(__CUDACC__)
#define EXT_PRED_NOINLINE __noinline__
#else
#define EXT_PRED_NOINLINE
#endif
  template <int N, uint32_t Stride>
  ALPAKA_FN_ACC EXT_PRED_NOINLINE void extPredCoeffLaneOutOfLine(Acc1D const& acc,
                                                                 uint32_t local_idx,
                                                                 const double bField,
                                                                 ::reco::TrackSoAConstView tracks,
                                                                 caStructures::tindex_type const* __restrict__ ptkids,
                                                                 double_st* __restrict__ phits,
                                                                 float_st* __restrict__ phits_ge,
                                                                 double_st* __restrict__ pfast_fit,
                                                                 double_st* __restrict__ pscratch,
                                                                 caExtension::ExtPredCoeff* __restrict__ out,
                                                                 const float* __restrict__ rhoMap_,
                                                                 const float* __restrict__ bMap_,
                                                                 const bool fitCorrections_) {
    extPredCoeffLane<N, Stride>(acc,
                                local_idx,
                                bField,
                                tracks,
                                ptkids,
                                phits,
                                phits_ge,
                                pfast_fit,
                                pscratch,
                                out,
                                rhoMap_,
                                bMap_,
                                fitCorrections_);
  }
#undef EXT_PRED_NOINLINE

  // ---------------------------------------------------------------------------------------------
  // N-exact per-bin slice bases of the prediction ladder.
  //
  // Every address a lane of bin b touches is built by Map3xNdS<N, Stride> / Map6xNfS<N, Stride> /
  // Map4dS<Stride> / PreparedBrokenLineDataMap<N, Stride> + LegacyFitWorkspaceMap<N, Stride> from the
  // slice base it is handed, using the bin's own compile-time N = b + kExtPredMinN (Kernel_BLFastFitRefit
  // <N, Stride>, which fills the slice, maps hits/hits_ge/fast_fit with the same <N, Stride>). So a bin
  // needs only its quota -- 3N doubles and 6N floats per lane for hits/hits_ge, 4 doubles for fast_fit,
  // kLegacyFitScratchDoubles<N> = 23N+3 doubles for the band scratch -- and the slice base is the prefix
  // sum below rather than b x (the N = kExtPredMaxN quota): Sum_{N=3..12} N = 75 against 120,
  // Sum (23N+3) = 1755 against 2790. Intra-slice layout, lane stride, column order and floating-point
  // operation order are the per-bin launch's; a slice is written and read by exactly one bin (the scans
  // partition the hosts by hit multiplicity); and the bases are multiples of Stride elements, so
  // alignment holds.
  //
  // Sum of N over bins [0, B):  B * kExtPredMinN + B*(B-1)/2.
  template <int B>
  inline constexpr int kExtPredNPrefix = B * caExtension::kExtPredMinN + B * (B - 1) / 2;
  // Sum of the per-lane band-scratch quota over bins [0, B). Recursive so it tracks
  // brokenline::kLegacyFitScratchDoubles<N> instead of restating its formula.
  template <int B>
  inline constexpr int kExtPredScrPrefix =
      kExtPredScrPrefix<B - 1> + brokenline::kLegacyFitScratchDoubles<(B - 1) + caExtension::kExtPredMinN>;
  template <>
  inline constexpr int kExtPredScrPrefix<0> = 0;

  template <uint32_t Stride, int B>
  inline constexpr std::size_t kExtPredHitsBase = std::size_t(Stride) * 3u * std::size_t(kExtPredNPrefix<B>);
  template <uint32_t Stride, int B>
  inline constexpr std::size_t kExtPredGeBase = std::size_t(Stride) * 6u * std::size_t(kExtPredNPrefix<B>);
  template <uint32_t Stride, int B>
  inline constexpr std::size_t kExtPredFfBase = std::size_t(Stride) * 4u * std::size_t(B);
  template <uint32_t Stride, int B>
  inline constexpr std::size_t kExtPredScrBase = std::size_t(Stride) * std::size_t(kExtPredScrPrefix<B>);
  template <uint32_t Stride, int B>
  inline constexpr std::size_t kExtPredTkBase = std::size_t(Stride) * std::size_t(B);

  static_assert(kExtPredNPrefix<int(caExtension::kExtPredNBins)> == 75, "N-exact hits/hits_ge quota moved");
  static_assert(kExtPredScrPrefix<int(caExtension::kExtPredNBins)> == 1755, "N-exact band-scratch quota moved");

  // ---------------------------------------------------------------------------------------------
  // The fused coefficient pass.
  //
  // Run per bin, this pass is ten serialised launches, one per compile-time N, each covering the
  // whole kRefitStride lane space although the walk hosts of an event, spread over ten bins, fill only
  // a few dense lanes; the kernel is latency-bound (one thread marching the material line integrals of
  // prepareBrokenLineData) and occupancy-starved. The bins are logically independent, so they share
  // one launch instead: the grid carries kExtPredNBins x (blocks per bin), block b belongs to bin
  // b / blocksPerBin, every bin is resident at once and the wall time is the slowest bin rather than
  // the sum. Each bin keeps its own compile-time N (the fit's matrix sizes are structural) and needs
  // its own scratch, sized to its own N via the prefix tables above (~26 MB for the ladder rather than
  // the ~46 MB of uniform kExtPredMaxN slices). The ten scan launches stay per bin: they are grid-rich
  // and live in the fit's own header, which this file does not touch.
  //
  // Identical results by construction: bin b runs extPredCoeffLane<b+3, Stride> on slice b of each
  // buffer with the same lane index and Stride, hence the same addresses relative to its slice base as
  // the per-bin launch. No reduction, no shared state, no cross-lane communication; out[tkid] is
  // written by exactly one lane of exactly one bin.
  template <uint32_t Stride>
  struct Kernel_extPredCoeffFused {
    const float* __restrict__ rhoMap_ = nullptr;
    const float* __restrict__ bMap_ = nullptr;
    bool fitCorrections_ = false;

    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  const double bField,
                                  ::reco::TrackSoAConstView tracks,
                                  caStructures::tindex_type const* __restrict__ ptkids,
                                  double_st* __restrict__ phits,
                                  float_st* __restrict__ phits_ge,
                                  double_st* __restrict__ pfast_fit,
                                  double_st* __restrict__ pscratch,
                                  caExtension::ExtPredCoeff* __restrict__ out) const {
      constexpr uint32_t kTotal = caExtension::kExtPredNBins * Stride;
      for (uint32_t grp : cms::alpakatools::uniform_groups(acc, kTotal)) {
        for (auto el : cms::alpakatools::uniform_group_elements(acc, grp, kTotal)) {
          const uint32_t bin = uint32_t(el.global) / Stride;
          const uint32_t lane = uint32_t(el.global) - bin * Stride;
          // The slice bases are the N-EXACT prefix sums, folded per case: the switch already carries
          // the bin as a literal, so every base below is a compile-time constant of the same case the
          // lane's compile-time N comes from -- there is no runtime table and no divergence added.
#define EXT_PRED_FUSED_CASE(BIN)                                                                                    \
  case (BIN):                                                                                                       \
    extPredCoeffLaneOutOfLine<(BIN) + caExtension::kExtPredMinN, Stride>(acc,                                       \
                                                                         lane,                                      \
                                                                         bField,                                    \
                                                                         tracks,                                    \
                                                                         ptkids + kExtPredTkBase<Stride, (BIN)>,    \
                                                                         phits + kExtPredHitsBase<Stride, (BIN)>,   \
                                                                         phits_ge + kExtPredGeBase<Stride, (BIN)>,  \
                                                                         pfast_fit + kExtPredFfBase<Stride, (BIN)>, \
                                                                         pscratch + kExtPredScrBase<Stride, (BIN)>, \
                                                                         out,                                       \
                                                                         rhoMap_,                                   \
                                                                         bMap_,                                     \
                                                                         fitCorrections_);                          \
    break
          switch (bin) {
            EXT_PRED_FUSED_CASE(0);
            EXT_PRED_FUSED_CASE(1);
            EXT_PRED_FUSED_CASE(2);
            EXT_PRED_FUSED_CASE(3);
            EXT_PRED_FUSED_CASE(4);
            EXT_PRED_FUSED_CASE(5);
            EXT_PRED_FUSED_CASE(6);
            EXT_PRED_FUSED_CASE(7);
            EXT_PRED_FUSED_CASE(8);
            EXT_PRED_FUSED_CASE(9);
            default:
              break;
          }
#undef EXT_PRED_FUSED_CASE
        }
      }
    }
  };

  namespace caExtension {

    // The smoothed-prediction pass. Same compaction substrate as the extended
    // refit (Kernel_BLFastFitRefit bins the masked population by selected-hit multiplicity and fills
    // the Eigen scratch maps), but the fit half is replaced by the band-only marginal extraction, and
    // NOTHING is written back to the track SoA. A FREE FUNCTION rather than a HelixFit member, so this
    // TU never has to see -- or re-instantiate -- the fit's class template.
    void launchExtPredCoeff(Queue& queue,
                            ::reco::TrackingRecHitConstView hv,
                            ::reco::CAModulesConstView cm,
                            ::reco::TrackSoAView mergedTracks,
                            ::reco::TrackHitSoAView mergedHits,
                            const int32_t* hostMask,
                            const OTHitsSource* otSource,
                            float bField_,
                            const float* rhoMap_,
                            const float* bMap_,
                            bool fitCorrections_,
                            ExtPredCoeff* out,
                            uint32_t nTracksCap) {
      {
        if (out == nullptr || hostMask == nullptr || nTracksCap == 0)
          return;
        const uint32_t maxNumberOfTuples = nTracksCap;
        using pixelTopology::Phase2OTStubs;
        constexpr uint32_t kRefitStride = HelixFit<Phase2OTStubs>::kRefitStride;
        constexpr uint32_t kRefitMaxN = HelixFit<Phase2OTStubs>::kRefitMaxN;
        // The merged-CSR container, built exactly as refitMergedTwins builds it (offsets from
        // hitOffsets, content a copy of the trackHits id column with the bit30 OT tags preserved),
        // then TRAJECTORY-ORDERED over the selected hosts: the band's last node must be the OUTERMOST
        // fitted hit, because that node is the anchor every road integral and every Delta-s is measured
        // from. Nothing is written back to the tracks.
        const uint32_t nHitsTot = uint32_t(mergedHits.metadata().size());
        auto contHeader = cms::alpakatools::make_device_buffer<caStructures::SequentialContainer>(queue);
        auto offBuf = cms::alpakatools::make_device_buffer<caStructures::SequentialContainerOffsets[]>(
            queue, std::size_t(nTracksCap) + 1u);
        auto contentBuf = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, std::max(nHitsTot, 1u));
        if (nHitsTot > 0) {
          auto src = cms::alpakatools::make_device_view(alpaka::getDev(queue), mergedHits.id().data(), nHitsTot);
          auto dst = cms::alpakatools::make_device_view(alpaka::getDev(queue), contentBuf.data(), nHitsTot);
          alpaka::memcpy(queue, dst, src);
        }
        caStructures::SequentialContainerView cview{
            contHeader.data(), offBuf.data(), contentBuf.data(), nTracksCap + 1u, std::max(nHitsTot, 1u)};
        {
          constexpr uint32_t bs = 128;
          const uint32_t nb = cms::alpakatools::divide_up_by(nTracksCap, bs);
          alpaka::exec<Acc1D>(queue,
                              cms::alpakatools::make_workdiv<Acc1D>(nb, bs),
                              Kernel_extPredFillOffsets{},
                              offBuf.data(),
                              mergedTracks,
                              nTracksCap);
          const caExtension::OTHitsSource otSrcSort =
              (otSource != nullptr && otSource->nOTHits > 0u) ? *otSource : caExtension::OTHitsSource{};
          alpaka::exec<Acc1D>(queue,
                              cms::alpakatools::make_workdiv<Acc1D>(nb, bs),
                              Kernel_extPredSortHits{otSrcSort},
                              contentBuf.data(),
                              mergedTracks,
                              hv,
                              hostMask,
                              nTracksCap);
          alpaka::exec<Acc1D>(
              queue, cms::alpakatools::make_workdiv<Acc1D>(1u, 1u), Kernel_extPredInitContainer{}, cview);
        }
        caStructures::SequentialContainer const* hitContainer = contHeader.data();
        const caExtension::OTHitsSource otSrcVal =
            (otSource != nullptr && otSource->nOTHits > 0u) ? *otSource : caExtension::OTHitsSource{};
        constexpr uint32_t nt = kRefitStride;
        // The N-bin ladder is a shared constant (CAExtensionKernels.h) because the FUSED launch below
        // maps grid blocks onto bins; bin b carries N = b + kExtPredMinN, the top bin absorbing the tail.
        static_assert(int(kRefitMaxN) == kExtPredMaxN, "the coefficient pass's N-bin ladder must track kRefitMaxN");
        constexpr uint32_t nBins = kExtPredNBins;
        // Per-bin slices of one allocation each, N-exact: every bin owns its scratch so all ten can be
        // in flight at once, sized for the bin's own N (the prefix tables at the head of this file):
        // Stride*3*75 doubles + Stride*6*75 floats + Stride*4*10 doubles + Stride*1755 doubles. Inside
        // a slice the layout is the per-bin launch's, so the arithmetic is identical.
        constexpr std::size_t kHitsTotal = kExtPredHitsBase<nt, int(nBins)>;
        constexpr std::size_t kGeTotal = kExtPredGeBase<nt, int(nBins)>;
        constexpr std::size_t kFfTotal = kExtPredFfBase<nt, int(nBins)>;
        constexpr std::size_t kScrTotal = kExtPredScrBase<nt, int(nBins)>;
        constexpr std::size_t kTkTotal = kExtPredTkBase<nt, int(nBins)>;

        auto tkidDevice = cms::alpakatools::make_device_buffer<typename caStructures::tindex_type[]>(queue, kTkTotal);
        auto hitsDevice = cms::alpakatools::make_device_buffer<double_st[]>(queue, kHitsTotal);
        auto hits_geDevice = cms::alpakatools::make_device_buffer<float_st[]>(queue, kGeTotal);
        auto fast_fit_resultsDevice = cms::alpakatools::make_device_buffer<double_st[]>(queue, kFfTotal);
        // Band scratch: kLegacyFitScratchDoubles<N> = 23N+3 doubles per lane ([prepared 7N | band
        // block 6N+1 | helpers 10N+2]), strided by kRefitStride.
        auto scratchDevice = cms::alpakatools::make_device_buffer<double_st[]>(queue, kScrTotal);
        auto slotDevice = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, nBins);

        constexpr uint32_t blockSize = 64;
        const uint32_t numberOfBlocksFit = cms::alpakatools::divide_up_by(nt, blockSize);
        const WorkDiv1D workDivFitFused = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocksFit * nBins, blockSize);
        const uint32_t numberOfBlocksScan =
            cms::alpakatools::divide_up_by(std::max<uint32_t>(1u, maxNumberOfTuples), blockSize);
        const WorkDiv1D workDivScan = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocksScan, blockSize);
        const uint32_t maxFitSel = pixelTopology::Phase2OTStubs::maxHitsOnTrack - 1;

        // One memset for the whole ladder.
        alpaka::memset(queue, slotDevice, 0);
        alpaka::memset(queue, tkidDevice, 0xff);

        // The fit runs as one fused launch over the whole ladder (Kernel_extPredCoeffFused).

        // The scans stay per bin: they are grid-rich (they sweep the whole track list) and live in the
        // fit's own header. Each writes into its own bin's slice, so all ten survive to the single fit
        // launch that follows. Bin b is N - kExtPredMinN at every call site below (the top bin's
        // catch-all scan included), so it is derived from the compile-time N tag and the N-exact slice
        // bases are compile-time constants of the same N the kernel is instantiated with; the base and
        // the intra-slice layout cannot disagree.
        auto launchScan = [&](auto Ntag, uint32_t nHitsL, uint32_t nHitsH) {
          constexpr int Nv = decltype(Ntag)::value;
          constexpr int Bv = Nv - kExtPredMinN;
          alpaka::exec<Acc1D>(queue,
                              workDivScan,
                              Kernel_BLFastFitRefit<Nv, kRefitStride>{otSrcVal, nullptr, nullptr},
                              hitContainer,
                              hv,
                              cm,
                              hostMask,
                              tkidDevice.data() + kExtPredTkBase<nt, Bv>,
                              hitsDevice.data() + kExtPredHitsBase<nt, Bv>,
                              hits_geDevice.data() + kExtPredGeBase<nt, Bv>,
                              fast_fit_resultsDevice.data() + kExtPredFfBase<nt, Bv>,
                              slotDevice.data() + Bv,
                              nHitsL,
                              nHitsH,
                              maxNumberOfTuples,
                              /*hasStubs=*/true);
        };
        launchScan(std::integral_constant<int, kExtPredMinN>{}, uint32_t(kExtPredMinN), uint32_t(kExtPredMinN));
        riemannFit::rolling_fits<kExtPredMinN + 1, int(kRefitMaxN), 1>([&](auto i) {
          constexpr int Nv = decltype(i)::value;
          launchScan(i, uint32_t(Nv), uint32_t(Nv));
        });
        launchScan(std::integral_constant<int, int(kRefitMaxN)>{}, kRefitMaxN, maxFitSel);

        Kernel_extPredCoeffFused<kRefitStride> fusedKernel{rhoMap_, bMap_};
        fusedKernel.fitCorrections_ = fitCorrections_;
        alpaka::exec<Acc1D>(queue,
                            workDivFitFused,
                            fusedKernel,
                            double(bField_),
                            mergedTracks,
                            tkidDevice.data(),
                            hitsDevice.data(),
                            hits_geDevice.data(),
                            fast_fit_resultsDevice.data(),
                            scratchDevice.data(),
                            out);
      }
    }

  }  // namespace caExtension

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
