#ifndef RecoTracker_PixelSeeding_plugins_alpaka_BrokenLineFitKernels_h
#define RecoTracker_PixelSeeding_plugins_alpaka_BrokenLineFitKernels_h

// Shared BL-fit device kernels + per-N launcher helpers for the SPLIT build.
// Every heavy per-N kernel is `extern template` here and explicitly instantiated in exactly one small
// N-range TU (BrokenLineFit_*.dev.cc), so nvcc compiles disjoint kernel subsets in parallel. Build-time
// division only: the kernels and their launch arguments are unchanged either way.

// #define BROKENLINE_DEBUG
// #define GPU_DEBUG
// #define FIT_DEBUG
#include <cstdint>

#if defined(FIT_DEBUG) || defined(GPU_DEBUG)
#include <iostream>
#endif

#include <alpaka/alpaka.hpp>
#include "Utilities/Cadna/interface/CadnaEigenTypes.h"
#include "Utilities/Cadna/interface/CadnaOutput.h"

#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometrySoA.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/BrokenLine.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/GeneralBrokenLine.h"  // unfactorized GBL fit
#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMap.h"
#include "RecoTracker/PixelTrackFitting/interface/BLBFieldMap.h"  // (Bz,Br) r-z map for the merger GBL beff

#include "CAFitHitSelection.h"
#include "CAExtensionKernels.h"                           // caExtension::OTHitsSource (extended-N refit OT hit source)
#include "RecoTracker/PixelSeeding/interface/OTHitTag.h"  // caOTHitTag::isOTId / otIdx (refit tagged-id fetch)
#include "HelixFit.h"

using OutputSoAView = reco::TrackSoAView;
using TupleMultiplicity = caStructures::GenericContainer;
using Tuples = caStructures::SequentialContainer;
namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // CUDA block dim of the fit work divisions (launch dim only: the kernels are grid-stride loops that
  // break on the invalid-tkid sentinel). 32 is the L1-pressure optimum of this solver. Kept here so
  // the FUSED refit ladder can express its bin <-> lane partition in units of it.
  inline constexpr uint32_t kFitBlock = 32u;

  // ---------------------------------------------------------------------------------------------
  // FUSED REFIT LADDER -- the dynamic bin <-> lane-range partition. The ten N-bins partition one
  // lane-strided buffer set into disjoint lane ranges, so all ten run in one fused launch per
  // (phase, iteration). The partition is computed per event on the device from the actual populations
  // (no fixed per-bin quota, nothing crosses to the host): Kernel_BLRefitBinCount fills the ten per-bin
  // demands; Kernel_BLRefitLaneRanges prefix-sums them into contiguous ranges [base_b, base_b+n_b) and
  // writes the block->bin dispatch table the fused kernels read.
  // Disjointness: hits/hits_ge/fast_fit are column-major (stride kRefitStride), so two lanes never share
  // a byte; pphase is lane-major at the fixed per-lane quota kBLPhaseDoubles; pgnodes/pgblScratch are
  // lane-major with a per-lane quota that is non-decreasing in N (kRefitQuotaMonotone below pins this),
  // so the ordered ranges stay disjoint for any partition.
  // Capacity: one pass seats kRefitStride lanes; unseated tracks wait for the next round (exact through
  // the per-track "served" flag), and ceil(nTracksCap/kRefitStride) rounds always drain the population.
  inline constexpr uint32_t kRefitNBins = 10u;  // N = 3 .. kRefitMaxN, the top bin absorbing the tail
  inline constexpr int kRefitMinN = 3;

  // Device tables of one round. pRange holds {firstLane, nLanes} per bin; pBlockMap holds
  // {bin, firstLane, endLane} per fused-grid block, with bin == kRefitNBins marking an idle block.
  inline constexpr uint32_t kRefitRangeStride = 2u;
  inline constexpr uint32_t kRefitBlockMapStride = 3u;

  // Fused-grid width covers every partition the count can produce: a bin holding n_b lanes needs
  // ceil(n_b/kFitBlock) blocks, and sum_b <= kRefitStride/kFitBlock + kRefitNBins (a block carries
  // exactly one bin, so the switch over compile-time N never diverges inside a warp).
  inline constexpr uint32_t kFusedBlocks =
      HelixFit<::pixelTopology::Phase2OTStubs>::kRefitStride / kFitBlock + kRefitNBins;

  // Per-lane pgnodes/pgblScratch quotas -- the SAME expressions the phase kernels build their per-lane
  // pointers with, restated so the disjointness static_assert below can reference them.
  template <int N, typename TrackerTraits>
  inline constexpr int kRefitSlotNodes = generalBrokenLine::kGblSplitNodes<(
      int(N) > int(TrackerTraits::maxHitsOnTrackForFullFit) ? int(N) : int(TrackerTraits::maxHitsOnTrackForFullFit))>;
  template <int N>
  inline constexpr int kRefitScratchStride =
      generalBrokenLine::kGblScratchDoubles<generalBrokenLine::kGblSplitNodes<N> - 1> +
      3 * generalBrokenLine::kGblSplitNodes<N>;
  template <typename TrackerTraits, int N>
  inline constexpr bool kRefitQuotaMonotone =
      kRefitQuotaMonotone<TrackerTraits, N - 1> &&
      (kRefitSlotNodes<N - 1, TrackerTraits> <= kRefitSlotNodes<N, TrackerTraits>) &&
      (kRefitScratchStride<N - 1> <= kRefitScratchStride<N>);
  template <typename TrackerTraits>
  inline constexpr bool kRefitQuotaMonotone<TrackerTraits, kRefitMinN> = true;
  static_assert(
      kRefitQuotaMonotone<::pixelTopology::Phase2OTStubs, int(HelixFit<::pixelTopology::Phase2OTStubs>::kRefitMaxN)>,
      "the ordered lane ranges only stay disjoint in pgnodes/pgblScratch while the per-lane "
      "quotas are non-decreasing in N");
  static_assert(kRefitNBins == HelixFit<::pixelTopology::Phase2OTStubs>::kRefitMaxN - uint32_t(kRefitMinN) + 1u,
                "the bin ladder must cover N = kRefitMinN .. kRefitMaxN");

  // Per-lane quota of the fit-hit-id / pixel-core-flag tables. Under the lane partition the scan works in
  // bin-local slots while the fused outlier works in absolute lanes, so the quota must be the
  // ladder-wide kRefitMaxN for every bin (same values, uniform addressing).
  inline constexpr uint32_t kRefitFitIdQuota = HelixFit<::pixelTopology::Phase2OTStubs>::kRefitMaxN;

  // Selected-hit multiplicity -> bin. THE ONE DEFINITION of the ladder's binning, so the count kernel
  // and the per-bin scans cannot drift apart: it reproduces exactly the (nHitsL, nHitsH) pairs
  // refitExtended launches the scans with -- the exact bins N = kRefitMinN .. kRefitMaxN-1, and the
  // top bin absorbing the tail [kRefitMaxN, maxFitSel]. Returns kRefitNBins for a track the ladder
  // does not fit at all.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t refitBinOfNSel(uint32_t nSel, uint32_t maxFitSel) {
    constexpr uint32_t kMaxN = HelixFit<::pixelTopology::Phase2OTStubs>::kRefitMaxN;
    if (nSel < uint32_t(kRefitMinN) || nSel > maxFitSel)
      return kRefitNBins;
    const uint32_t nEff = (nSel < kMaxN) ? nSel : kMaxN;
    return nEff - uint32_t(kRefitMinN);
  }

  // ---------------------------------------------------------------------------------------------
  // FUSED MAIN-FIT LADDER -- the refit ladder's dynamic partition (above) applied to the CA fit. The
  // bins are disjoint in tuples (tupleMultiplicity bins every tuple by its selected multiplicity, and
  // begin(nHitsL)..end(nHitsH) are contiguous non-overlapping slices of it), so two bins never write the
  // same SoA row; they partition the one lane-strided buffer set into disjoint lane ranges and run in
  // one fused launch per phase. No count kernel is needed: bin b's demand is end(nHitsH) - begin(nHitsL),
  // the expression the per-bin fast fit's totTK reads; Kernel_BLMainLaneRanges (single thread) prefix-sums
  // the demands into ranges and writes the block -> bin dispatch table. Lane disjointness is
  // unconditional: every per-lane buffer is column-major with the fixed inter-element stride
  // riemannFit::stride, so lane l occupies exactly the elements {l + e*stride} whatever N it carries.
  // One pass seats riemannFit::maxNumberOfConcurrentFits lanes, handed out in bin order; a per-bin cursor
  // (lane l of bin b is that bin's (tupleBase_b + l - base_b)-th tuple, a deterministic map rather than
  // an atomic claim) carries unseated tuples to the next round, and ceil(chunkBound / lanes) rounds
  // always drain the population.
  inline constexpr int kMainMinN = 3;
  template <typename TrackerTraits>
  inline constexpr uint32_t kMainNBins = uint32_t(TrackerTraits::maxHitsOnTrackForFullFit) - uint32_t(kMainMinN) + 1u;

  // Device tables of one round. pRange holds {firstLane, nLanes, tupleBase} per bin; pBlockMap holds
  // {bin, firstLane, endLane} per fused-grid block, with bin == kMainNBins marking an idle block.
  inline constexpr uint32_t kMainRangeStride = 3u;
  inline constexpr uint32_t kMainBlockMapStride = 3u;

  // Fused-grid width. A bin holding n_b lanes needs ceil(n_b / kFitBlock) blocks, and
  //   sum_b ceil(n_b / kFitBlock) <= (sum_b n_b) / kFitBlock + kMainNBins
  //                               <= maxNumberOfConcurrentFits / kFitBlock + kMainNBins,
  // so this width covers EVERY partition the populations can produce. The grid is fixed and
  // host-known; which of its blocks are live, and on which bin, is what the device table decides. A
  // block carries exactly one bin, so the switch over compile-time N never diverges inside a warp.
  template <typename TrackerTraits>
  inline constexpr uint32_t kMainFusedBlocks =
      riemannFit::maxNumberOfConcurrentFits / kFitBlock + kMainNBins<TrackerTraits>;

  // THE ONE DEFINITION of the main ladder's binning, so the range kernel and the fused switch cannot
  // drift apart: it reproduces exactly the (nHitsL, nHitsH) pairs launchBrokenLineKernels launches the
  // per-bin ladder with -- the exact bins N = kMainMinN .. maxHitsOnTrackForFullFit - 1 (rolling_fits
  // is End-EXCLUSIVE), and the top bin N = maxHitsOnTrackForFullFit absorbing the tail
  // [maxHitsOnTrackForFullFit, maxHitsOnTrack - 1].
  template <typename TrackerTraits>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr uint32_t mainBinNHitsL(uint32_t b) {
    return uint32_t(kMainMinN) + b;
  }
  template <typename TrackerTraits>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr uint32_t mainBinNHitsH(uint32_t b) {
    return (b + 1u == kMainNBins<TrackerTraits>) ? (uint32_t(TrackerTraits::maxHitsOnTrack) - 1u)
                                                 : (uint32_t(kMainMinN) + b);
  }

  // Out-of-line boundary of the per-lane fit bodies. RecoTracker/PixelSeeding is built with -Ofast
  // (-ffast-math), so the compiler may reassociate FP and form FMAs per function: two inlined copies of
  // the same source (the per-bin ladder's kernel vs the fused ladder's switch) would produce different
  // doubles, and a bordered-band solve amplifies that into a different chi2. Pinned out of line, both
  // ladders call one compiled function. `noclone` keeps -fipa-cp-clone from specialising a copy on one
  // call site's constants.
#if defined(__CUDACC__)
#define BL_REFIT_NOINLINE __noinline__
#elif defined(__clang__)
#define BL_REFIT_NOINLINE __attribute__((noinline))
#elif defined(__GNUC__)
#define BL_REFIT_NOINLINE __attribute__((noinline, noclone))
#else
#define BL_REFIT_NOINLINE
#endif

  template <int N>
  class Kernel_BLFastFit {
  public:
    // Out-of-line per-lane body of the fast fit (BL_REFIT_NOINLINE, see above): the per-bin ladder's
    // operator() and the fused ladder's switch call this one compiled function. No cross-lane or
    // cross-block state: the lane index only forms addresses.
    ALPAKA_FN_ACC BL_REFIT_NOINLINE static void lane(Acc1D const& acc,
                                                     uint32_t local_idx,
                                                     uint32_t tuple_idx,
                                                     Tuples const* __restrict__ foundNtuplets,
                                                     TupleMultiplicity const* __restrict__ tupleMultiplicity,
                                                     ::reco::TrackingRecHitConstView const& hh,
                                                     ::reco::CAModulesConstView const& cm,
                                                     typename caStructures::tindex_type* __restrict__ ptkids,
                                                     double_st* __restrict__ phits,
                                                     float_st* __restrict__ phits_ge,
                                                     double_st* __restrict__ pfast_fit,
                                                     uint32_t nHitsL,
                                                     uint32_t nHitsH,
                                                     bool hasStubs) {
      constexpr uint32_t hitsInFit = N;
      // get it from the ntuple container (one to one to helix)
      auto tkid = *(tupleMultiplicity->begin(nHitsL) + tuple_idx);
      ALPAKA_ASSERT_ACC(tkid < foundNtuplets->nOnes());

      ptkids[local_idx] = tkid;

      auto nHits = foundNtuplets->size(tkid);

      // multiplicity binning / assertions use the SELECTED hit count (nSel), computed below.
      riemannFit::Map3xNd<N> hits(phits + local_idx);
      riemannFit::Map4d fast_fit(pfast_fit + local_idx);
      riemannFit::Map6xNf<N> hits_ge(phits_ge + local_idx);

      // Prepare data structure
      auto const* hitId = foundNtuplets->begin(tkid);

      // Number of hits the fit uses: kMode filter + same-layer PIXEL overlap dedup (OT stubs never merged),
      // by the SAME caFitHitSel::dedupWalk used in count/fillMultiplicity. hasStubs is the compile-time
      // OT-stubs topology flag; combine it with the runtime stub offset as count/fill do.
      const bool hasStubsRt = hasStubs && (static_cast<int32_t>(hh.offsetStubs()) >= 0);
      uint32_t nSel = caFitHitSel::dedupWalk(foundNtuplets, tkid, hh, hasStubsRt, /*k=*/-1);
      ALPAKA_ASSERT_ACC(nSel >= nHitsL);
      ALPAKA_ASSERT_ACC(nSel <= nHitsH);

      // Uniform sampling: select hitsInFit hits uniformly from the DEDUPED selected hits.
      uint32_t selectedHits[N];
      uint32_t nSelected = 0;
      {
        float incr = std::max(1.f, float(nSel) / float(hitsInFit));
        float fn = 0;
        for (uint32_t i = 0; i < hitsInFit; ++i) {
          int k = int(fn + 0.5f);  // round -> k-th kept hit
          if (hitsInFit - 1 == i)
            k = int(nSel) - 1;  // force last kept hit (max lever arm)
          // map the k-th kept hit to its position j in [0, nHits) via the shared dedup walk
          uint32_t j = caFitHitSel::dedupWalk(foundNtuplets, tkid, hh, hasStubsRt, k);
          ALPAKA_ASSERT_ACC(j < nHits);
          selectedHits[nSelected++] = j;
          fn += incr;
        }
      }
      ALPAKA_ASSERT_ACC(nSelected == hitsInFit);

      for (uint32_t i = 0; i < hitsInFit; ++i) {
        int j = selectedHits[i];
        auto hit = hitId[j];
        float ge[6];

        // Unified per-hit global-error propagation. The "picked" sensor per the
        // one-fit-point-per-stub contract is stored in CAModulesSoA::innerSensorFrame (the pixel-side
        // sensor for PS stubs, the physically-inner sensor for SS stubs, == detFrame for pixels). For SS
        // stubs the global error is built here from the local error through that frame (OTRecHitsSoA has no
        // pre-computed global-error columns).
        auto frame = cm.innerSensorFrame(hh.detectorIndex(hit));
        float xerrFit = hh[hit].xerrLocal();
        // Stub transverse-error calibration (fit input only): scale the local-x VARIANCE by the squared
        // pull width (2S barrel 0.68, PS barrel 0.80, disks 0.92/0.95) so the fit weights match the real
        // resolution. Fit-input only -- track finding is untouched.
        if (hasStubsRt && reco::isStub(hh, int32_t(hit))) {
          const bool is2S = hh[hit].yerrLocal() > 0.1f;                                // strip vs macro-pixel
          const bool isBarrelHit = alpaka::math::abs(acc, hh[hit].zGlobal()) < 118.f;  // OT barrel vs TEDD
          const float f = is2S ? (isBarrelHit ? 0.4624f : 0.8464f)                     // sigma 0.68 / 0.92
                               : (isBarrelHit ? 0.64f : 0.9025f);                      // sigma 0.80 / 0.95
          xerrFit *= f;
        }
        // The strip-length variance (degenerate/unmeasured for a 2S stub) enters the fit as measured.
        float yerrFit = hh[hit].yerrLocal();
        frame.toGlobal(xerrFit, 0, yerrFit, ge);

        // Fill position - for stubs: use the global position of the lower hit that is stored in the hit SoA
        hits.col(i) << hh[hit].xGlobal(), hh[hit].yGlobal(), hh[hit].zGlobal();
        hits_ge.col(i) << ge[0], ge[1], ge[2], ge[3], ge[4], ge[5];
      }
      brokenline::fastFit(acc, hits, fast_fit);

#ifdef BROKENLINE_DEBUG
      // any NaN value should cause the track to be rejected at a later stage
      ALPAKA_ASSERT_ACC(not alpaka::math::isnan(acc, fast_fit(0)));
      ALPAKA_ASSERT_ACC(not alpaka::math::isnan(acc, fast_fit(1)));
      ALPAKA_ASSERT_ACC(not alpaka::math::isnan(acc, fast_fit(2)));
      ALPAKA_ASSERT_ACC(not alpaka::math::isnan(acc, fast_fit(3)));
#endif
    }

    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  Tuples const* __restrict__ foundNtuplets,
                                  TupleMultiplicity const* __restrict__ tupleMultiplicity,
                                  ::reco::TrackingRecHitConstView hh,
                                  ::reco::CAModulesConstView cm,
                                  typename caStructures::tindex_type* __restrict__ ptkids,
                                  double_st* __restrict__ phits,
                                  float_st* __restrict__ phits_ge,
                                  double_st* __restrict__ pfast_fit,
                                  uint32_t nHitsL,
                                  uint32_t nHitsH,
                                  int32_t offset,
                                  bool hasStubs) const {
      constexpr uint32_t hitsInFit = N;
      constexpr auto invalidTkId = std::numeric_limits<typename caStructures::tindex_type>::max();

      ALPAKA_ASSERT_ACC(hitsInFit <= nHitsL);
      ALPAKA_ASSERT_ACC(nHitsL <= nHitsH);
      ALPAKA_ASSERT_ACC(phits);
      ALPAKA_ASSERT_ACC(pfast_fit);
      ALPAKA_ASSERT_ACC(foundNtuplets);
      ALPAKA_ASSERT_ACC(tupleMultiplicity);

      // look in bin for this hit multiplicity
      int totTK = tupleMultiplicity->end(nHitsH) - tupleMultiplicity->begin(nHitsL);
      ALPAKA_ASSERT_ACC(totTK <= int(tupleMultiplicity->size()));
      ALPAKA_ASSERT_ACC(totTK >= 0);

#ifdef BROKENLINE_DEBUG
      if (cms::alpakatools::once_per_grid(acc)) {
        printf("%d total Ntuple\n", tupleMultiplicity->size());
        printf("%d Ntuple of size %d/%d for %d hits to fit\n", totTK, nHitsL, nHitsH, hitsInFit);
      }
#endif
      const auto nt = riemannFit::maxNumberOfConcurrentFits;
      for (auto local_idx : cms::alpakatools::uniform_elements(acc, nt)) {
        auto tuple_idx = local_idx + offset;
        if ((int)tuple_idx >= totTK) {
          ptkids[local_idx] = invalidTkId;
          break;
        }
        Kernel_BLFastFit<N>::lane(acc,
                                  local_idx,
                                  uint32_t(tuple_idx),
                                  foundNtuplets,
                                  tupleMultiplicity,
                                  hh,
                                  cm,
                                  ptkids,
                                  phits,
                                  phits_ge,
                                  pfast_fit,
                                  nHitsL,
                                  nHitsH,
                                  hasStubs);
      }
    }
  };

  // Per-track effective field for the GBL curvature->pT conversion. The fit solves a geometric curvature of
  // the transverse projection, whose bending field is B_bend = Bz - Br*tanLambda*cos(alpha) (see
  // BLBFieldMap.h). The hit-average of B_bend/Bz(0,0) scales the origin scalar, with tanLambda*cos(alpha)
  // = (cx*y - cy*x)/(|R|*r) carrying q^2 = 1, so the correction is charge-free. Null bMap -> origin field.
  template <typename TAcc, typename M3xN, typename V4>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE double blEffectiveBField(
      const TAcc& acc, const M3xN& hits, int n, const V4& fast_fit, double bField, const float* bMap) {
    if (bMap == nullptr)
      return bField;
    const double_st cx = fast_fit(0);
    const double_st cy = fast_fit(1);
    const double_st absR = alpaka::math::abs(acc, fast_fit(2));
    const double_st slopeDen = fast_fit(3) * absR;
    double_st sSum = 0.;
    for (int i = 0; i < n; ++i) {
      const double_st x = hits(0, i);
      const double_st y = hits(1, i);
      const double_st r = alpaka::math::sqrt(acc, x * x + y * y);
      const double_st den = slopeDen * r;
      const double_st tanLambdaCosAlpha = (den != 0.) ? -(cx * y - cy * x) / den : static_cast<double_st>(0.);
      sSum += blBFieldMap::bBendAt(bMap, r, hits(2, i), tanLambdaCosAlpha);
    }
    return bField * (sSum / double(n));
  }

  // Fit-consistent effective bending field: the SAME B_bend samples blEffectiveBField averages, but
  // weighted by the fit's own curvature-information kernel instead of by 1/n. The published momentum is
  // bFieldEff/|curvature|, so bFieldEff must be the B_bend average the fit's curvature estimator actually
  // takes. A linearized curvature estimator gives near-zero weight to the arc ends (where hits fix the
  // chord) and peaks at mid-arc (where the sagitta lives); a 1/n comb would weight crowded pixel nodes the
  // same as mid-arc nodes, making the answer depend on hit placement rather than the field.
  //
  // The fit's functional is free: the GBL normal system eliminates the 1x1 q/p border with a scalar Schur
  // complement, delta(0) = (b(0) - sum_i w[i] b(1+i)) / schur. A residual along the first (bending) offset
  // at measurement node j enters the curvature with coefficient -g/schur, where
  //     g = w[u-1]*P(0,0) + w[u]*P(1,0),   u = 1 + 2j (node's offset index),
  // and the common -1/schur cancels in the ratio. Feeding that estimator the varying-field trajectory
  //     utilde_i = INT_0^{s_i} (s_i - t) b(t) dt,   b = B_bend/Bz(0,0),
  // normalised by the constant-unit case (utilde_i = s_i^2/2):
  //     B_eff = bField * SUM_i g_i utilde_i / SUM_i g_i s_i^2/2 .
  // Computed on the DEVIATION beta = b - 1 (constant field returns bField EXACTLY, and the correction is
  // ~1e-2 of the total, so the deviation form keeps its relative precision):
  //     utilde_i = s_i^2/2 + (s_i J1_i - J2_i),  J1/J2 = running integrals of beta,
  //     B_eff = bField * (1 + SUM_i g_i (s_i J1_i - J2_i) / SUM_i g_i s_i^2/2) .
  // `nodes`/`nNodes` are the solve's node array (measured nodes in hit order). `infl` is the influence
  // vector (see kGblInfluenceOffset). Degenerate input returns `fallback` (the hit-count average).
  template <typename TAcc, typename M3xN, typename V4>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE double blKernelWeightedBField(const TAcc& acc,
                                                               const M3xN& hits,
                                                               int n,
                                                               const V4& fast_fit,
                                                               int qCharge,
                                                               double bField,
                                                               const float* bMap,
                                                               const generalBrokenLine::GblNodeData* nodes,
                                                               int nNodes,
                                                               const double_st* infl,
                                                               double fallback) {
    if (bMap == nullptr)
      return fallback;
    const double_st cx = fast_fit(0);
    const double_st cy = fast_fit(1);
    const double_st cNorm = alpaka::math::sqrt(acc, cx * cx + cy * cy);
    const double_st absR = alpaka::math::abs(acc, fast_fit(2));
    if (!(cNorm > 0.) || !(absR > 0.))
      return fallback;
    // reference direction of the arc-length origin, exactly as prepareGblFitData builds it
    const double_st ex = -fast_fit(2) * cx / cNorm;
    const double_st ey = -fast_fit(2) * cy / cNorm;
    const double_st slopeDen = fast_fit(3) * absR;
    // Running node walk: J1/J2 are the two integrals of the field profile's DEVIATION from unity, from
    // the FIRST measured node up to the current one, over the cells that reach midway to each neighbour.
    double_st s0 = 0., sPrev = 0., bPrev = 0., j1 = 0., j2 = 0., num = 0., den = 0.;
    int iHit = -1;
    for (int j = 0; j < nNodes; ++j) {
      if (!nodes[j].hasMeas)
        continue;
      ++iHit;
      if (iHit >= n)
        break;
      const double_st x = hits(0, iHit);
      const double_st y = hits(1, iHit);
      const double_st dx = x - cx;
      const double_st dy = y - cy;
      double_st s = static_cast<double_st>(qCharge) * fast_fit(2) * alpaka::math::atan2(acc, dx * ey - dy * ex, dx * ex + dy * ey);
      const double_st r = alpaka::math::sqrt(acc, x * x + y * y);
      const double_st denAlpha = slopeDen * r;
      const double_st tanLambdaCosAlpha = (denAlpha != 0.) ? -(cx * y - cy * x) / denAlpha : static_cast<double_st>(0.);
      const double_st bDev = blBFieldMap::bBendAt(bMap, r, hits(2, iHit), tanLambdaCosAlpha) - 1.;
      if (iHit == 0) {
        s0 = s;  // the integration origin: utilde and s^2/2 are then the same functional of the same arc
        s = 0.;
      } else {
        s -= s0;
        const double_st mid = 0.5 * (sPrev + s);  // cell boundary between this node and the previous one
        j1 += bPrev * (mid - sPrev);
        j2 += bPrev * (mid * mid - sPrev * sPrev) * 0.5;
        j1 += bDev * (s - mid);
        j2 += bDev * (s * s - mid * mid) * 0.5;
      }
      const int u = 1 + 2 * j;
      const double_st g = infl[u - 1] * nodes[j].measPrec(0, 0) + infl[u] * nodes[j].measPrec(1, 0);
      num += g * (s * j1 - j2);
      den += g * s * s * 0.5;
      sPrev = s;
      bPrev = bDev;
    }
    if (den == 0.)
      return fallback;
    const double_st scale = 1. + num / den;  // beta == 0 => num == 0 => scale == 1 => exactly bField
    if (alpaka::math::isnan(acc, scale) || alpaka::math::isinf(acc, scale))
      return fallback;
    return bField * scale;
  }
  // The CA main fit: the factorized Blobel circle+line fit, run by every CA iteration of every topology.
  // Every track then gets exactly one General Broken Lines fit in the merger refit, which lives in the
  // Kernel_BLFitPhase* pipeline further down (instantiated only by the refit ladder, Phase2OTStubs, at
  // HelixFit::kRefitStride). Stride = the launch's concurrent-fit count (the fit-buffer lane stride);
  // it defaults to the global riemannFit::stride.
  template <int N, typename TrackerTraits, uint32_t Stride = riemannFit::stride>
  struct Kernel_BLFit {
  public:
    // Out-of-line per-lane body of the fit (BL_REFIT_NOINLINE): the per-bin ladder's operator() below and
    // the fused ladder's switch (Kernel_BLFitFused, through blMainFitLaneOutOfLine) call this one compiled
    // function. It fits lane `local_idx` of the shared stride-wide buffers and writes the one SoA row
    // ptkids[local_idx] names.
    ALPAKA_FN_ACC BL_REFIT_NOINLINE void lane(Acc1D const& acc,
                                              uint32_t local_idx,
                                              TupleMultiplicity const* __restrict__ tupleMultiplicity,
                                              double bField,
                                              OutputSoAView& results_view,
                                              typename caStructures::tindex_type const* __restrict__ ptkids,
                                              double_st* __restrict__ phits,
                                              float_st* __restrict__ phits_ge,
                                              double_st* __restrict__ pfast_fit,
                                              double_st* __restrict__ pscratch) const {
      auto tkid = ptkids[local_idx];

      ALPAKA_ASSERT_ACC(tkid < tupleMultiplicity->capacity());

      riemannFit::Map3xNdS<N, Stride> hits(phits + local_idx);
      riemannFit::Map4dS<Stride> fast_fit(pfast_fit + local_idx);
      riemannFit::Map6xNfS<N, Stride> hits_ge(phits_ge + local_idx);

      // Factorized Blobel fit: circle + line. All O(N) state (the prepared-data vectors, the shared band
      // block and the helper vectors) lives in this lane's slice of pscratch.
      static_assert(int(N) <= int(TrackerTraits::maxHitsOnTrackForFullFit),
                    "legacy-fit scratch quota is sized at maxHitsOnTrackForFullFit");
      // [element][lane] layout: element e of lane local_idx is at pscratch[e*Stride + local_idx], so a
      // warp reads consecutive lanes at consecutive addresses. Stride = this launch's lane count.
      brokenline::PreparedBrokenLineDataMap<N, Stride> data(pscratch + local_idx);
      brokenline::LegacyFitWorkspaceMap<N, Stride> fitWs(
          pscratch + local_idx + std::size_t(brokenline::kPreparedDataDoubles<N>) * std::size_t(Stride));
      brokenline::karimaki_circle_fit circle;
      riemannFit::LineFit line;
      // Per-track effective bending field (the same helper the GBL refit ladder uses): the hit-average
      // of B_bend(r,z)/Bz(0,0), charge-free, returning bField exactly where the map is flat. It carries
      // the |z| falloff of Bz and the endcap radial component, i.e. the forward pT bias of a scalar field.
      // Evaluated before the fit state exists so its scalars are dead by prepareBrokenLineData's O(N) set.
      // fitCorrections_ off, or null bMap_, => scalar bField.
      const double bFieldEff = fitCorrections_ ? blEffectiveBField(acc, hits, int(N), fast_fit, bField, bMap_) : bField;
      // Every field-dependent term of the fast fit reads bFieldEff:
      //   - prepareBrokenLineData/lineFit/circleFit use it only inside multScatt, as the momentum
      //     p = bFieldEff * radius * sqrt(1+slope^2) the Highland variance is divided by;
      //   - copyFromCircle's 1/bFieldEff converts the fitted geometric curvature (and its covariance
      //     row/column, and the (2,2) element quadratically) to q/pT;
      //   - pt is bFieldEff/|curvature|, the same expression Kernel_BLFitPhaseOut publishes.
      brokenline::prepareBrokenLineData(
          acc, hits, fast_fit, bFieldEff, rhoMap_, data, fitWs, fitCorrections_, /*elossGaps=*/fitCorrections_);
      brokenline::lineFit(acc, hits_ge, fast_fit, bFieldEff, data, line, fitWs, fitCorrections_);
      // Ionization energy loss, the per-track scalar curvature growth per unit material column,
      //     elossCurv = kappa_0 * (dE/dX)_eff / p ,   (dE/dX)_eff = dE(X_tot)/X_tot ,
      // with dE() the same Landau law the GBL refit charges per node (elossTypicalColumn, cumulative-column
      // form, exact at the summed thickness). Anchoring on the total column makes the loss exact at the
      // outermost node and low by a few percent in the middle; momentum is the fit's own
      // p = bFieldEff*R*sqrt(1+slope^2). Evaluated after lineFit so its temps never overlap lineFit's set.
      // fitCorrections_ off => 0.0 (circleFit skips every added expression).
      double_st elossCurv = 0.;
      if (fitCorrections_) {
        const double_st slopeK = -static_cast<double_st>(data.qCharge) / fast_fit(3);
        const double_st xTot = data.innerXX0 + fitWs.gapXX0(int(N) - 2);  // beamline -> outermost node [X/X0]
        const double_st pTot =
            alpaka::math::sqrt(acc, riemannFit::sqr(bFieldEff * fast_fit(2)) * (1. + riemannFit::sqr(slopeK)));
        if (xTot > 0. && pTot > 0. && fast_fit(2) > 0.)
          elossCurv = (generalBrokenLine::elossTypicalColumn(acc, pTot, xTot) / xTot) / (pTot * fast_fit(2));
      }
      brokenline::circleFit(acc, hits, hits_ge, fast_fit, bFieldEff, data, circle, fitWs, fitCorrections_, elossCurv);
      reco::copyFromCircle(results_view, circle.par, circle.cov, line.par, line.cov, 1.f / float(bFieldEff), tkid);
      results_view[tkid].pt() = static_cast<float_st>(bFieldEff) / static_cast<float_st>(abs(circle.par(2)));
      results_view[tkid].eta() = alpaka::math::asinh(acc, line.par(0));
      results_view[tkid].chi2() = (circle.chi2 + line.chi2) / (2 * N - 5);
      results_view[tkid].ndof() = int8_t(2 * N - 5);
    }

    // Device pointer to the uploaded Geant4 material density grid (blMaterialMap, kSize floats).
    const float* __restrict__ rhoMap_ = nullptr;
    // Device pointer to the (Bz,Br) r-z field map (blBFieldMap), from the same EventSetup condition the
    // merger GBL refit reads. Consumed only under fitCorrections_: the material/field model is one, so the
    // scattering variance is only correct if its momentum comes from the same field that bent the track.
    // Off, or null, => scalar bField (the upstream algebra).
    const float* __restrict__ bMap_ = nullptr;
    // Fit correctness package (producer parameter useFitCorrections; see the head of BrokenLine.h): thin
    // scatterer per gap, rigid-node guard, Karimaki-Fisher covariance blend, pion 1/beta Highland form,
    // trapezoid material quadrature, full 3x3 blend, per-track effective bending field, and ionization
    // energy-loss offset. Off => upstream algebra. Default true for Phase2OTStubs, false otherwise.
    bool fitCorrections_ = false;

    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  TupleMultiplicity const* __restrict__ tupleMultiplicity,
                                  double bField,
                                  OutputSoAView results_view,
                                  typename caStructures::tindex_type const* __restrict__ ptkids,
                                  double_st* __restrict__ phits,
                                  float_st* __restrict__ phits_ge,
                                  double_st* __restrict__ pfast_fit,
                                  // Per-lane fit scratch: the prepared-data vectors, the shared band block and
                                  // the helper vectors of the factorized fit, held off the kernel stack frame
                                  // (the frame drives the driver's per-thread local-memory reservation).
                                  double_st* __restrict__ pscratch) const {
      ALPAKA_ASSERT_ACC(results_view.pt().data());
      ALPAKA_ASSERT_ACC(results_view.eta().data());
      ALPAKA_ASSERT_ACC(results_view.chi2().data());
      ALPAKA_ASSERT_ACC(pfast_fit);

      constexpr auto invalidTkId = std::numeric_limits<typename caStructures::tindex_type>::max();

      // same as above...
      // look in bin for this hit multiplicity
      const auto nt = Stride;  // lane count = this launch's buffer stride (main: global stride)
      for (auto local_idx : cms::alpakatools::uniform_elements(acc, nt)) {
        if (invalidTkId == ptkids[local_idx])
          break;
        lane(acc, local_idx, tupleMultiplicity, bField, results_view, ptkids, phits, phits_ge, pfast_fit, pscratch);
      }
    }
  };

  // ---------------------------------------------------------------------------------------------
  // THE DYNAMIC PARTITION of the main fit ladder. Single-threaded, device-only (no host crossing): hands
  // lanes out in bin order, writes the block->bin dispatch table (one bin per block, idle tail at
  // kMainNBins). Nothing is memset -- the kernel writes every word it reads.
  // ---------------------------------------------------------------------------------------------
  template <typename TrackerTraits>
  class Kernel_BLMainLaneRanges {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  TupleMultiplicity const* __restrict__ tupleMultiplicity,
                                  uint32_t* __restrict__ pCursor,
                                  uint32_t* __restrict__ pRange,
                                  uint32_t* __restrict__ pBlockMap,
                                  uint32_t laneTotal,
                                  bool firstRound) const {
      if (alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0] != 0)
        return;
      constexpr uint32_t kNBins = kMainNBins<TrackerTraits>;
      uint32_t base = 0u;
      for (uint32_t b = 0; b < kNBins; ++b) {
        // The bin's population, off the SAME container slice the per-bin fast fit's totTK reads.
        const int32_t tot = int32_t(tupleMultiplicity->end(mainBinNHitsH<TrackerTraits>(b)) -
                                    tupleMultiplicity->begin(mainBinNHitsL<TrackerTraits>(b)));
        const uint32_t have = firstRound ? 0u : pCursor[b];  // seated in earlier rounds
        const uint32_t want = (tot > 0 && uint32_t(tot) > have) ? (uint32_t(tot) - have) : 0u;
        const uint32_t room = (base < laneTotal) ? (laneTotal - base) : 0u;
        const uint32_t got = (want < room) ? want : room;
        pRange[kMainRangeStride * b + 0] = base;
        pRange[kMainRangeStride * b + 1] = got;
        pRange[kMainRangeStride * b + 2] = have;  // first tuple of the bin this round seats
        pCursor[b] = have + got;
        base += got;
      }
      uint32_t blk = 0u;
      for (uint32_t b = 0; b < kNBins; ++b) {
        const uint32_t lo = pRange[kMainRangeStride * b + 0];
        const uint32_t hi = lo + pRange[kMainRangeStride * b + 1];
        for (uint32_t l = lo; l < hi; l += kFitBlock) {
          ALPAKA_ASSERT_ACC(blk < kMainFusedBlocks<TrackerTraits>);  // sum_b ceil(n_b/kFitBlock) <= width
          const uint32_t e = (l + kFitBlock < hi) ? (l + kFitBlock) : hi;
          pBlockMap[kMainBlockMapStride * blk + 0] = b;
          pBlockMap[kMainBlockMapStride * blk + 1] = l;
          pBlockMap[kMainBlockMapStride * blk + 2] = e;
          ++blk;
        }
      }
      for (; blk < kMainFusedBlocks<TrackerTraits>; ++blk) {
        pBlockMap[kMainBlockMapStride * blk + 0] = kNBins;  // idle
        pBlockMap[kMainBlockMapStride * blk + 1] = 0u;
        pBlockMap[kMainBlockMapStride * blk + 2] = 0u;
      }
    }
  };

  // N-independent config of one fused main-fit launch: the per-bin kernels' members + N-independent args.
  // One object serves every bin; the trampoline rebuilds the bin's kernel inside the callee's frame.
  struct BLMainFusedCfg {
    const float* __restrict__ rhoMap = nullptr;
    const float* __restrict__ bMap = nullptr;
    bool fitCorrections = false;
    bool hasStubs = false;
  };

  // The fit's out-of-line trampoline: rebuilds the bin's kernel object from the shared config inside the
  // callee's frame and calls the pinned lane body. (The fast fit needs no trampoline -- Kernel_BLFastFit<N>
  // carries no members, so the fused switch calls its static `lane` directly.)
  template <int N, typename TrackerTraits, uint32_t Stride>
  ALPAKA_FN_ACC BL_REFIT_NOINLINE void blMainFitLaneOutOfLine(
      Acc1D const& acc,
      BLMainFusedCfg const& cfg,
      uint32_t lane,
      TupleMultiplicity const* __restrict__ tupleMultiplicity,
      double bField,
      OutputSoAView& results_view,
      typename caStructures::tindex_type const* __restrict__ ptkids,
      double_st* __restrict__ phits,
      float_st* __restrict__ phits_ge,
      double_st* __restrict__ pfast_fit,
      double_st* __restrict__ pscratch) {
    const Kernel_BLFit<N, TrackerTraits, Stride> k{cfg.rhoMap, cfg.bMap, cfg.fitCorrections};
    k.lane(acc, lane, tupleMultiplicity, bField, results_view, ptkids, phits, phits_ge, pfast_fit, pscratch);
  }

  // The fused main-fit kernels. The block index picks the bin (and its compile-time N) out of the device
  // table Kernel_BLMainLaneRanges wrote for this round; the elements of that block are the bin's lanes.
  // One block carries one bin, so the switch over N cannot diverge inside a warp; an unused block returns
  // at once. Bin b runs the same out-of-line Kernel_BLFastFit/BLFit<b+kMainMinN>::lane the per-bin ladder
  // calls; only the lane a tuple lands in differs, and no lane body reads its lane index beyond its own
  // addresses, so the arithmetic and the set of fitted tuples are the same on both ladders.
  template <typename TrackerTraits>
  struct Kernel_BLFastFitFused {
    // The switch dispatches bins 0..7 by hand; kMainNBins is traits-derived. A traits set carrying more
    // bins would pass the idle test, fall to `default:` and lose its tuples silently, so raising
    // maxHitsOnTrackForFullFit past 10 must be a build error.
    static_assert(kMainNBins<TrackerTraits> <= 8u,
                  "the fused main ladder dispatches bins 0..7; add cases when a traits set carries more");
    // No Stride template here (unlike Kernel_BLFitFused): the fast fit uses the default-stride maps
    // (riemannFit::stride == maxNumberOfConcurrentFits), the same lane space the fit kernel asserts on.
    BLMainFusedCfg cfg_;

    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  Tuples const* __restrict__ foundNtuplets,
                                  TupleMultiplicity const* __restrict__ tupleMultiplicity,
                                  ::reco::TrackingRecHitConstView hh,
                                  ::reco::CAModulesConstView cm,
                                  typename caStructures::tindex_type* __restrict__ ptkids,
                                  double_st* __restrict__ phits,
                                  float_st* __restrict__ phits_ge,
                                  double_st* __restrict__ pfast_fit,
                                  const uint32_t* __restrict__ pRange,
                                  const uint32_t* __restrict__ pBlockMap) const {
      // The per-bin kernel's entry assertions (see Kernel_BLFastFit::operator()).
      ALPAKA_ASSERT_ACC(foundNtuplets);
      ALPAKA_ASSERT_ACC(tupleMultiplicity);
      ALPAKA_ASSERT_ACC(phits);
      ALPAKA_ASSERT_ACC(pfast_fit);
      constexpr uint32_t kNBins = kMainNBins<TrackerTraits>;
      constexpr uint32_t kTotal = kMainFusedBlocks<TrackerTraits> * kFitBlock;
      for (uint32_t blk : cms::alpakatools::uniform_groups(acc, kTotal)) {
        const uint32_t bin = pBlockMap[kMainBlockMapStride * blk];
        if (bin >= kNBins)
          continue;  // idle block: this round's partition did not need it
        ALPAKA_ASSERT_ACC(mainBinNHitsL<TrackerTraits>(bin) <= mainBinNHitsH<TrackerTraits>(bin));
        const uint32_t laneLo = pBlockMap[kMainBlockMapStride * blk + 1];
        const uint32_t laneHi = pBlockMap[kMainBlockMapStride * blk + 2];
        // Where this bin starts in the lane space, and which of its tuples this round seats first.
        const uint32_t binBase = pRange[kMainRangeStride * bin];
        const uint32_t tupleBase = pRange[kMainRangeStride * bin + 2];
        for (auto el : cms::alpakatools::uniform_group_elements(acc, blk, kTotal)) {
          const uint32_t local_idx = laneLo + uint32_t(el.local);
          if (local_idx >= laneHi)
            break;
          const uint32_t tuple_idx = tupleBase + (local_idx - binBase);
#define BL_MAIN_FUSED_FAST_CASE(BIN)                                               \
  case (BIN):                                                                      \
    if constexpr (uint32_t(BIN) < kMainNBins<TrackerTraits>) {                     \
      Kernel_BLFastFit<(BIN) + kMainMinN>::lane(acc,                               \
                                                local_idx,                         \
                                                tuple_idx,                         \
                                                foundNtuplets,                     \
                                                tupleMultiplicity,                 \
                                                hh,                                \
                                                cm,                                \
                                                ptkids,                            \
                                                phits,                             \
                                                phits_ge,                          \
                                                pfast_fit,                         \
                                                mainBinNHitsL<TrackerTraits>(BIN), \
                                                mainBinNHitsH<TrackerTraits>(BIN), \
                                                cfg_.hasStubs);                    \
    }                                                                              \
    break
          switch (bin) {
            BL_MAIN_FUSED_FAST_CASE(0);
            BL_MAIN_FUSED_FAST_CASE(1);
            BL_MAIN_FUSED_FAST_CASE(2);
            BL_MAIN_FUSED_FAST_CASE(3);
            BL_MAIN_FUSED_FAST_CASE(4);
            BL_MAIN_FUSED_FAST_CASE(5);
            BL_MAIN_FUSED_FAST_CASE(6);
            BL_MAIN_FUSED_FAST_CASE(7);
            default:
              break;
          }
#undef BL_MAIN_FUSED_FAST_CASE
        }
      }
    }
  };

  template <typename TrackerTraits, uint32_t Stride = riemannFit::stride>
  struct Kernel_BLFitFused {
    static_assert(Stride == riemannFit::maxNumberOfConcurrentFits,
                  "the fused main ladder spans the main fit's lane buffers");
    // Same guard as Kernel_BLFastFitFused: the switch dispatches bins 0..7 by hand.
    static_assert(kMainNBins<TrackerTraits> <= 8u,
                  "the fused main ladder dispatches bins 0..7; add cases when a traits set carries more");
    BLMainFusedCfg cfg_;

    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  TupleMultiplicity const* __restrict__ tupleMultiplicity,
                                  double bField,
                                  OutputSoAView results_view,
                                  typename caStructures::tindex_type const* __restrict__ ptkids,
                                  double_st* __restrict__ phits,
                                  float_st* __restrict__ phits_ge,
                                  double_st* __restrict__ pfast_fit,
                                  double_st* __restrict__ pscratch,
                                  const uint32_t* __restrict__ pBlockMap) const {
      ALPAKA_ASSERT_ACC(results_view.pt().data());
      ALPAKA_ASSERT_ACC(results_view.eta().data());
      ALPAKA_ASSERT_ACC(results_view.chi2().data());
      ALPAKA_ASSERT_ACC(pfast_fit);
      constexpr auto invalidTkId = std::numeric_limits<typename caStructures::tindex_type>::max();
      constexpr uint32_t kNBins = kMainNBins<TrackerTraits>;
      constexpr uint32_t kTotal = kMainFusedBlocks<TrackerTraits> * kFitBlock;
      for (uint32_t blk : cms::alpakatools::uniform_groups(acc, kTotal)) {
        const uint32_t bin = pBlockMap[kMainBlockMapStride * blk];
        if (bin >= kNBins)
          continue;  // idle block
        const uint32_t laneLo = pBlockMap[kMainBlockMapStride * blk + 1];
        const uint32_t laneHi = pBlockMap[kMainBlockMapStride * blk + 2];
        for (auto el : cms::alpakatools::uniform_group_elements(acc, blk, kTotal)) {
          const uint32_t local_idx = laneLo + uint32_t(el.local);
          if (local_idx >= laneHi)
            break;
          // Safety net only: a bin's range IS its exact count and the fused fast fit above filled
          // every lane of it, so the sentinel cannot be reached unless the range and the fill ever
          // disagree.
          if (invalidTkId == ptkids[local_idx])
            continue;
#define BL_MAIN_FUSED_FIT_CASE(BIN)                                                                                     \
  case (BIN):                                                                                                           \
    if constexpr (uint32_t(BIN) < kMainNBins<TrackerTraits>) {                                                          \
      blMainFitLaneOutOfLine<(BIN) + kMainMinN, TrackerTraits, Stride>(                                                 \
          acc, cfg_, local_idx, tupleMultiplicity, bField, results_view, ptkids, phits, phits_ge, pfast_fit, pscratch); \
    }                                                                                                                   \
    break
          switch (bin) {
            BL_MAIN_FUSED_FIT_CASE(0);
            BL_MAIN_FUSED_FIT_CASE(1);
            BL_MAIN_FUSED_FIT_CASE(2);
            BL_MAIN_FUSED_FIT_CASE(3);
            BL_MAIN_FUSED_FIT_CASE(4);
            BL_MAIN_FUSED_FIT_CASE(5);
            BL_MAIN_FUSED_FIT_CASE(6);
            BL_MAIN_FUSED_FIT_CASE(7);
            default:
              break;
          }
#undef BL_MAIN_FUSED_FIT_CASE
        }
      }
    }
  };

  // Phase-buffer lane stride (doubles): the per-lane cross-launch scratch of the Kernel_BLFitPhase*
  // pipeline, indexed by the named slots below. The device allocation strides at this value
  // (see launchBrokenLineKernels / refitExtended).
  //
  // Slots [64..104] hold the ITERATION-INVARIANT tail (values a second linearization would recompute):
  //   [64..64+N-1] matXX0      the Geant4 material march (function of hit positions alone)
  //   [64+N]       innerXX0     function of hit 0; at the END of the window where prepareGblFitData's
  //                              matCached contract reads it (matCached[n]). Window sized for N<=12.
  //   [77] [78]    innerD1/W1   segmentXX0Moments, function of hit 0 alone
  //   [79]         bFieldEff    blEffectiveBField of THIS linearization (reads fast_fit, but invariant
  //                              across the prep/out/outlier phases of one linearization).
  // so no phase indexes the buffer by a bare number.
  constexpr int kBLPhaseJacBack = 0;     // [0..24]  hit0 -> PCA backward jacobian (hits-only node layout)
  constexpr int kBLPhaseUsedInner = 25;  // 1 if the inner-node layout was built for this lane
  constexpr int kBLPhaseQCharge = 26;
  constexpr int kBLPhaseSTrans0 = 27;  // sTransverse(0), the arc length of the first node
  constexpr int kBLPhaseInnerXX0 = 28;
  constexpr int kBLPhaseGChi2 = 29;    // chi2 of the solve
  constexpr int kBLPhaseCorrPca = 30;  // [30..34] the five perigee corrections
  constexpr int kBLPhaseCovPca = 35;   // [35..59] the 5x5 perigee covariance, row-major
  constexpr int kBLPhaseNextRef = 60;  // [60..63] the reference the next linearization starts from
  // The matXX0 window is sized for the largest N any instantiation carries (kRefitMaxN = 12).
  constexpr int kBLPhaseMatMax = 12;
  constexpr int kBLPhaseMatOff = 64;
  static_assert(kBLPhaseNextRef + 4 == kBLPhaseMatOff, "the base layout must end exactly where the tail begins");
  constexpr int kBLPhaseInnerD1 = kBLPhaseMatOff + kBLPhaseMatMax + 1;  // past the widest matCached[n]
  constexpr int kBLPhaseInnerW1 = kBLPhaseInnerD1 + 1;
  constexpr int kBLPhaseBFieldEff = kBLPhaseInnerW1 + 1;
  //   [80 .. 80+N-1]   gapD1   } the per-gap two-thin split of the exact-split material model
  //   [92 .. 92+N-1]   gapW1   } (segmentXX0GapSplit; iteration-invariant, cached like matXX0)
  //   [104]            splitLayout  1 if prepareGblDataSplit built this lane's 2N+1 node chain
  constexpr int kBLPhaseGapD1 = kBLPhaseBFieldEff + 1;
  constexpr int kBLPhaseGapW1 = kBLPhaseGapD1 + kBLPhaseMatMax;
  constexpr int kBLPhaseSplit = kBLPhaseGapW1 + kBLPhaseMatMax;
  constexpr int kBLPhaseDoubles = kBLPhaseSplit + 1;  // 105

  // Ionization-loss enable flag for the GBL node builders (the refit ladder; passed as
  // applyELossCorrection). The size of the charge comes from the Landau laws in GeneralBrokenLine.h
  // (elossMostProbable/elossTypicalColumn), so no magnitude belongs here.
  constexpr bool kApplyELossCorrection = true;

  // ---------------------------------------------------------------------------------------------
  // PHASE-SPLIT GBL fit pipeline. One kernel carrying prep+solve+extract+outlier at once spills a
  // per-thread frame the driver reserves for every max-resident thread; per-phase kernels keep the live
  // set to a few hundred bytes with no spill. Per-fit intermediates cross launches through a per-lane
  // phase buffer (kBLPhaseDoubles doubles, caching-allocator scratch; slot layout above). Every boundary
  // value is a fully-materialized double, so the store/load boundary cannot change any rounding.
  //
  // Phase (i): per-node preparation. Loads the (possibly overridden) linearization reference (before
  // prepareGblFitData, since every prep step depends on it), rebuilds arc-length/material/gnodes, and
  // persists the iteration-invariant scalars + jacBack into the phase buffer.
  template <int N, typename TrackerTraits, uint32_t Stride = riemannFit::stride>
  struct Kernel_BLFitPhasePrep {
  public:
    const float* __restrict__ rhoMap_ = nullptr;
    // Refit iteration 1: take the linearization reference from this lane's phase buffer (written by the
    // iteration-0 extract) instead of the fast fit the lane arrived with.
    bool iterFromPhase_ = false;
    // Normalized (Bz,Br) r-z field map. When set, the momentum feeding the MS weights / dE/dx uses a
    // per-track hit-averaged effective field; when null the scalar bField is used instead.
    const float* __restrict__ bMap_ = nullptr;
    // This launch is a RE-linearization of a fit whose FIRST linearization ran on the SAME lane of the
    // SAME phase buffer, so the iteration-invariant material rows are already there and the Geant4 map
    // march can be skipped. Only ever true where that is structurally guaranteed: the refit ladder's
    // iteration 1 (one scan, two fit iterations, one phase buffer -- see runRefitFusedPrep). When false the
    // material is marched from the map.
    bool matFromPhase_ = false;
    // Take the effective field from the phase buffer instead of averaging B_bend over the hits. Set only on
    // a RE-linearization whose predecessor's solve published a fit-consistent field there: the field the
    // fit is linearized on is then the one the fit's own curvature weights define, and re-deriving the
    // hit-count average here would discard it. When false this phase computes the hit-count average.
    bool fieldFromPhase_ = false;
    // Charge-symmetric corrections package (see the block at the head of BrokenLine.h). Two terms, both
    // charge-odd by construction: on, the node-0 -> PCA step in gblHelixAtPca uses a SIGNED arc and the
    // node prep adds the bending-field PROFILE deterministic offset; off, the arc is unsigned and no
    // profile offset is applied. The second term additionally needs bMap_, so it is inert where the map
    // is null.
    bool chargeSymmetric_ = false;
    // Reference-trajectory corrections package (see the block at the head of BrokenLine.h). On, both node
    // builders seed the node-0 path length from the reference helix, take the arc->azimuth sign that
    // places a measurement-less node from it, and form the field's B_r lambda row (which additionally
    // needs bMap_); off, the path length is seeded 0, the sign is taken unsigned and only the B_z row is
    // formed. The node LAYOUT is the same either way.
    bool trajectoryCorrections_ = false;
    // On, Highland's log is evaluated at the track's TOTAL declared material and the resulting variance
    // apportioned to the gaps by thickness; off, each gap evaluates the log at its own thickness (see
    // prepareGblData in GeneralBrokenLine.h). Forwarded to BOTH node builders.
    bool scatteringLogAtTotal_ = false;
    // On, the typical energy loss is the single-column Landau law at the accumulated thickness and each
    // node is charged its increment of it (elossTypicalColumn in GeneralBrokenLine.h); off, each lump is
    // charged its own Landau MPV. Forwarded to BOTH node builders.
    bool elossCumulative_ = false;

    // One lane of this phase, pinned out of line (BL_REFIT_NOINLINE): the fused kernel's trampoline
    // calls this one compiled function. The invalid-tkid sentinel is not tested here: the caller tests
    // it before dispatching.
    ALPAKA_FN_ACC BL_REFIT_NOINLINE void lane(Acc1D const& acc,
                                              uint32_t local_idx,
                                              double bField,
                                              [[maybe_unused]]
                                              typename caStructures::tindex_type const* __restrict__ ptkids,
                                              double_st* __restrict__ phits,
                                              float_st* __restrict__ phits_ge,
                                              double_st* __restrict__ pfast_fit,
                                              generalBrokenLine::GblNodeData* __restrict__ pgnodes,
                                              double* __restrict__ pphase) const {
      riemannFit::Map3xNdS<N, Stride> hits(phits + local_idx);
      riemannFit::Map4dS<Stride> fast_fit(pfast_fit + local_idx);
      riemannFit::Map6xNfS<N, Stride> hits_ge(phits_ge + local_idx);
      double* phase = pphase + std::size_t(local_idx) * std::size_t(kBLPhaseDoubles);
      // Reference override BEFORE any reference-dependent preparation.
      if (iterFromPhase_) {
        fast_fit(0) = phase[kBLPhaseNextRef + 0];
        fast_fit(1) = phase[kBLPhaseNextRef + 1];
        fast_fit(2) = phase[kBLPhaseNextRef + 2];
        fast_fit(3) = phase[kBLPhaseNextRef + 3];
      }
      // Per-track effective field; with bMap_ null this is the scalar bField.
      const double bFieldEff =
          fieldFromPhase_ ? phase[kBLPhaseBFieldEff] : blEffectiveBField(acc, hits, int(N), fast_fit, bField, bMap_);
      // Publish this linearization's effective field so the out and outlier phases read it instead of
      // re-running the same O(N) two-bilinears-per-hit map loop on the same fast_fit. Written on every
      // prep, so each phase always sees its own iteration's value. When the value CAME from the slot it
      // is already there, and rewriting it would only cost a store.
      if (!fieldFromPhase_)
        phase[kBLPhaseBFieldEff] = bFieldEff;
      static_assert(int(N) <= kBLPhaseMatMax, "the phase buffer's matXX0 window must cover N");
      brokenline::PreparedGblData<N> data;
      // Per-gap two-thin partition of the exact-split material model, written by prepareGblFitData
      // (uninitialized here: it writes every slot before anything reads one).
      double_st gapD1[N], gapW1[N];
      // On a re-linearization the material rows are loaded from the phase buffer (see
      // prepareGblFitData's matCached contract); otherwise they are marched from the map and stored.
      brokenline::prepareGblFitData(acc,
                                    hits,
                                    fast_fit,
                                    bFieldEff,
                                    rhoMap_,
                                    data,
                                    matFromPhase_ ? (phase + kBLPhaseMatOff) : nullptr,
                                    gapD1,
                                    gapW1);
      constexpr int kSlotNodes = generalBrokenLine::kGblSplitNodes<(
          int(N) > int(TrackerTraits::maxHitsOnTrackForFullFit) ? int(N)
                                                                : int(TrackerTraits::maxHitsOnTrackForFullFit))>;
      generalBrokenLine::GblNodeData* gnodes = pgnodes + std::size_t(local_idx) * std::size_t(kSlotNodes);
      double_st innerD1 = 0., innerW1 = 0.;
      if (matFromPhase_) {
        // The upstream two-thin split is a function of hit 0 alone, so it rides the same cache.
        // Loaded unconditionally: the iteration-0 prep left 0/0 here whenever innerXX0 was not
        // positive, which is exactly what the else branch would produce.
        innerD1 = phase[kBLPhaseInnerD1];
        innerW1 = phase[kBLPhaseInnerW1];
      } else {
        // one quadrature rule for the whole trajectory: the beamline->hit0 moments come from the
        // same trapezoid march as the gaps (its W is the innerXX0 already stored).
        const double_st rHit0 = alpaka::math::sqrt(acc, hits(0, 0) * hits(0, 0) + hits(1, 0) * hits(1, 0));
        brokenline::segmentXX0GapSplit(acc, rhoMap_, 0., 0., rHit0, hits(2, 0), innerD1, innerW1);
      }
      if (matFromPhase_) {
        for (int i = 0; i < int(N); ++i) {
          gapD1[i] = phase[kBLPhaseGapD1 + i];
          gapW1[i] = phase[kBLPhaseGapW1 + i];
        }
      }
      if (!matFromPhase_) {  // publish the invariant rows for the next linearization
        for (int i = 0; i < int(N); ++i)
          phase[kBLPhaseMatOff + i] = data.matXX0(i);
        phase[kBLPhaseMatOff + int(N)] = data.innerXX0;  // matCached[n], see the layout comment
        phase[kBLPhaseInnerD1] = innerD1;
        phase[kBLPhaseInnerW1] = innerW1;
        for (int i = 0; i < int(N); ++i) {
          phase[kBLPhaseGapD1 + i] = gapD1[i];
          phase[kBLPhaseGapW1 + i] = gapW1[i];
        }
      }
      // hit0->PCA backward transport: only the hits-only node layout uses it, so it is set by
      // prepareGblData; identity here keeps the phase buffer defined when a layout that extracts at
      // node 0 (the inner-node layout, and the split) is built instead.
      generalBrokenLine::Matrix5d jacBack = generalBrokenLine::Matrix5d::Identity();
      bool usedInner = false;
      bool usedSplit = false;
      if constexpr (int(N) >= 3) {
        // the 2N+1 layout. It refuses a track whose gaps cannot carry an interior scatterer at all;
        // that track then takes the arrival-node layout below, which is well posed for any gap.
        usedSplit = generalBrokenLine::prepareGblDataSplit<Acc1D, N>(acc,
                                                                     hits,
                                                                     hits_ge,
                                                                     fast_fit,
                                                                     bFieldEff,
                                                                     data.qCharge,
                                                                     data.sTransverse,
                                                                     data.sTotal,
                                                                     data.matXX0,
                                                                     gapD1,
                                                                     gapW1,
                                                                     data.innerXX0,
                                                                     innerD1,
                                                                     innerW1,
                                                                     gnodes,
                                                                     /*msScale=*/1.0,
                                                                     kApplyELossCorrection,
                                                                     chargeSymmetric_ ? bMap_ : nullptr,
                                                                     bField,
                                                                     trajectoryCorrections_,
                                                                     scatteringLogAtTotal_,
                                                                     elossCumulative_);
      }
      if (!usedSplit)
        generalBrokenLine::prepareGblData<Acc1D, N>(acc,
                                                    hits,
                                                    hits_ge,
                                                    fast_fit,
                                                    bFieldEff,
                                                    data.qCharge,
                                                    data.sTransverse,
                                                    data.sTotal,
                                                    data.matXX0,
                                                    data.innerXX0,
                                                    gnodes,
                                                    /*msScale=*/1.0,
                                                    kApplyELossCorrection,
                                                    &jacBack,
                                                    innerD1,
                                                    innerW1,
                                                    &usedInner,
                                                    chargeSymmetric_ ? bMap_ : nullptr,
                                                    bField,
                                                    trajectoryCorrections_,
                                                    scatteringLogAtTotal_,
                                                    elossCumulative_);
#ifdef BL_LAYER_DUMP
      // BL_LAYER_DUMP: fit-input trace, one block per track per linearization -- the reference fast fit,
      // the field, the charge and the material this pass was built on, plus every hit and its ge error
      // row. The last block emitted for a tkid is the final linearization, whose solve produces the
      // published state. Run single-threaded on the serial backend so a track's TRK/HIT lines stay
      // contiguous in stdout.
      const auto tkid = ptkids[local_idx];  // the dump is this phase's only consumer of the tuple id
      printf("BLDUMP_TRK %u N %d bfield %.17g ff %.17g %.17g %.17g %.17g q %d innerXX0 %.17g\n",
             (unsigned)tkid,
             int(N),
             bField,
             fast_fit(0),
             fast_fit(1),
             fast_fit(2),
             fast_fit(3),
             data.qCharge,
             static_cast<double_st>(data.innerXX0));
      for (int i = 0; i < int(N); ++i)
        printf("BLDUMP_HIT %u %d %.17g %.17g %.17g %.9g %.9g %.9g %.9g %.9g %.9g %.17g\n",
               (unsigned)tkid,
               i,
               hits(0, i),
               hits(1, i),
               hits(2, i),
               hits_ge(0, i),
               hits_ge(1, i),
               hits_ge(2, i),
               hits_ge(3, i),
               hits_ge(4, i),
               hits_ge(5, i),
               i < int(N) - 1 ? static_cast<double_st>(data.matXX0(i)) : 0.0);
#endif
      phase[kBLPhaseSplit] = usedSplit ? 1. : 0.;
      for (int r = 0; r < 5; ++r)
        for (int c = 0; c < 5; ++c)
          phase[kBLPhaseJacBack + 5 * r + c] = jacBack(r, c);
      phase[kBLPhaseUsedInner] = usedInner ? 1. : 0.;
      phase[kBLPhaseQCharge] = static_cast<double_st>(data.qCharge);
      phase[kBLPhaseSTrans0] = static_cast<double_st>(data.sTransverse(0));
      phase[kBLPhaseInnerXX0] = static_cast<double_st>(data.innerXX0);
    }
  };

  // Phase (ii): bordered-band solve. gblFitPca over the prepared nodes; persists the curvilinear
  // corrections/covariance at the PCA + the native chi2 (and, on the final linearization, the
  // smoothed-offset pulls into the scratch tail for the outlier phase).
  template <int N, typename TrackerTraits, uint32_t Stride = riemannFit::stride>
  struct Kernel_BLFitPhaseSolve {
  public:
    const bool outlierReject_ = true;
    // True on the fit's LAST linearization (refit iteration 1): request the pulls
    // the outlier phase consumes.
    const bool finalIter_ = true;
    // Normalized (Bz,Br) r-z field map, forwarded only so this phase can re-sample B_bend at the nodes;
    // when null the effective field in the phase buffer is left alone.
    const float* __restrict__ bMap_ = nullptr;
    // On, replace the hit-count-averaged effective field left in the phase buffer by the fit-consistent
    // one this solve's own influence vector defines (see blKernelWeightedBField); off, this phase writes
    // no field and the hit-count average stands.
    bool fieldKernelWeights_ = false;
    // ONE LANE of this phase, pinned OUT OF LINE (BL_REFIT_NOINLINE): the fused kernel's trampoline calls
    // THIS function.
    ALPAKA_FN_ACC BL_REFIT_NOINLINE void lane(Acc1D const& acc,
                                              uint32_t local_idx,
                                              double bField,
                                              double_st* __restrict__ phits,
                                              double_st* __restrict__ pfast_fit,
                                              generalBrokenLine::GblNodeData* __restrict__ pgnodes,
                                              double_st* __restrict__ pscratch,
                                              double* __restrict__ pphase) const {

      const int pfast_fit_digits_in = pfast_fit[0].nb_significant_digit();

      riemannFit::Map4dS<Stride> fast_fit(pfast_fit + local_idx);
      constexpr int kSlotNodes = generalBrokenLine::kGblSplitNodes<(
          int(N) > int(TrackerTraits::maxHitsOnTrackForFullFit) ? int(N)
                                                                : int(TrackerTraits::maxHitsOnTrackForFullFit))>;
      generalBrokenLine::GblNodeData* gnodes = pgnodes + std::size_t(local_idx) * std::size_t(kSlotNodes);
      // sized for the widest node chain this lane can build: the exact split's 2N+1 nodes when the
      // material model is on, the arrival-node layout's N+2 otherwise. One stride for both, so the
      // solve's band region and the outlier phase's pull region never overlap in either layout.
      constexpr int kSplitN = generalBrokenLine::kGblSplitNodes<N> - 1;
      constexpr int kBandDoubles = generalBrokenLine::kGblScratchDoubles<kSplitN>;
      // Scratch overlay: gFullDelta lives in the head of the band region (= Mb); gNodeVar stays
      // outside it. See GeneralBrokenLine.h for the lifetime proof.
      constexpr int kNodeVarDoubles = 3 * (kSplitN + 1);
      constexpr int kPullsDoubles = kNodeVarDoubles;
      constexpr int kScratchStride = kBandDoubles + kPullsDoubles;
      // structural guard: the N-1 branch has the smallest Mb of the three layouts.
      static_assert((2 * kSplitN + 3) <= 2 * int(N) * (generalBrokenLine::kGblBand + 1),
                    "scratch overlay: fullDelta does not fit inside Mb");
      double_st* gblScratch = pscratch + std::size_t(local_idx) * std::size_t(kScratchStride);
      double_st* gFullDelta = gblScratch;
      double_st* gNodeVar = gblScratch + kBandDoubles;
      double* phase = pphase + std::size_t(local_idx) * std::size_t(kBLPhaseDoubles);
      const bool usedInner = phase[kBLPhaseUsedInner] != 0.;
      const bool usedSplit = phase[kBLPhaseSplit] != 0.;
      const int qCharge = int(phase[kBLPhaseQCharge]);
      const bool wantPulls = outlierReject_ && finalIter_ && N >= 5;
      generalBrokenLine::Vector5d corrPca, gcorr;
      generalBrokenLine::Matrix5d covPca, gcov;
      double_st gchi2 = 0.;
      if (usedSplit) {
        // 2N+1 nodes; like the inner-node layout, node 0 IS the PCA, so the extraction is exact.
        covPca = generalBrokenLine::gblFitPca<Acc1D, kSplitN>(acc,
                                                              gnodes,
                                                              gblScratch,
                                                              &corrPca,
                                                              wantPulls ? gFullDelta : nullptr,
                                                              &gchi2,
                                                              wantPulls ? gNodeVar : nullptr,
                                                              /*fullDeltaInScratch=*/true);
      } else if (usedInner) {
        covPca = generalBrokenLine::gblFitPca<Acc1D, N + 1>(acc,
                                                            gnodes,
                                                            gblScratch,
                                                            &corrPca,
                                                            wantPulls ? gFullDelta : nullptr,
                                                            &gchi2,
                                                            wantPulls ? gNodeVar : nullptr,
                                                            /*fullDeltaInScratch=*/true);
      } else {
        const double_st th2Inner = gnodes[1].hasScat ? 1.0 / gnodes[1].scatPrec(0, 0) : static_cast<double_st>(0.0);
        gnodes[1].hasScat = false;
        gcov = generalBrokenLine::gblFitPca<Acc1D, N - 1>(acc,
                                                          gnodes + 1,
                                                          gblScratch,
                                                          &gcorr,
                                                          wantPulls ? gFullDelta : nullptr,
                                                          &gchi2,
                                                          wantPulls ? gNodeVar : nullptr,
                                                          /*fullDeltaInScratch=*/true);
        generalBrokenLine::Matrix5d jacBack;
        for (int r = 0; r < 5; ++r)
          for (int c = 0; c < 5; ++c)
            jacBack(r, c) = phase[kBLPhaseJacBack + 5 * r + c];
        const double_st slopeQ = -static_cast<double_st>(qCharge) / fast_fit(3);
        gcov(1, 1) += th2Inner;
        gcov(2, 2) += th2Inner * (1.0 + slopeQ * slopeQ);
        corrPca = jacBack * gcorr;
        covPca = jacBack * gcov * jacBack.transpose();
      }
      phase[kBLPhaseGChi2] = gchi2;
      for (int a = 0; a < 5; ++a)
        phase[kBLPhaseCorrPca + a] = corrPca(a);
      for (int r = 0; r < 5; ++r)
        for (int c = 0; c < 5; ++c)
          phase[kBLPhaseCovPca + 5 * r + c] = covPca(r, c);
      // Fit-consistent conversion field. The influence vector this solve just built is the only place the
      // fit's own curvature weights exist, so the effective field is re-derived here and republished into
      // the slot the extraction and outlier phases read; the prep of the NEXT linearization picks it up
      // from the same slot. The solve's own arithmetic is untouched -- this reads w and the node
      // precisions after the fact and writes one double.
      if (fieldKernelWeights_ && bMap_ != nullptr) {
        constexpr int kInflInner = generalBrokenLine::kGblInfluenceOffset<N + 1>;
        constexpr int kInflOuter = generalBrokenLine::kGblInfluenceOffset<N - 1>;
        constexpr int kInflSplit = generalBrokenLine::kGblInfluenceOffset<kSplitN>;
        const double_st* infl = gblScratch + (usedSplit ? kInflSplit : (usedInner ? kInflInner : kInflOuter));
        const generalBrokenLine::GblNodeData* fitNodes = (usedSplit || usedInner) ? gnodes : gnodes + 1;
        const int nFitNodes = usedSplit ? (kSplitN + 1) : (usedInner ? (int(N) + 2) : int(N));
        riemannFit::Map3xNdS<N, Stride> hits(phits + local_idx);
        phase[kBLPhaseBFieldEff] = blKernelWeightedBField(
            acc, hits, int(N), fast_fit, qCharge, bField, bMap_, fitNodes, nFitNodes, infl, phase[kBLPhaseBFieldEff]);
      }

      std::cout << "-- BrokenLineFitKernels.h | Kernel_BLFitPhaseSolve --" << std::endl;
      print_cadna_metric("fast_fit", pfast_fit[0], pfast_fit_digits_in);
    }
  };

  // Phase (iii): extraction + output. Perigee transform at the PCA (gblHelixAtPca), the
  // re-linearization reference hand-off into the phase buffer, and the SoA writeback (pre-outlier
  // values; the outlier phase conditionally overwrites).
  template <int N, typename TrackerTraits, uint32_t Stride = riemannFit::stride>
  struct Kernel_BLFitPhaseOut {
  public:
    // Normalized (Bz,Br) r-z field map. When set, the curvature->pT conversion + the PCA perigee use a
    // per-track hit-averaged effective field; when null they use the scalar bField.
    const float* __restrict__ bMap_ = nullptr;
    // Read this linearization's effective field from the phase buffer instead of recomputing it (the prep
    // of the SAME iteration wrote it there). Only set where the prep is structurally guaranteed to have
    // run on this lane in this iteration: the refit ladder.
    bool bFieldFromPhase_ = false;
    // Charge-symmetric corrections package (see Kernel_BLFitPhasePrep::chargeSymmetric_).
    bool chargeSymmetric_ = false;

    // One lane of this phase, pinned out of line (BL_REFIT_NOINLINE): the fused kernel's trampoline calls it.
    ALPAKA_FN_ACC BL_REFIT_NOINLINE void lane(Acc1D const& acc,
                                              uint32_t local_idx,
                                              double bField,
                                              OutputSoAView results_view,
                                              typename caStructures::tindex_type const* __restrict__ ptkids,
                                              double_st* __restrict__ phits,
                                              double_st* __restrict__ pfast_fit,
                                              double* __restrict__ pphase) const {
      auto tkid = ptkids[local_idx];
      riemannFit::Map3xNdS<N, Stride> hits(phits + local_idx);
      riemannFit::Map4dS<Stride> fast_fit(pfast_fit + local_idx);
      double* phase = pphase + std::size_t(local_idx) * std::size_t(kBLPhaseDoubles);
      const int qCharge = int(phase[kBLPhaseQCharge]);
      const double_st sTrans0 = phase[kBLPhaseSTrans0];
      const double_st gchi2 = phase[kBLPhaseGChi2];
      // When bFieldFromPhase_ is set, THIS linearization's prep already evaluated blEffectiveBField on
      // exactly this fast_fit and these hits (neither has been touched since) and published the double
      // into the phase buffer, so re-running the O(N) two-bilinears-per-hit map loop here can only
      // reproduce it. When it is false the field is recomputed here; with bMap_ null that call is a
      // single early return anyway.
      const double bFieldEff =
          bFieldFromPhase_ ? phase[kBLPhaseBFieldEff] : blEffectiveBField(acc, hits, int(N), fast_fit, bField, bMap_);
      generalBrokenLine::Vector5d corrPca;
      generalBrokenLine::Matrix5d covPca;
      for (int a = 0; a < 5; ++a)
        corrPca(a) = phase[kBLPhaseCorrPca + a];
      for (int r = 0; r < 5; ++r)
        for (int c = 0; c < 5; ++c)
          covPca(r, c) = phase[kBLPhaseCovPca + 5 * r + c];
      generalBrokenLine::Vector5d hp;
      generalBrokenLine::Matrix5d hc;
      Eigen::Vector4d nextRef;
      generalBrokenLine::gblHelixAtPca(
          acc, fast_fit, qCharge, bFieldEff, sTrans0, hits(2, 0), corrPca, covPca, hp, hc, &nextRef, chargeSymmetric_);
      phase[kBLPhaseNextRef + 0] = nextRef(0);
      phase[kBLPhaseNextRef + 1] = nextRef(1);
      phase[kBLPhaseNextRef + 2] = nextRef(2);
      phase[kBLPhaseNextRef + 3] = nextRef(3);
      const float ptVal = float(bFieldEff / alpaka::math::abs(acc, hp(2)));
      hp(2) /= bFieldEff;
      for (int a = 0; a < 5; ++a) {
        hc(2, a) /= bFieldEff;
        hc(a, 2) /= bFieldEff;
      }
      reco::copyFromDense(results_view, hp, hc, tkid);
      results_view[tkid].pt() = ptVal;
      results_view[tkid].eta() = alpaka::math::asinh(acc, hp(3));
      const int ndof = 2 * N - 5;  // no outlier drop yet (the outlier phase overwrites on a drop)
      results_view[tkid].chi2() = float(gchi2 / (ndof > 0 ? ndof : 1));
      results_view[tkid].ndof() = int8_t(ndof > 0 ? ndof : 1);
#ifdef BL_LAYER_DUMP
      // BL_LAYER_DUMP: the fitted state (SoA convention: hp(2) = 1/pt) plus the native chi2, in the same
      // stdout stream as the input blocks above.
      printf("BLDUMP_FIT %u hp %.17g %.17g %.17g %.17g %.17g chi2 %.17g\n",
             (unsigned)tkid,
             hp(0),
             hp(1),
             hp(2),
             hp(3),
             hp(4),
             gchi2);
#endif
    }
  };

  // Phase (iv): outlier rejection (final linearization only, N >= 5; the launcher skips the
  // launch otherwise). Smoothed-residual pulls from the solve's gFullDelta/gNodeVar, drop the
  // single worst hit above the cut, re-solve ONCE and overwrite the SoA output.
  template <int N, typename TrackerTraits, uint32_t Stride = riemannFit::stride>
  struct Kernel_BLFitPhaseOutlier {
  public:
    const bool outlierReject_ = true;
    // When BOTH pointers are non-null (the merger refit, when the fit-rejected hit is to be removed from
    // the emitted list) and a hit is dropped, name the dropped hit into dropHitId_[tkid] using the
    // per-lane fitHitId_ table. Both null wherever the caller left that compaction off, and nothing is written.
    const uint32_t* __restrict__ fitHitId_ = nullptr;
    uint32_t* __restrict__ dropHitId_ = nullptr;
    // Core-protected outlier: when coreProtect_ and fitHitIsCore_ are set, the worst-pull scan skips
    // ORIGINAL PIXEL-CORE nodes (fitHitIsCore_[lane*kRefitFitIdQuota + i] != 0), so the drop can only ever land on an
    // appended extra. With coreProtect_ false / fitHitIsCore_ null the scan considers every node.
    const uint8_t* __restrict__ fitHitIsCore_ = nullptr;
    bool coreProtect_ = false;
    // Core-protect FIRE COUNTER (optional per-event atomic): incremented once per lane where
    // core-protection PREVENTED a drop. Diagnostic only; never changes the drop decision. Null => off.
    uint32_t* __restrict__ pCoreProtectFires_ = nullptr;
    // Normalized (Bz,Br) r-z field map. When set, the re-solve's curvature->pT conversion + PCA perigee
    // use a per-track hit-averaged effective field; when null (as on the CA path) they use the scalar
    // bField.
    const float* __restrict__ bMap_ = nullptr;
    // See Kernel_BLFitPhaseOut::bFieldFromPhase_. The outlier phase runs after the FINAL out phase, so
    // the value in the buffer is the final linearization's -- exactly what its own recompute would yield
    // (fast_fit is not touched after the final prep).
    bool bFieldFromPhase_ = false;
    // Charge-symmetric corrections package (see Kernel_BLFitPhasePrep::chargeSymmetric_).
    bool chargeSymmetric_ = false;
    // One lane of this phase, pinned out of line (BL_REFIT_NOINLINE): the fused kernel's trampoline
    // calls this one compiled function. The invalid-tkid sentinel is not tested here: the caller tests
    // it before dispatching.
    ALPAKA_FN_ACC BL_REFIT_NOINLINE void lane(Acc1D const& acc,
                                              uint32_t local_idx,
                                              double bField,
                                              OutputSoAView results_view,
                                              typename caStructures::tindex_type const* __restrict__ ptkids,
                                              double_st* __restrict__ phits,
                                              double_st* __restrict__ pfast_fit,
                                              generalBrokenLine::GblNodeData* __restrict__ pgnodes,
                                              double_st* __restrict__ pscratch,
                                              double* __restrict__ pphase) const {
      auto tkid = ptkids[local_idx];
      riemannFit::Map3xNdS<N, Stride> hits(phits + local_idx);
      riemannFit::Map4dS<Stride> fast_fit(pfast_fit + local_idx);
      constexpr int kSlotNodes = generalBrokenLine::kGblSplitNodes<(
          int(N) > int(TrackerTraits::maxHitsOnTrackForFullFit) ? int(N)
                                                                : int(TrackerTraits::maxHitsOnTrackForFullFit))>;
      generalBrokenLine::GblNodeData* gnodes = pgnodes + std::size_t(local_idx) * std::size_t(kSlotNodes);
      // sized for the widest node chain this lane can build: the exact split's 2N+1 nodes when the
      // material model is on, the arrival-node layout's N+2 otherwise. One stride for both, so the
      // solve's band region and the outlier phase's pull region never overlap in either layout.
      constexpr int kSplitN = generalBrokenLine::kGblSplitNodes<N> - 1;
      constexpr int kBandDoubles = generalBrokenLine::kGblScratchDoubles<kSplitN>;
      // Scratch overlay: gFullDelta lives in the head of the band region (= Mb); gNodeVar stays
      // outside it. See GeneralBrokenLine.h for the lifetime proof.
      constexpr int kNodeVarDoubles = 3 * (kSplitN + 1);
      constexpr int kPullsDoubles = kNodeVarDoubles;
      constexpr int kScratchStride = kBandDoubles + kPullsDoubles;
      // structural guard: the N-1 branch has the smallest Mb of the three layouts.
      static_assert((2 * kSplitN + 3) <= 2 * int(N) * (generalBrokenLine::kGblBand + 1),
                    "scratch overlay: fullDelta does not fit inside Mb");
      double_st* gblScratch = pscratch + std::size_t(local_idx) * std::size_t(kScratchStride);
      double_st* gFullDelta = gblScratch;
      double_st* gNodeVar = gblScratch + kBandDoubles;
      double* phase = pphase + std::size_t(local_idx) * std::size_t(kBLPhaseDoubles);
      const bool usedInner = phase[kBLPhaseUsedInner] != 0.;
      const int qCharge = int(phase[kBLPhaseQCharge]);
      const double_st sTrans0 = phase[kBLPhaseSTrans0];
      const double_st innerXX0 = phase[kBLPhaseInnerXX0];
      // This linearization's effective field (see Kernel_BLFitPhaseOut::lane for the phase-buffer read).
      const double bFieldEff =
          bFieldFromPhase_ ? phase[kBLPhaseBFieldEff] : blEffectiveBField(acc, hits, int(N), fast_fit, bField, bMap_);
      // The hits-only layout carrying upstream inner material cannot be re-solved faithfully, so it is
      // excluded here; the launcher only runs this phase after the fit's final linearization.
      const bool canResolve = outlierReject_ && (phase[kBLPhaseSplit] != 0. || usedInner || !(innerXX0 > 0.));
      if (!canResolve)
        return;
      const bool usedSplit = phase[kBLPhaseSplit] != 0.;
      generalBrokenLine::GblNodeData* fitNodes = (usedSplit || usedInner) ? gnodes : gnodes + 1;
      const int nFitNodes = usedSplit ? (kSplitN + 1) : (usedInner ? (N + 2) : N);
      auto uIdxOf = [](int k) { return 1 + 2 * k; };
      double_st worst = 0.;
      int worstNode = -1;
      // Fit-hit index of the current measurement node (# measured nodes before k), used to look up
      // its core flag. Advances only on measured nodes -- same mapping the drop-id emit below uses.
      int diScan = 0;
      const bool coreProtectOn = coreProtect_ && fitHitIsCore_ != nullptr;
      // Fire-counter shadow: only computed when the counter is wired AND protection is on. Tracks
      // the worst pull among PROTECTED (core) nodes so that, after the scan, the stage can tell whether
      // an unprotected scan would have dropped a core node.
      const bool countCore = coreProtectOn && pCoreProtectFires_ != nullptr;
      double_st worstCore = 0.;
      for (int k = 0; k < nFitNodes; ++k) {
        if (!fitNodes[k].hasMeas)
          continue;
        const int diThis = diScan++;
        const bool isCoreNode =
            coreProtectOn &&
            fitHitIsCore_[std::size_t(local_idx) * std::size_t(kRefitFitIdQuota) + std::size_t(diThis)] != 0;
        // A core node is scanned but never becomes a drop CANDIDATE: it only feeds the worstCore
        // shadow, which decides whether the stage abstains this round (see the decision below).
        const double_st ru = fitNodes[k].measResidual(0) - gFullDelta[uIdxOf(k)];
        const double_st rv = fitNodes[k].measResidual(1) - gFullDelta[uIdxOf(k) + 1];
        const generalBrokenLine::Matrix2d V = generalBrokenLine::inv2(fitNodes[k].measPrec);
        const double_st s00 = V(0, 0) - gNodeVar[3 * k];
        const double_st s01 = V(0, 1) - gNodeVar[3 * k + 1];
        const double_st s11 = V(1, 1) - gNodeVar[3 * k + 2];
        const double_st det = s00 * s11 - s01 * s01;
        if (s00 <= 0. || s11 <= 0. || det <= 0.)
          continue;  // numerically non-PD residual covariance: no reliable pull
        const double_st pull2 = (ru * (s11 * ru - s01 * rv) + rv * (s00 * rv - s01 * ru)) / det;
        if (isCoreNode) {
          if (pull2 > worstCore)  // shadow only; a core node is still never a real drop candidate
            worstCore = pull2;
          continue;
        }
        if (pull2 > worst) {
          worst = pull2;
          worstNode = k;
        }
      }
      constexpr double kOutlierChi2Cut = 13.8;  // ~99.9% of the chi2 distribution for 2 dof
      // Core-protect PREVENTED a drop iff, absent protection, a core node would have been the
      // outlier (its pull exceeds both the cut and every non-core pull). Count once per such lane.
      if (countCore && worstCore > kOutlierChi2Cut && worstCore > worst)
        alpaka::atomicAdd(acc, pCoreProtectFires_, 1u, alpaka::hierarchy::Blocks{});
      // Abstain when the largest-pull measured node is a protected core node: the evidence points at a
      // hit the stage may not delete, and deleting the next-worst instead would be a different hypothesis
      // that degrades the curvature resolution. The cut is not consulted: when worstCore <= worst the rule
      // is inactive anyway, and when worstCore > worst the non-core candidate is the one being declined.
      const bool abstain = coreProtectOn && worstCore > worst;
      if (!abstain && worstNode >= 0 && worst > kOutlierChi2Cut) {
        fitNodes[worstNode].hasMeas = false;
        // The dropped measurement node maps to fit hit i = (# of measured nodes before it),
        // and fitHitId_[lane*kRefitFitIdQuota + i] is that hit's raw id. Publish it so the merger drops it from the
        // emitted hit list (nHits -> nMeasFit). Guarded on both pointers, so a caller that left them null
        // gets no write.
        if (fitHitId_ != nullptr && dropHitId_ != nullptr) {
          int di = 0;
          for (int k2 = 0; k2 < worstNode; ++k2)
            if (fitNodes[k2].hasMeas)
              ++di;
          dropHitId_[tkid] = fitHitId_[std::size_t(local_idx) * std::size_t(kRefitFitIdQuota) + std::size_t(di)];
        }
        double_st gchi2 = 0.;
        generalBrokenLine::Vector5d corrPca2;
        generalBrokenLine::Matrix5d covPca2;
        if (usedSplit) {
          covPca2 = generalBrokenLine::gblFitPca<Acc1D, kSplitN>(acc, gnodes, gblScratch, &corrPca2, nullptr, &gchi2);
        } else if (usedInner) {
          covPca2 = generalBrokenLine::gblFitPca<Acc1D, N + 1>(acc, gnodes, gblScratch, &corrPca2, nullptr, &gchi2);
        } else {  // canResolve guarantees no upstream re-add is needed here
          covPca2 = generalBrokenLine::gblFitPca<Acc1D, N - 1>(acc, gnodes + 1, gblScratch, &corrPca2, nullptr, &gchi2);
          generalBrokenLine::Matrix5d jacBack;
          for (int r = 0; r < 5; ++r)
            for (int c = 0; c < 5; ++c)
              jacBack(r, c) = phase[kBLPhaseJacBack + 5 * r + c];
          corrPca2 = jacBack * corrPca2;
          covPca2 = jacBack * covPca2 * jacBack.transpose();
        }
        generalBrokenLine::Vector5d hp;
        generalBrokenLine::Matrix5d hc;
        generalBrokenLine::gblHelixAtPca(
            acc, fast_fit, qCharge, bFieldEff, sTrans0, hits(2, 0), corrPca2, covPca2, hp, hc, nullptr, chargeSymmetric_);
        const float ptVal = float(bFieldEff / alpaka::math::abs(acc, hp(2)));
        hp(2) /= bFieldEff;
        for (int a = 0; a < 5; ++a) {
          hc(2, a) /= bFieldEff;
          hc(a, 2) /= bFieldEff;
        }
        reco::copyFromDense(results_view, hp, hc, tkid);
        results_view[tkid].pt() = ptVal;
        results_view[tkid].eta() = alpaka::math::asinh(acc, hp(3));
        const int ndof = 2 * (N - 1) - 5;  // one measurement dropped
        results_view[tkid].chi2() = float(gchi2 / (ndof > 0 ? ndof : 1));
        results_view[tkid].ndof() = int8_t(ndof > 0 ? ndof : 1);
      }
    }
  };

  // ---------------------------------------------------------------------------------------------
  // THE FUSED PHASE LADDER. Run per bin, one (phase, iteration) of the refit ladder would be ten
  // serialised launches, each with only a few hundred live lanes while the rest of the device idles; the
  // solver is latency-bound (one thread marching a bordered-band double solve), so the per-bin
  // serialisation starves occupancy. With the lane partition at the head of this file all ten bins are
  // resident in one launch per (phase, iteration), and wall time becomes the slowest bin, not their sum.
  // Bin b runs the out-of-line Kernel_BLFitPhase*<b+kRefitMinN>::lane (the trampolines below rebuild the
  // bin's kernel object from the shared config); no lane body reads its lane index beyond its own
  // addresses, so the arithmetic does not depend on where in the lane space a track lands.
  //
  // N-independent config of one fused launch: the members the per-N phase kernels carry, so one object
  // serves all ten cases; the trampoline rebuilds the bin's kernel from it inside the callee's frame.
  struct BLRefitFusedCfg {
    const float* __restrict__ rhoMap = nullptr;
    const float* __restrict__ bMap = nullptr;
    const uint32_t* __restrict__ fitHitId = nullptr;
    uint32_t* __restrict__ dropHitId = nullptr;
    const uint8_t* __restrict__ fitHitIsCore = nullptr;
    uint32_t* __restrict__ coreProtectFires = nullptr;
    bool iterFromPhase = false;
    bool matFromPhase = false;
    bool fieldFromPhase = false;
    bool bFieldFromPhase = false;
    bool chargeSymmetric = false;
    bool trajectoryCorrections = false;
    bool scatteringLogAtTotal = false;
    bool elossCumulative = false;
    bool outlierReject = true;
    bool finalIter = true;
    bool fieldKernelWeights = false;
    bool coreProtect = false;
  };

  template <int N, typename TrackerTraits, uint32_t Stride>
  ALPAKA_FN_ACC BL_REFIT_NOINLINE void blFitPhasePrepLaneOutOfLine(
      Acc1D const& acc,
      BLRefitFusedCfg const& cfg,
      uint32_t lane,
      double bField,
      typename caStructures::tindex_type const* __restrict__ ptkids,
      double_st* __restrict__ phits,
      float_st* __restrict__ phits_ge,
      double_st* __restrict__ pfast_fit,
      generalBrokenLine::GblNodeData* __restrict__ pgnodes,
      double* __restrict__ pphase) {
    const Kernel_BLFitPhasePrep<N, TrackerTraits, Stride> k{cfg.rhoMap,
                                                            cfg.iterFromPhase,
                                                            cfg.bMap,
                                                            cfg.matFromPhase,
                                                            cfg.fieldFromPhase,
                                                            cfg.chargeSymmetric,
                                                            cfg.trajectoryCorrections,
                                                            cfg.scatteringLogAtTotal,
                                                            cfg.elossCumulative};
    k.lane(acc, lane, bField, ptkids, phits, phits_ge, pfast_fit, pgnodes, pphase);
  }

  template <int N, typename TrackerTraits, uint32_t Stride>
  ALPAKA_FN_ACC BL_REFIT_NOINLINE void blFitPhaseSolveLaneOutOfLine(Acc1D const& acc,
                                                                    BLRefitFusedCfg const& cfg,
                                                                    uint32_t lane,
                                                                    double bField,
                                                                    double_st* __restrict__ phits,
                                                                    double_st* __restrict__ pfast_fit,
                                                                    generalBrokenLine::GblNodeData* __restrict__ pgnodes,
                                                                    double_st* __restrict__ pscratch,
                                                                    double* __restrict__ pphase) {
    const Kernel_BLFitPhaseSolve<N, TrackerTraits, Stride> k{
        cfg.outlierReject, cfg.finalIter, cfg.bMap, cfg.fieldKernelWeights};
    k.lane(acc, lane, bField, phits, pfast_fit, pgnodes, pscratch, pphase);
  }

  template <int N, typename TrackerTraits, uint32_t Stride>
  ALPAKA_FN_ACC BL_REFIT_NOINLINE void blFitPhaseOutLaneOutOfLine(
      Acc1D const& acc,
      BLRefitFusedCfg const& cfg,
      uint32_t lane,
      double bField,
      OutputSoAView results_view,
      typename caStructures::tindex_type const* __restrict__ ptkids,
      double_st* __restrict__ phits,
      double_st* __restrict__ pfast_fit,
      double* __restrict__ pphase) {
    const Kernel_BLFitPhaseOut<N, TrackerTraits, Stride> k{cfg.bMap, cfg.bFieldFromPhase, cfg.chargeSymmetric};
    k.lane(acc, lane, bField, results_view, ptkids, phits, pfast_fit, pphase);
  }

  template <int N, typename TrackerTraits, uint32_t Stride>
  ALPAKA_FN_ACC BL_REFIT_NOINLINE void blFitPhaseOutlierLaneOutOfLine(
      Acc1D const& acc,
      BLRefitFusedCfg const& cfg,
      uint32_t lane,
      double bField,
      OutputSoAView results_view,
      typename caStructures::tindex_type const* __restrict__ ptkids,
      double_st* __restrict__ phits,
      double_st* __restrict__ pfast_fit,
      generalBrokenLine::GblNodeData* __restrict__ pgnodes,
      double_st* __restrict__ pscratch,
      double* __restrict__ pphase) {
    const Kernel_BLFitPhaseOutlier<N, TrackerTraits, Stride> k{cfg.outlierReject,
                                                               cfg.fitHitId,
                                                               cfg.dropHitId,
                                                               cfg.fitHitIsCore,
                                                               cfg.coreProtect,
                                                               cfg.coreProtectFires,
                                                               cfg.bMap,
                                                               cfg.bFieldFromPhase,
                                                               cfg.chargeSymmetric};
    k.lane(acc, lane, bField, results_view, ptkids, phits, pfast_fit, pgnodes, pscratch, pphase);
  }

  // THE FUSED KERNELS. The block index picks the bin (and its compile-time N) out of the device table
  // Kernel_BLRefitLaneRanges wrote; the block's elements are the bin's lanes. One bin per block, so the
  // switch over N cannot diverge inside a warp; an unused block returns at once.
  template <typename TrackerTraits, uint32_t Stride>
  struct Kernel_BLFitPhasePrepFused {
    static_assert(Stride == HelixFit<TrackerTraits>::kRefitStride, "the fused ladder spans the refit lane buffer");
    BLRefitFusedCfg cfg_;

    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  double bField,
                                  typename caStructures::tindex_type const* __restrict__ ptkids,
                                  double_st* __restrict__ phits,
                                  float_st* __restrict__ phits_ge,
                                  double_st* __restrict__ pfast_fit,
                                  generalBrokenLine::GblNodeData* __restrict__ pgnodes,
                                  double* __restrict__ pphase,
                                  const uint32_t* __restrict__ pBlockMap) const {
      constexpr auto invalidTkId = std::numeric_limits<typename caStructures::tindex_type>::max();
      constexpr uint32_t kTotal = kFusedBlocks * kFitBlock;
      for (uint32_t blk : cms::alpakatools::uniform_groups(acc, kTotal)) {
        const uint32_t bin = pBlockMap[kRefitBlockMapStride * blk];
        if (bin >= kRefitNBins)
          continue;  // idle block: this round's partition did not need it
        const uint32_t laneLo = pBlockMap[kRefitBlockMapStride * blk + 1];
        const uint32_t laneHi = pBlockMap[kRefitBlockMapStride * blk + 2];
        for (auto el : cms::alpakatools::uniform_group_elements(acc, blk, kTotal)) {
          const uint32_t local_idx = laneLo + uint32_t(el.local);
          if (local_idx >= laneHi)
            break;
          // Safety net only: a bin's range IS its exact count, so the scan fills every lane of it and
          // the sentinel cannot be reached unless the count and the fill ever disagree.
          if (invalidTkId == ptkids[local_idx])
            continue;
#define BL_REFIT_FUSED_PREP_CASE(BIN)                                                       \
  case (BIN):                                                                               \
    blFitPhasePrepLaneOutOfLine<(BIN) + kRefitMinN, TrackerTraits, Stride>(                 \
        acc, cfg_, local_idx, bField, ptkids, phits, phits_ge, pfast_fit, pgnodes, pphase); \
    break
          switch (bin) {
            BL_REFIT_FUSED_PREP_CASE(0);
            BL_REFIT_FUSED_PREP_CASE(1);
            BL_REFIT_FUSED_PREP_CASE(2);
            BL_REFIT_FUSED_PREP_CASE(3);
            BL_REFIT_FUSED_PREP_CASE(4);
            BL_REFIT_FUSED_PREP_CASE(5);
            BL_REFIT_FUSED_PREP_CASE(6);
            BL_REFIT_FUSED_PREP_CASE(7);
            BL_REFIT_FUSED_PREP_CASE(8);
            BL_REFIT_FUSED_PREP_CASE(9);
            default:
              break;
          }
#undef BL_REFIT_FUSED_PREP_CASE
        }
      }
    }
  };

  template <typename TrackerTraits, uint32_t Stride>
  struct Kernel_BLFitPhaseSolveFused {
    static_assert(Stride == HelixFit<TrackerTraits>::kRefitStride, "the fused ladder spans the refit lane buffer");
    BLRefitFusedCfg cfg_;

    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  double bField,
                                  typename caStructures::tindex_type const* __restrict__ ptkids,
                                  double_st* __restrict__ phits,
                                  double_st* __restrict__ pfast_fit,
                                  generalBrokenLine::GblNodeData* __restrict__ pgnodes,
                                  double_st* __restrict__ pscratch,
                                  double* __restrict__ pphase,
                                  const uint32_t* __restrict__ pBlockMap) const {
      constexpr auto invalidTkId = std::numeric_limits<typename caStructures::tindex_type>::max();
      constexpr uint32_t kTotal = kFusedBlocks * kFitBlock;
      for (uint32_t blk : cms::alpakatools::uniform_groups(acc, kTotal)) {
        const uint32_t bin = pBlockMap[kRefitBlockMapStride * blk];
        if (bin >= kRefitNBins)
          continue;
        const uint32_t laneLo = pBlockMap[kRefitBlockMapStride * blk + 1];
        const uint32_t laneHi = pBlockMap[kRefitBlockMapStride * blk + 2];
        for (auto el : cms::alpakatools::uniform_group_elements(acc, blk, kTotal)) {
          const uint32_t local_idx = laneLo + uint32_t(el.local);
          if (local_idx >= laneHi)
            break;
          if (invalidTkId == ptkids[local_idx])
            continue;
#define BL_REFIT_FUSED_SOLVE_CASE(BIN)                                              \
  case (BIN):                                                                       \
    blFitPhaseSolveLaneOutOfLine<(BIN) + kRefitMinN, TrackerTraits, Stride>(        \
        acc, cfg_, local_idx, bField, phits, pfast_fit, pgnodes, pscratch, pphase); \
    break
          switch (bin) {
            BL_REFIT_FUSED_SOLVE_CASE(0);
            BL_REFIT_FUSED_SOLVE_CASE(1);
            BL_REFIT_FUSED_SOLVE_CASE(2);
            BL_REFIT_FUSED_SOLVE_CASE(3);
            BL_REFIT_FUSED_SOLVE_CASE(4);
            BL_REFIT_FUSED_SOLVE_CASE(5);
            BL_REFIT_FUSED_SOLVE_CASE(6);
            BL_REFIT_FUSED_SOLVE_CASE(7);
            BL_REFIT_FUSED_SOLVE_CASE(8);
            BL_REFIT_FUSED_SOLVE_CASE(9);
            default:
              break;
          }
#undef BL_REFIT_FUSED_SOLVE_CASE
        }
      }
    }
  };

  template <typename TrackerTraits, uint32_t Stride>
  struct Kernel_BLFitPhaseOutFused {
    static_assert(Stride == HelixFit<TrackerTraits>::kRefitStride, "the fused ladder spans the refit lane buffer");
    BLRefitFusedCfg cfg_;

    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  double bField,
                                  OutputSoAView results_view,
                                  typename caStructures::tindex_type const* __restrict__ ptkids,
                                  double_st* __restrict__ phits,
                                  double_st* __restrict__ pfast_fit,
                                  double* __restrict__ pphase,
                                  const uint32_t* __restrict__ pBlockMap) const {
      constexpr auto invalidTkId = std::numeric_limits<typename caStructures::tindex_type>::max();
      constexpr uint32_t kTotal = kFusedBlocks * kFitBlock;
      for (uint32_t blk : cms::alpakatools::uniform_groups(acc, kTotal)) {
        const uint32_t bin = pBlockMap[kRefitBlockMapStride * blk];
        if (bin >= kRefitNBins)
          continue;
        const uint32_t laneLo = pBlockMap[kRefitBlockMapStride * blk + 1];
        const uint32_t laneHi = pBlockMap[kRefitBlockMapStride * blk + 2];
        for (auto el : cms::alpakatools::uniform_group_elements(acc, blk, kTotal)) {
          const uint32_t local_idx = laneLo + uint32_t(el.local);
          if (local_idx >= laneHi)
            break;
          if (invalidTkId == ptkids[local_idx])
            continue;
#define BL_REFIT_FUSED_OUT_CASE(BIN)                                                   \
  case (BIN):                                                                          \
    blFitPhaseOutLaneOutOfLine<(BIN) + kRefitMinN, TrackerTraits, Stride>(             \
        acc, cfg_, local_idx, bField, results_view, ptkids, phits, pfast_fit, pphase); \
    break
          switch (bin) {
            BL_REFIT_FUSED_OUT_CASE(0);
            BL_REFIT_FUSED_OUT_CASE(1);
            BL_REFIT_FUSED_OUT_CASE(2);
            BL_REFIT_FUSED_OUT_CASE(3);
            BL_REFIT_FUSED_OUT_CASE(4);
            BL_REFIT_FUSED_OUT_CASE(5);
            BL_REFIT_FUSED_OUT_CASE(6);
            BL_REFIT_FUSED_OUT_CASE(7);
            BL_REFIT_FUSED_OUT_CASE(8);
            BL_REFIT_FUSED_OUT_CASE(9);
            default:
              break;
          }
#undef BL_REFIT_FUSED_OUT_CASE
        }
      }
    }
  };

  // The outlier phase runs for N >= 5 only, so bins 0 and 1 fall through to `default` and are left
  // untouched: their solve never filled the pulls the phase would read.
  template <typename TrackerTraits, uint32_t Stride>
  struct Kernel_BLFitPhaseOutlierFused {
    static_assert(Stride == HelixFit<TrackerTraits>::kRefitStride, "the fused ladder spans the refit lane buffer");
    BLRefitFusedCfg cfg_;

    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  double bField,
                                  OutputSoAView results_view,
                                  typename caStructures::tindex_type const* __restrict__ ptkids,
                                  double_st* __restrict__ phits,
                                  double_st* __restrict__ pfast_fit,
                                  generalBrokenLine::GblNodeData* __restrict__ pgnodes,
                                  double_st* __restrict__ pscratch,
                                  double* __restrict__ pphase,
                                  const uint32_t* __restrict__ pBlockMap) const {
      constexpr auto invalidTkId = std::numeric_limits<typename caStructures::tindex_type>::max();
      constexpr uint32_t kTotal = kFusedBlocks * kFitBlock;
      for (uint32_t blk : cms::alpakatools::uniform_groups(acc, kTotal)) {
        const uint32_t bin = pBlockMap[kRefitBlockMapStride * blk];
        if (bin >= kRefitNBins)
          continue;
        const uint32_t laneLo = pBlockMap[kRefitBlockMapStride * blk + 1];
        const uint32_t laneHi = pBlockMap[kRefitBlockMapStride * blk + 2];
        for (auto el : cms::alpakatools::uniform_group_elements(acc, blk, kTotal)) {
          const uint32_t local_idx = laneLo + uint32_t(el.local);
          if (local_idx >= laneHi)
            break;
          if (invalidTkId == ptkids[local_idx])
            continue;
#define BL_REFIT_FUSED_OUTLIER_CASE(BIN)                                                                  \
  case (BIN):                                                                                             \
    blFitPhaseOutlierLaneOutOfLine<(BIN) + kRefitMinN, TrackerTraits, Stride>(                            \
        acc, cfg_, local_idx, bField, results_view, ptkids, phits, pfast_fit, pgnodes, pscratch, pphase); \
    break
          switch (bin) {
            BL_REFIT_FUSED_OUTLIER_CASE(2);
            BL_REFIT_FUSED_OUTLIER_CASE(3);
            BL_REFIT_FUSED_OUTLIER_CASE(4);
            BL_REFIT_FUSED_OUTLIER_CASE(5);
            BL_REFIT_FUSED_OUTLIER_CASE(6);
            BL_REFIT_FUSED_OUTLIER_CASE(7);
            BL_REFIT_FUSED_OUTLIER_CASE(8);
            BL_REFIT_FUSED_OUTLIER_CASE(9);
            default:
              break;  // bins 0 and 1 are N = 3, 4: no outlier phase
          }
#undef BL_REFIT_FUSED_OUTLIER_CASE
        }
      }
    }
  };

#undef BL_REFIT_NOINLINE

  // ---------------------------------------------------------------------------------------------
  // Extended-N REFIT. After the merger's OT hit-attach rewrite, each accepted-extended track's REWRITTEN
  // hit container holds its originals + attached extras (tagged OT-rechit ids carry bit30). A full GBL
  // refit -- the OT lever arm shrinks the longitudinal/pT covariance -- overwrites the pre-refit state/cov/chi2.
  // The fit kernels are hit-source agnostic (reused as-is); only the hit LOAD needs merged-vs-OT dispatch.
  // Runs wherever refitExtended is called: refitMergedTwins and refitDedupUnions.
  // ---------------------------------------------------------------------------------------------

  // Fit-hit selection over the rewritten container (caFitHitSel::dedupWalk), but INCLUDING tagged
  // OT-rechit ids (bit30): they index the raw OT source and must be counted as fit hits (unlike the
  // verify guard, which SKIPS them). Same kMode filter + pixel-overlap dedup. k<0 -> count; k>=0 -> k-th.
  template <typename TupleCont, typename HitsView>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t
  refitDedupWalk(TupleCont const* __restrict__ hitContainer, uint32_t it, HitsView hh, bool hasStubs, int k) {
    auto const* hitId = hitContainer->begin(it);
    auto const nhits = hitContainer->size(it);
    auto const nTot = hh.metadata().size();
    uint32_t nkept = 0;
    int lastKeptJ = -1;
    for (uint32_t j = 0; j < uint32_t(nhits); ++j) {
      auto const h = hitId[j];
      const bool ot = caOTHitTag::isOTId(h);
      if (!ot && h >= static_cast<uint32_t>(nTot))
        break;  // genuine content-overflow guard (untagged out-of-range id)
      // OT extras count as stubs for the kMode filter; merged ids consult the SoA stub flag.
      const bool hitIsStub = ot ? true : reco::isStub(hh, int32_t(h));
      if (!caFitHitSel::useHit(hitIsStub, hasStubs))
        continue;
      // Merge only two consecutive kept MERGED-PIXEL hits (never an OT extra or a stub).
      if (hasStubs && lastKeptJ >= 0 && !ot && !reco::isStub(hh, int32_t(h))) {
        auto const hp = hitId[lastKeptJ];
        if (!caOTHitTag::isOTId(hp) && !reco::isStub(hh, int32_t(hp))) {
          float const dx = float(hh[h].xGlobal()) - float(hh[hp].xGlobal());
          float const dy = float(hh[h].yGlobal()) - float(hh[hp].yGlobal());
          if (dx * dx + dy * dy < caFitHitSel::kDedupDsMin2)
            continue;
        }
      }
      if (int(nkept) == k)
        return j;
      ++nkept;
      lastKeptJ = int(j);
    }
    return nkept;
  }

  // ---------------------------------------------------------------------------------------------
  // THE DYNAMIC PARTITION, part 1: count. One launch for the WHOLE ladder -- nSel does not depend
  // on N, so this kernel is not templated and one sweep produces all ten demands. It walks the same
  // population/gates/refitDedupWalk as the per-bin scans, binned with refitBinOfNSel, so count and
  // claim cannot disagree. The counters are the DEMAND of this round (exclude earlier-round seated).
  // ---------------------------------------------------------------------------------------------
  class Kernel_BLRefitBinCount {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  Tuples const* __restrict__ hitContainer,
                                  ::reco::TrackingRecHitConstView hh,
                                  const int32_t* __restrict__ acceptedByTuple,
                                  const uint8_t* __restrict__ pServed,
                                  uint32_t* __restrict__ pCounts,
                                  uint32_t maxFitSel,
                                  uint32_t nTracksCap,
                                  bool hasStubs) const {
      const bool hasStubsRt = hasStubs && (static_cast<int32_t>(hh.offsetStubs()) >= 0);
      for (auto tkid : cms::alpakatools::uniform_elements(acc, nTracksCap)) {
        if (acceptedByTuple[tkid] < 0)
          continue;
        if (pServed != nullptr && pServed[tkid] != 0u)
          continue;  // seated in an earlier round
        const uint32_t nSel = refitDedupWalk(hitContainer, tkid, hh, hasStubsRt, /*k=*/-1);
        const uint32_t bin = refitBinOfNSel(nSel, maxFitSel);
        if (bin >= kRefitNBins)
          continue;  // outside the ladder: no bin fits this track, exactly as in the per-bin scans
        alpaka::atomicAdd(acc, pCounts + bin, 1u, alpaka::hierarchy::Grids{});
      }
    }
  };

  // ---------------------------------------------------------------------------------------------
  // THE DYNAMIC PARTITION, part 2: ranges. Single-threaded, device-only (no host crossing): hands
  // kRefitStride lanes out in bin order, writes the block->bin dispatch table (one bin per block,
  // idle tail at kRefitNBins). The per-track served flag makes the round-bound exact.
  // ---------------------------------------------------------------------------------------------
  class Kernel_BLRefitLaneRanges {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  const uint32_t* __restrict__ pCounts,
                                  uint32_t* __restrict__ pRange,
                                  uint32_t* __restrict__ pBlockMap,
                                  uint32_t laneTotal) const {
      if (alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0] != 0)
        return;
      uint32_t base = 0u;
      for (uint32_t b = 0; b < kRefitNBins; ++b) {
        const uint32_t want = pCounts[b];
        const uint32_t room = (base < laneTotal) ? (laneTotal - base) : 0u;
        const uint32_t got = (want < room) ? want : room;
        pRange[kRefitRangeStride * b + 0] = base;
        pRange[kRefitRangeStride * b + 1] = got;
        base += got;
      }
      uint32_t blk = 0u;
      for (uint32_t b = 0; b < kRefitNBins; ++b) {
        const uint32_t lo = pRange[kRefitRangeStride * b + 0];
        const uint32_t hi = lo + pRange[kRefitRangeStride * b + 1];
        for (uint32_t l = lo; l < hi; l += kFitBlock) {
          ALPAKA_ASSERT_ACC(blk < kFusedBlocks);  // sum_b ceil(n_b/kFitBlock) <= kFusedBlocks
          const uint32_t e = (l + kFitBlock < hi) ? (l + kFitBlock) : hi;
          pBlockMap[kRefitBlockMapStride * blk + 0] = b;
          pBlockMap[kRefitBlockMapStride * blk + 1] = l;
          pBlockMap[kRefitBlockMapStride * blk + 2] = e;
          ++blk;
        }
      }
      for (; blk < kFusedBlocks; ++blk) {
        pBlockMap[kRefitBlockMapStride * blk + 0] = kRefitNBins;  // idle
        pBlockMap[kRefitBlockMapStride * blk + 1] = 0u;
        pBlockMap[kRefitBlockMapStride * blk + 2] = 0u;
      }
    }
  };

  // Prepares the BLFit input buffers for the extended-N refit. Keeps accepted-extended tuples whose
  // selected multiplicity falls in [nHitsL,nHitsH], dense-compacts into ptkids/buffers via a grid atomic
  // (ptkids pre-set to the invalid sentinel, preserving the fit kernels' break-on-invalid tail).
  // Per-hit dispatch: raw OT SoA for tagged ids, merged SoA + innerSensorFrame otherwise (same as Kernel_BLFastFit).
  template <int N, uint32_t Stride = riemannFit::stride>
  class Kernel_BLFastFitRefit {
  public:
    // OT source by value (as the extension kernels take it): default (nOTHits==0) => no tagged ids.
    caExtension::OTHitsSource otSource_{};
    // Per-lane record of the RAW hit id loaded into each fit slot (fitHitId_[lane*kRefitFitIdQuota + i] = id of fit
    // hit i). The outlier phase maps its dropped measurement node back to this id so the merger can
    // remove the fit-rejected hit from the emitted TrackHitSoA list. Null means no such record is kept.
    uint32_t* __restrict__ fitHitId_ = nullptr;
    // Core-protected outlier: per-lane record of whether each fit slot is an ORIGINAL PIXEL-CORE hit
    // (id below offsetStubs, not a bit30 OT tag). The outlier phase reads it to restrict the drop to
    // non-core (appended) nodes. Null (core protection off) means no such record is kept.
    uint8_t* __restrict__ fitHitIsCore_ = nullptr;
    // Device-resident lane range of this bin for this round, {firstLane, nLanes} at
    // pLaneRange_[kRefitRangeStride*(N-kRefitMinN)], written by Kernel_BLRefitLaneRanges. Null means
    // base 0 and the whole buffer stride (the extension's prediction scans, ExtPredCoeff.dev.cc).
    // Launch-only: decides where, never a value.
    const uint32_t* __restrict__ pLaneRange_ = nullptr;
    // Per-track "already seated in an earlier round" flag, nTracksCap bytes. A track is skipped while
    // it is set and it is set the moment the track claims a lane, so consecutive rounds partition the
    // population exactly: each round seats min(remaining, capacity) tracks and none is seated twice.
    // Null means single-round behaviour with no gate and no marking (the extension's prediction scans).
    uint8_t* __restrict__ pServed_ = nullptr;

    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  Tuples const* __restrict__ hitContainer,
                                  ::reco::TrackingRecHitConstView hh,
                                  ::reco::CAModulesConstView cm,
                                  const int32_t* __restrict__ acceptedByTuple,
                                  typename caStructures::tindex_type* __restrict__ ptkids,
                                  double_st* __restrict__ phits,
                                  float_st* __restrict__ phits_ge,
                                  double_st* __restrict__ pfast_fit,
                                  uint32_t* __restrict__ pSlot,
                                  uint32_t nHitsL,
                                  uint32_t nHitsH,
                                  uint32_t nTracksCap,
                                  bool hasStubs) const {
      constexpr uint32_t hitsInFit = N;
      constexpr uint32_t kBin = uint32_t(N) - uint32_t(kRefitMinN);
      // Where this bin's lanes are, and how many of them there are, in the shared stride-wide buffer.
      // Without a range table this is the whole buffer from lane 0 (the extension's prediction scans).
      const uint32_t laneBase = (pLaneRange_ != nullptr) ? pLaneRange_[kRefitRangeStride * kBin] : 0u;
      const uint32_t nt = (pLaneRange_ != nullptr) ? pLaneRange_[kRefitRangeStride * kBin + 1] : Stride;
      const bool hasStubsRt = hasStubs && (static_cast<int32_t>(hh.offsetStubs()) >= 0);
      for (auto tkid : cms::alpakatools::uniform_elements(acc, nTracksCap)) {
        if (acceptedByTuple[tkid] < 0)
          continue;  // only accepted-extended tuples get refit
        if (pServed_ != nullptr && pServed_[tkid] != 0u)
          continue;  // already refit in an earlier round
        const uint32_t nSel = refitDedupWalk(hitContainer, tkid, hh, hasStubsRt, /*k=*/-1);
        if (nSel < nHitsL || nSel > nHitsH)
          continue;  // not this N-bin
        const uint32_t slot = alpaka::atomicAdd(acc, pSlot, 1u, alpaka::hierarchy::Grids{});
        if (slot >= nt) {
          // Under the round loop (pServed_ set) this is the expected round overflow: the bin's lane
          // range was cut by the round capacity and the track is simply left for the next round, so
          // nothing is reported. Without a range table the slot cap binds and the
          // track keeps its unrefit CA state: exactly one thread per launch observes slot == nt (the
          // slot comes from a grid-scope atomicAdd), so the report is one line per binding launch.
          if (pServed_ == nullptr && slot == nt)
            printf(
                "[refit slot cap] BOUND: N-bin %u..%u demanded > kRefitStride %u; tracks beyond the cap keep "
                "their unrefit CA state.\n",
                nHitsL,
                nHitsH,
                nt);
          continue;
        }
        // The lane this claim owns in the shared buffer (laneBase is 0 without a range table).
        const uint32_t lane = laneBase + slot;
        if (pServed_ != nullptr)
          pServed_[tkid] = 1u;  // seated: no later round may claim it again
        ptkids[lane] = caStructures::tindex_type(tkid);

        auto const* hitId = hitContainer->begin(tkid);
        riemannFit::Map3xNdS<N, Stride> hits(phits + lane);
        riemannFit::Map4dS<Stride> fast_fit(pfast_fit + lane);
        riemannFit::Map6xNfS<N, Stride> hits_ge(phits_ge + lane);

        // Uniform sampling of hitsInFit hits from the deduped selected set (force the last kept
        // hit for maximum lever arm -- the outermost OT extra), identical to Kernel_BLFastFit.
        uint32_t selectedHits[N];
        {
          float incr = std::max(1.f, float(nSel) / float(hitsInFit));
          float fn = 0;
          for (uint32_t i = 0; i < hitsInFit; ++i) {
            int kk = int(fn + 0.5f);
            if (hitsInFit - 1 == i)
              kk = int(nSel) - 1;
            selectedHits[i] = refitDedupWalk(hitContainer, tkid, hh, hasStubsRt, kk);
            fn += incr;
          }
        }

        for (uint32_t i = 0; i < hitsInFit; ++i) {
          const uint32_t hid = hitId[selectedHits[i]];
          // Record this fit slot's raw hit id (bit30 OT tag preserved) so the outlier phase can
          // name the hit it drops. lane == the lane the fit/outlier kernels iterate (ptkids[lane]).
          if (fitHitId_ != nullptr)
            fitHitId_[std::size_t(lane) * std::size_t(kRefitFitIdQuota) + i] = hid;
          // Mark ORIGINAL PIXEL-CORE fit hits (id < offsetStubs and not a bit30 OT tag). Appended
          // extras -- bit30 raw-OT (isOTId) and merged OT stubs (id >= offsetStubs) -- are non-core and
          // remain droppable; only the pixel core is protected. Same lane*kRefitFitIdQuota + i indexing as fitHitId_.
          if (fitHitIsCore_ != nullptr) {
            const bool isCore = !caOTHitTag::isOTId(hid) && (!hasStubsRt || int32_t(hid) < int32_t(hh.offsetStubs()));
            fitHitIsCore_[std::size_t(lane) * std::size_t(kRefitFitIdQuota) + i] = isCore ? uint8_t(1) : uint8_t(0);
          }
          float ge[6];
          float px, py, pz;
          if (caOTHitTag::isOTId(hid)) {
            // Raw OT rechit: lower/upper sensor frame per position in the stack (verify/WriteFinal recipe).
            const uint32_t o = caOTHitTag::otIdx(hid);
            const uint32_t geom = uint32_t(otSource_.otHits[o].detectorIndex()) - ::phase2PixelTopology::nModulesPix;
            const bool isUpper = (o >= otSource_.otHitModules.upperSensorStart()[geom]);
            const auto& frame = isUpper ? otSource_.stackedGeometry.upperSensorFrame()[geom]
                                        : otSource_.stackedGeometry.lowerSensorFrame()[geom];
            float xerrFit = otSource_.otHits[o].xerrLocal();
            const bool is2S = otSource_.otHits[o].yerrLocal() > 0.1f;
            const bool isBarrelHit = alpaka::math::abs(acc, otSource_.otHits[o].zGlobal()) < 118.f;
            const float f = is2S ? (isBarrelHit ? 0.4624f : 0.8464f) : (isBarrelHit ? 0.64f : 0.9025f);
            xerrFit *= f;
            float yerrFit = otSource_.otHits[o].yerrLocal();
            frame.toGlobal(xerrFit, 0.f, yerrFit, ge);
            px = otSource_.otHits[o].xGlobal();
            py = otSource_.otHits[o].yGlobal();
            pz = otSource_.otHits[o].zGlobal();
          } else {
            // Merged SoA hit: the same load as Kernel_BLFastFit (innerSensorFrame + stub xerr scale).
            auto frame = cm.innerSensorFrame(hh.detectorIndex(hid));
            float xerrFit = hh[hid].xerrLocal();
            if (reco::isStub(hh, int32_t(hid))) {
              const bool is2S = hh[hid].yerrLocal() > 0.1f;
              const bool isBarrelHit = alpaka::math::abs(acc, hh[hid].zGlobal()) < 118.f;
              const float f = is2S ? (isBarrelHit ? 0.4624f : 0.8464f) : (isBarrelHit ? 0.64f : 0.9025f);
              xerrFit *= f;
            }
            float yerrFit = hh[hid].yerrLocal();
            frame.toGlobal(xerrFit, 0.f, yerrFit, ge);
            px = hh[hid].xGlobal();
            py = hh[hid].yGlobal();
            pz = hh[hid].zGlobal();
          }
          hits.col(i) << px, py, pz;
          hits_ge.col(i) << ge[0], ge[1], ge[2], ge[3], ge[4], ge[5];
        }
        brokenline::fastFit(acc, hits, fast_fit);
      }
    }
  };

  // ---------------------------------------------------------------------------------------------
  // Per-N launcher helpers (extern template). Each bundles the fast-fit + fit launches for ONE
  // multiplicity N. The launch state travels in a small context POD, so the helper is a plain function
  // template that can be `extern template`-declared here and instantiated in a disjoint-N TU.
  // ---------------------------------------------------------------------------------------------

  // Main-fit launch context: the launchBrokenLineKernels locals and HelixFit members the launches need.
  template <typename TrackerTraits>
  struct BLMainLaunchCtx {
    Queue& queue;
    Tuples const* tuples;
    TupleMultiplicity const* tupleMultiplicity;
    ::reco::TrackingRecHitConstView hv;
    ::reco::CAModulesConstView cm;
    OutputSoAView outputSoa;
    double bField;
    const float* rhoMap;
    const float* bMap;  // normalized (Bz,Br) r-z field map; null (or fitCorrections off) => the scalar bField
    typename caStructures::tindex_type* tkids;
    double_st* phits;
    float_st* phits_ge;
    double_st* pfast_fit;
    double_st* pgblScratch;
    bool fitCorrections;  // fit correctness package (see Kernel_BLFit::fitCorrections_)
    // Dynamic-partition state of the fused ladder (unused on the per-bin path, which addresses the whole
    // buffer from lane 0). Both tables are written on the device by Kernel_BLMainLaneRanges, once per round.
    const uint32_t* pRange;     // per-bin {firstLane, nLanes, tupleBase}, kMainRangeStride * kMainNBins
    const uint32_t* pBlockMap;  // per-block {bin, firstLane, endLane}, kMainBlockMapStride * kMainFusedBlocks
    WorkDiv1D workDivFused;     // kMainFusedBlocks x kFitBlock, fixed and host-known
  };

  // One (fast-fit + fit) launch of multiplicity N over the current chunk. nHitsL/nHitsH are the
  // multiplicity bin of the fast-fit (== N for exact bins, [maxHitsOnTrackForFullFit, maxHitsOnTrack-1]
  // for the "rest" bin). The CA main fit is the factorized fast BrokenLine fit: one launch, one
  // linearization, no phase split. The GBL fit runs in the merger only (refitExtended, the fused refit ladder).
  template <int N, typename TrackerTraits>
  void runMainBin(
      BLMainLaunchCtx<TrackerTraits> const& c, WorkDiv1D const& wd, uint32_t nHitsL, uint32_t nHitsH, int32_t offset) {
    alpaka::exec<Acc1D>(c.queue,
                        wd,
                        Kernel_BLFastFit<N>{},
                        c.tuples,
                        c.tupleMultiplicity,
                        c.hv,
                        c.cm,
                        c.tkids,
                        c.phits,
                        c.phits_ge,
                        c.pfast_fit,
                        nHitsL,
                        nHitsH,
                        offset,
                        std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>);
    alpaka::exec<Acc1D>(c.queue,
                        wd,
                        Kernel_BLFit<N, TrackerTraits>{c.rhoMap, c.bMap, c.fitCorrections},
                        c.tupleMultiplicity,
                        c.bField,
                        c.outputSoa,
                        c.tkids,
                        c.phits,
                        c.phits_ge,
                        c.pfast_fit,
                        c.pgblScratch);
  }

  // ---------------------------------------------------------------------------------------------
  // FUSED MAIN-LADDER LAUNCHERS. One launch per phase, all N-bins, on the fixed fused grid. Declared
  // `extern template` and instantiated ONE PHASE PER TU: a fused kernel pulls every compile-time N of
  // its phase into whichever TU instantiates it (per-phase rather than per-N build-time split).
  // ---------------------------------------------------------------------------------------------
  template <typename TrackerTraits>
  void runMainFusedFast(BLMainLaunchCtx<TrackerTraits> const& c) {
    BLMainFusedCfg cfg{};
    cfg.hasStubs = std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>;
    alpaka::exec<Acc1D>(c.queue,
                        c.workDivFused,
                        Kernel_BLFastFitFused<TrackerTraits>{cfg},
                        c.tuples,
                        c.tupleMultiplicity,
                        c.hv,
                        c.cm,
                        c.tkids,
                        c.phits,
                        c.phits_ge,
                        c.pfast_fit,
                        c.pRange,
                        c.pBlockMap);
  }

  template <typename TrackerTraits>
  void runMainFusedFit(BLMainLaunchCtx<TrackerTraits> const& c) {
    BLMainFusedCfg cfg{};
    cfg.rhoMap = c.rhoMap;
    cfg.bMap = c.bMap;
    cfg.fitCorrections = c.fitCorrections;
    alpaka::exec<Acc1D>(c.queue,
                        c.workDivFused,
                        Kernel_BLFitFused<TrackerTraits>{cfg},
                        c.tupleMultiplicity,
                        c.bField,
                        c.outputSoa,
                        c.tkids,
                        c.phits,
                        c.phits_ge,
                        c.pfast_fit,
                        c.pgblScratch,
                        c.pBlockMap);
  }

  // Extended-N refit launch context: the refitExtended locals and HelixFit members the launches need.
  template <typename TrackerTraits>
  struct BLRefitLaunchCtx {
    Queue& queue;
    Tuples const* hitContainer;
    TupleMultiplicity const* tupleMultiplicity;
    ::reco::TrackingRecHitConstView hv;
    ::reco::CAModulesConstView cm;
    OutputSoAView outputSoa;
    const int32_t* acceptedByTuple;
    caExtension::OTHitsSource otSrc;
    double bField;
    const float* rhoMap;
    const float* bMap;  // normalized (Bz,Br) r-z field map; null => the scalar bField everywhere
    bool outlierReject;
    uint32_t maxNumberOfTuples;
    WorkDiv1D workDivScan;
    typename caStructures::tindex_type* tkids;
    double_st* phits;
    float_st* phits_ge;
    double_st* pfast_fit;
    generalBrokenLine::GblNodeData* pgnodes;
    double_st* pgblScratch;
    uint32_t* pslot;
    double* pphase;
    // Per-lane fit-hit-id table (written by Kernel_BLFastFitRefit) + the per-merged-track
    // dropped-hit-id output (consumed by the merger's post-refit hit-list compaction). Both null when
    // the fit-rejected hit is left in the emitted list, in which case neither is written.
    uint32_t* pFitHitId;
    uint32_t* pDropHitId;
    // Core-protected outlier: per-lane pixel-core flag table (written by
    // Kernel_BLFastFitRefit) + the enable. With pFitHitIsCore null / coreProtect false the outlier scan
    // considers every node.
    uint8_t* pFitHitIsCore;
    bool coreProtect;
    // Core-protect fire counter: optional single-uint32 device counter forwarded to the outlier kernels.
    // Null means no shadow scan and no atomic. To use it: allocate a uint32[1], memset it to 0, set this
    // field, then memcpy it back after the refit ladder.
    uint32_t* pCoreProtectFires = nullptr;
    // Fit-consistent curvature->pT conversion field (see Kernel_BLFitPhaseSolve::fieldKernelWeights_).
    // Needs the field map, so it is inert wherever bMap is null.
    bool fieldKernelWeights = false;
    // Charge-symmetric corrections package (see Kernel_BLFitPhasePrep::chargeSymmetric_). Its arc-sign
    // half is map-independent; its field-profile half needs bMap and is inert wherever the map is null.
    bool chargeSymmetric = false;
    // Reference-trajectory corrections package (see Kernel_BLFitPhasePrep::trajectoryCorrections_).
    bool trajectoryCorrections = false;
    // Highland's log at the track's TOTAL declared material rather than gap by gap (see
    // Kernel_BLFitPhasePrep::scatteringLogAtTotal_). Map-independent.
    bool scatteringLogAtTotal = false;
    // Cumulative-column typical-loss law rather than the per-lump Landau MPV (see
    // Kernel_BLFitPhasePrep::elossCumulative_). Map-independent.
    bool cumulativeEloss = false;
    // Dynamic-partition state of the fused ladder.
    // Per-bin demand of the current round (kRefitNBins uint32), filled by Kernel_BLRefitBinCount.
    uint32_t* pCounts = nullptr;
    // Per-bin {firstLane, nLanes} of the current round (kRefitRangeStride * kRefitNBins uint32).
    uint32_t* pRange = nullptr;
    // Per-block {bin, firstLane, endLane} dispatch table (kRefitBlockMapStride * kFusedBlocks uint32).
    uint32_t* pBlockMap = nullptr;
    // Per-track "already seated in an earlier round" flag (nTracksCap bytes).
    uint8_t* pServed = nullptr;
    // Work division of the fused phase launches: kFusedBlocks x kFitBlock, fixed and host-known.
    WorkDiv1D workDivFused;
  };

  // ---------------------------------------------------------------------------------------------
  // FUSED-LADDER LAUNCHERS. One (phase, iteration) per call, all ten N-bins in one launch (sum_b W_b ==
  // kRefitStride). Declared `extern template` and instantiated ONE PHASE PER TU: a fused kernel pulls all
  // ten compile-time N of its phase into whichever TU instantiates it (per-phase build-time split).
  // PHASE-BUFFER CACHES: the material march + segmentXX0Moments rows and the bFieldEff slot are carried
  // across the two linearizations instead of being recomputed. The material rows are carried between two
  // launches of the SAME kernel object (same machine code, so the cached doubles cannot differ from the
  // recompute); bFieldEff is carried from PREP to OUT/OUTLIER (different inlining contexts of
  // blEffectiveBField, may contract FMAs differently, so it can legitimately move a result at 1 ULP).
  // ---------------------------------------------------------------------------------------------

  // One bin's fast-fit compaction scan for the FUSED ladder. It keeps its own launch for the grid-scope
  // atomic slot claim that must complete before any phase reads ptkids; it claims against the bin's OWN
  // counter and seats tracks in the bin's OWN lane range.
  template <int N, typename TrackerTraits>
  void runRefitScanBin(BLRefitLaunchCtx<TrackerTraits> const& c, uint32_t nHitsL, uint32_t nHitsH) {
    constexpr uint32_t kRefitStride = HelixFit<TrackerTraits>::kRefitStride;
    constexpr uint32_t kBin = uint32_t(N) - uint32_t(kRefitMinN);
    constexpr bool hasStubs = true;  // Phase2OTStubs
    Kernel_BLFastFitRefit<N, kRefitStride> scan{c.otSrc, c.pFitHitId, c.pFitHitIsCore, c.pRange, c.pServed};
    alpaka::exec<Acc1D>(c.queue,
                        c.workDivScan,
                        scan,
                        c.hitContainer,
                        c.hv,
                        c.cm,
                        c.acceptedByTuple,
                        c.tkids,
                        c.phits,
                        c.phits_ge,
                        c.pfast_fit,
                        c.pCounts + kRefitNBins + kBin,  // the claim counters live past the demands
                        nHitsL,
                        nHitsH,
                        c.maxNumberOfTuples,
                        hasStubs);
  }

  template <typename TrackerTraits>
  void runRefitFusedPrep(BLRefitLaunchCtx<TrackerTraits> const& c, bool finalIter, bool fieldKernelOn) {
    constexpr uint32_t kRefitStride = HelixFit<TrackerTraits>::kRefitStride;
    BLRefitFusedCfg cfg{};
    cfg.rhoMap = c.rhoMap;
    cfg.bMap = c.bMap;
    // iterFromPhase / matFromPhase / fieldFromPhase are all finalIter: the bin's one scan fixed the
    // lane->track map for both iterations and the hits never move, so iteration 1 finds its own
    // iteration-0 reference hand-off, material rows and fit-consistent field in its own lane of c.pphase.
    // fieldFromPhase additionally needs the fit-consistent conversion field to be on (fieldKernelOn:
    // fieldKernelWeights with a map); otherwise the prep recomputes the hit-count average itself.
    cfg.iterFromPhase = finalIter;
    cfg.matFromPhase = finalIter;
    cfg.fieldFromPhase = finalIter && fieldKernelOn;
    cfg.chargeSymmetric = c.chargeSymmetric;
    cfg.trajectoryCorrections = c.trajectoryCorrections;
    cfg.scatteringLogAtTotal = c.scatteringLogAtTotal;
    cfg.elossCumulative = c.cumulativeEloss;
    alpaka::exec<Acc1D>(c.queue,
                        c.workDivFused,
                        Kernel_BLFitPhasePrepFused<TrackerTraits, kRefitStride>{cfg},
                        c.bField,
                        c.tkids,
                        c.phits,
                        c.phits_ge,
                        c.pfast_fit,
                        c.pgnodes,
                        c.pphase,
                        c.pBlockMap);
  }

  template <typename TrackerTraits>
  void runRefitFusedSolve(BLRefitLaunchCtx<TrackerTraits> const& c, bool finalIter) {
    constexpr uint32_t kRefitStride = HelixFit<TrackerTraits>::kRefitStride;
    BLRefitFusedCfg cfg{};
    cfg.bMap = c.bMap;
    cfg.outlierReject = c.outlierReject;
    cfg.finalIter = finalIter;
    cfg.fieldKernelWeights = c.fieldKernelWeights;
    alpaka::exec<Acc1D>(c.queue,
                        c.workDivFused,
                        Kernel_BLFitPhaseSolveFused<TrackerTraits, kRefitStride>{cfg},
                        c.bField,
                        c.tkids,
                        c.phits,
                        c.pfast_fit,
                        c.pgnodes,
                        c.pgblScratch,
                        c.pphase,
                        c.pBlockMap);
  }

  template <typename TrackerTraits>
  void runRefitFusedOut(BLRefitLaunchCtx<TrackerTraits> const& c) {
    constexpr uint32_t kRefitStride = HelixFit<TrackerTraits>::kRefitStride;
    BLRefitFusedCfg cfg{};
    cfg.bMap = c.bMap;
    // This iteration's prep ran on this lane and published the effective field (see PHASE-BUFFER CACHES).
    cfg.bFieldFromPhase = true;
    cfg.chargeSymmetric = c.chargeSymmetric;
    alpaka::exec<Acc1D>(c.queue,
                        c.workDivFused,
                        Kernel_BLFitPhaseOutFused<TrackerTraits, kRefitStride>{cfg},
                        c.bField,
                        c.outputSoa,
                        c.tkids,
                        c.phits,
                        c.pfast_fit,
                        c.pphase,
                        c.pBlockMap);
  }

  template <typename TrackerTraits>
  void runRefitFusedOutlier(BLRefitLaunchCtx<TrackerTraits> const& c) {
    constexpr uint32_t kRefitStride = HelixFit<TrackerTraits>::kRefitStride;
    BLRefitFusedCfg cfg{};
    cfg.bMap = c.bMap;
    cfg.fitHitId = c.pFitHitId;
    cfg.dropHitId = c.pDropHitId;
    cfg.fitHitIsCore = c.pFitHitIsCore;
    cfg.coreProtectFires = c.pCoreProtectFires;
    cfg.outlierReject = c.outlierReject;
    cfg.coreProtect = c.coreProtect;
    cfg.bFieldFromPhase = true;  // the final prep ran on this lane (see PHASE-BUFFER CACHES)
    cfg.chargeSymmetric = c.chargeSymmetric;
    alpaka::exec<Acc1D>(c.queue,
                        c.workDivFused,
                        Kernel_BLFitPhaseOutlierFused<TrackerTraits, kRefitStride>{cfg},
                        c.bField,
                        c.outputSoa,
                        c.tkids,
                        c.phits,
                        c.pfast_fit,
                        c.pgnodes,
                        c.pgblScratch,
                        c.pphase,
                        c.pBlockMap);
  }

  // Explicit-instantiation signatures (`extern` in the orchestration TU, bare in each
  // BrokenLineFit_*.dev.cc to instantiate its disjoint N-subset). Factorized fast BrokenLine main fit:
  // stubs N=3..10 in BrokenLineFit_stubsMainBL.dev.cc, four other traits N=3..6 in BrokenLineFit_main.dev.cc.
#define BLFIT_MAIN_SIG_BL(N, T)                    \
  template void runMainBin<N, ::pixelTopology::T>( \
      BLMainLaunchCtx<::pixelTopology::T> const&, WorkDiv1D const&, uint32_t, uint32_t, int32_t)
// The FUSED main ladder: one signature per PHASE and per traits set (each pulls every compile-time N
// of that phase into the TU that instantiates it).
#define BLFIT_MAIN_FUSED_FAST_SIG(T) \
  template void runMainFusedFast<::pixelTopology::T>(BLMainLaunchCtx<::pixelTopology::T> const&)
#define BLFIT_MAIN_FUSED_FIT_SIG(T) \
  template void runMainFusedFit<::pixelTopology::T>(BLMainLaunchCtx<::pixelTopology::T> const&)
#define BLFIT_REFIT_SCAN_SIG(N)                                     \
  template void runRefitScanBin<N, ::pixelTopology::Phase2OTStubs>( \
      BLRefitLaunchCtx<::pixelTopology::Phase2OTStubs> const&, uint32_t, uint32_t)
// The FUSED ladder: one signature per PHASE (each pulls all ten compile-time N of that phase into
// the TU that instantiates it), one TU per phase.
#define BLFIT_REFIT_FUSED_PREP_SIG()                               \
  template void runRefitFusedPrep<::pixelTopology::Phase2OTStubs>( \
      BLRefitLaunchCtx<::pixelTopology::Phase2OTStubs> const&, bool, bool)
#define BLFIT_REFIT_FUSED_SOLVE_SIG()                               \
  template void runRefitFusedSolve<::pixelTopology::Phase2OTStubs>( \
      BLRefitLaunchCtx<::pixelTopology::Phase2OTStubs> const&, bool)
#define BLFIT_REFIT_FUSED_OUT_SIG()                               \
  template void runRefitFusedOut<::pixelTopology::Phase2OTStubs>( \
      BLRefitLaunchCtx<::pixelTopology::Phase2OTStubs> const&)
#define BLFIT_REFIT_FUSED_OUTLIER_SIG()                               \
  template void runRefitFusedOutlier<::pixelTopology::Phase2OTStubs>( \
      BLRefitLaunchCtx<::pixelTopology::Phase2OTStubs> const&)

  // Main fit: N = 3 .. maxHitsOnTrackForFullFit(traits). Non-stubs traits fit up to 6, Phase2OTStubs up to 10.
  extern BLFIT_MAIN_SIG_BL(3, Phase1);
  extern BLFIT_MAIN_SIG_BL(4, Phase1);
  extern BLFIT_MAIN_SIG_BL(5, Phase1);
  extern BLFIT_MAIN_SIG_BL(6, Phase1);
  extern BLFIT_MAIN_SIG_BL(3, Phase2);
  extern BLFIT_MAIN_SIG_BL(4, Phase2);
  extern BLFIT_MAIN_SIG_BL(5, Phase2);
  extern BLFIT_MAIN_SIG_BL(6, Phase2);
  extern BLFIT_MAIN_SIG_BL(3, Phase2OT);
  extern BLFIT_MAIN_SIG_BL(4, Phase2OT);
  extern BLFIT_MAIN_SIG_BL(5, Phase2OT);
  extern BLFIT_MAIN_SIG_BL(6, Phase2OT);
  extern BLFIT_MAIN_SIG_BL(3, HIonPhase1);
  extern BLFIT_MAIN_SIG_BL(4, HIonPhase1);
  extern BLFIT_MAIN_SIG_BL(5, HIonPhase1);
  extern BLFIT_MAIN_SIG_BL(6, HIonPhase1);

  // Phase2OTStubs, N = 3 .. 10.
  extern BLFIT_MAIN_SIG_BL(3, Phase2OTStubs);
  extern BLFIT_MAIN_SIG_BL(4, Phase2OTStubs);
  extern BLFIT_MAIN_SIG_BL(5, Phase2OTStubs);
  extern BLFIT_MAIN_SIG_BL(6, Phase2OTStubs);
  extern BLFIT_MAIN_SIG_BL(7, Phase2OTStubs);
  extern BLFIT_MAIN_SIG_BL(8, Phase2OTStubs);
  extern BLFIT_MAIN_SIG_BL(9, Phase2OTStubs);
  extern BLFIT_MAIN_SIG_BL(10, Phase2OTStubs);

  // Main fit, FUSED ladder: one instantiation per (phase, traits). The stubs phases get a TU each
  // (they carry N = 3..10); the four non-stubs traits (N = 3..6) share one.
  extern BLFIT_MAIN_FUSED_FAST_SIG(Phase1);
  extern BLFIT_MAIN_FUSED_FAST_SIG(Phase2);
  extern BLFIT_MAIN_FUSED_FAST_SIG(Phase2OT);
  extern BLFIT_MAIN_FUSED_FAST_SIG(HIonPhase1);
  extern BLFIT_MAIN_FUSED_FAST_SIG(Phase2OTStubs);
  extern BLFIT_MAIN_FUSED_FIT_SIG(Phase1);
  extern BLFIT_MAIN_FUSED_FIT_SIG(Phase2);
  extern BLFIT_MAIN_FUSED_FIT_SIG(Phase2OT);
  extern BLFIT_MAIN_FUSED_FIT_SIG(HIonPhase1);
  extern BLFIT_MAIN_FUSED_FIT_SIG(Phase2OTStubs);

  // Extended-N refit, FUSED ladder: N = 3 .. kRefitMaxN (12), Phase2OTStubs only. The per-bin fast-fit
  // scans keep their own launch and are instantiated in the N-range TUs BrokenLineFit_refitLo/Hi.
  extern BLFIT_REFIT_SCAN_SIG(3);
  extern BLFIT_REFIT_SCAN_SIG(4);
  extern BLFIT_REFIT_SCAN_SIG(5);
  extern BLFIT_REFIT_SCAN_SIG(6);
  extern BLFIT_REFIT_SCAN_SIG(7);
  extern BLFIT_REFIT_SCAN_SIG(8);
  extern BLFIT_REFIT_SCAN_SIG(9);
  extern BLFIT_REFIT_SCAN_SIG(10);
  extern BLFIT_REFIT_SCAN_SIG(11);
  extern BLFIT_REFIT_SCAN_SIG(12);
  // One fused phase per TU: BrokenLineFit_refitFusedPrep/Solve/Out/Outlier.dev.cc.
  extern BLFIT_REFIT_FUSED_PREP_SIG();
  extern BLFIT_REFIT_FUSED_SOLVE_SIG();
  extern BLFIT_REFIT_FUSED_OUT_SIG();
  extern BLFIT_REFIT_FUSED_OUTLIER_SIG();

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_BrokenLineFitKernels_h
