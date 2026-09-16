#ifndef RecoTracker_PixelSeeding_plugins_alpaka_HelixFit_h
#define RecoTracker_PixelSeeding_plugins_alpaka_HelixFit_h

#include <alpaka/alpaka.hpp>

#include <Eigen/Core>

#include "DataFormats/TrackSoA/interface/alpaka/TrackUtilities.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "RecoTracker/PixelTrackFitting/interface/FitResult.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometrySoA.h"
#include "Utilities/Cadna/interface/CadnaEigenTypes.h"

#include "CAStructures.h"

namespace riemannFit {

  // Sizing constant, not a tuning knob: `stride` below is this value, and every fit-side Eigen Map
  // stride derives from it. Changing it re-lays every per-lane buffer.
  constexpr uint32_t maxNumberOfConcurrentFits = 8 * 1024;
  constexpr uint32_t stride = maxNumberOfConcurrentFits;
  using Matrix3x4d = Eigen::Matrix<double_st, 3, 4>;
  using Map3x4d = Eigen::Map<Matrix3x4d, 0, Eigen::Stride<3 * stride, stride> >;
  using Matrix6x4f = Eigen::Matrix<float_st, 6, 4>;
  using Map6x4f = Eigen::Map<Matrix6x4f, 0, Eigen::Stride<6 * stride, stride> >;

  // Stride-parameterized fit-buffer maps. The lane stride S = the launch's concurrent-fit count: the
  // per-fit buffers pack lane l's entries at base+l with an S-lane inter-element stride, so a launch
  // that runs only K << maxNumberOfConcurrentFits fits can allocate S=K-sized buffers instead of the
  // global 8192-wide ones (the extended-N refit does exactly that, on a population of ~1k fits/call).
  // S defaults to the global stride, which is what the main fit and every other caller use.
  // hits
  template <int N>
  using Matrix3xNd = Eigen::Matrix<double_st, 3, N>;
  template <int N, uint32_t S = stride>
  using Map3xNdS = Eigen::Map<Matrix3xNd<N>, 0, Eigen::Stride<3 * S, S> >;
  template <int N>
  using Map3xNd = Map3xNdS<N, stride>;
  // errors
  template <int N>
  using Matrix6xNf = Eigen::Matrix<float_st, 6, N>;
  template <int N, uint32_t S = stride>
  using Map6xNfS = Eigen::Map<Matrix6xNf<N>, 0, Eigen::Stride<6 * S, S> >;
  template <int N>
  using Map6xNf = Map6xNfS<N, stride>;
  // fast fit
  template <uint32_t S = stride>
  using Map4dS = Eigen::Map<Vector4d, 0, Eigen::InnerStride<S> >;
  using Map4d = Map4dS<stride>;

  template <auto Start, auto End, auto Inc, class F>  //a compile-time bounded for loop
  constexpr void rolling_fits(F &&f) {
    if constexpr (Start < End) {
      f(std::integral_constant<decltype(Start), Start>());
      rolling_fits<Start + Inc, End, Inc>(f);
    }
  }

}  // namespace riemannFit

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace caExtension {
    // Forward declaration: refitExtended takes the OT-rechit source (raw OT hits + module geometry)
    // by pointer only; the full definition lives in CAExtensionKernels.h, pulled into BrokenLineFit.dev.cc.
    struct OTHitsSource;
  }  // namespace caExtension

  // Inputs for the dedup merge-or-keep-both confirm. Threaded from the merger through
  // CAHitMaskingAndMergerKernels::finalDedup so the union GBL refit reuses the merger's fit inputs
  // (hv/cm/otSource/rhoMap/bfield mirror refitUnitedTracks). A null pointer, or enable == false,
  // makes the whole confirm path inert: the fallback then drops the contested loser outright,
  // without a union refit.
  struct MergerDedupConfirmInputs {
    ::reco::TrackingRecHitConstView hv;
    ::reco::CAModulesConstView cm;
    const caExtension::OTHitsSource *otSource;  // raw OT rechit positions/errors (null => merged-only)
    const float *rhoMap;                        // BL material-map device grid (EventSetup condition)
    const float *bFieldMap;                     // normalized (Bz,Br) r-z field map (null => scalar bfield)
    float bfield;
    bool enable;  // run the union refit before a contested loser may be dropped
    int delta;    // union-hit-loss budget for the verdict
    // ---- dedup ranking/guard set, resolved at the finalDedup launch site. hv above is the hit view
    // the weighted cluster count reads through ::reco::isStub. Carried in this struct so that finalDedup
    // keeps one input argument; the merger dedup path always builds it.
    bool finderOnly;        // the fallback scans and counts candidates but never drops one
    bool rankClusters;      // length key = weighted cluster count (a stub counts 2)
    bool rankNHits;         // length key = nHits alone, skipping the nLayers primary key
    bool guardCrossArm;     // cross-arm keep-longest corner guard
    float guardVertPosMin;  // guard engage threshold on |dxy| proxy (cm)
    float guardChi2Margin;  // chi2/ndof margin the longer track must also win by (guard)
    // 0-shared fallback tuning.
    float fbNSigma2;       // fallback cov-gate width; <= 0 makes it track the shared-hit gate width
    float fbDropBound;     // |eta| bound beyond which the fallback may not drop
    bool fbEnable;         // master switch for the fallback drop
    bool fbSameCharge;     // require both members to have the same charge before confirming
    float fbAbsFloorDPhi;  // confirm box cut on |dphi|; 1e30 leaves it open
    float fbAbsFloorDQoP;  // confirm box cut on |d(q/p)|; 1e30 leaves it open
    float fbAbsFloorDCot;  // confirm box cut on |d(cot theta)|; 1e30 leaves it open
  };

  template <typename TrackerTraits>
  class HelixFit {
  public:
    using HitView = ::reco::TrackingRecHitView;
    using HitConstView = ::reco::TrackingRecHitConstView;
    using OutputSoAView = ::reco::TrackSoAView;
    using OutputHitSoAView = ::reco::TrackHitSoAView;

    using Tuples = caStructures::SequentialContainer;
    using TupleMultiplicity = caStructures::GenericContainer;

    explicit HelixFit(float bf, bool fitNas4) : bField_(bf), fitNas4_(fitNas4) {}
    ~HelixFit() { deallocate(); }

    void setBField(double bField) { bField_ = bField; }

    // Device pointer to the BL-fit Geant4 material-map grid (kSize floats), provided by the EventSetup
    // BLMaterialMap condition (copied to the device once per IOV). Set before launchBrokenLineKernels;
    // it is forwarded into Kernel_BLFit and read by prepareBrokenLineData/segmentXX0. Not owned.
    void setMaterialMap(const float *rhoMap) { rhoMap_ = rhoMap; }
    // Device pointer to the normalized (Bz,Br) r-z field map (blBFieldMap::kNValues floats), provided by
    // the EventSetup BLBFieldMap condition. When set, the fit uses a per-track hit-averaged effective
    // field (blEffectiveBField) in the curvature->pT conversion and in the MS/dE/dx momentum; when null
    // the fit uses the scalar bField at the origin everywhere. Not owned. Read unconditionally by the GBL
    // refit ladder, and by the CA main fit (Kernel_BLFit) only under fitCorrections_: the Highland
    // variance the material map feeds is divided by a momentum that only the true bending field gives,
    // so the field belongs to the same correctness package as the material map.
    // Set by the CA producers (both iterations) and by the merger refit / dedup-confirm paths.
    void setBFieldMap(const float *bMap) { bMap_ = bMap; }
    // Host copy of the tuple-multiplicity assoc's finalized per-N-bin cumulative offsets (see
    // CAHitNtupletGeneratorKernels::tupleMultiplicityOffsets), already read back by the caller at the
    // producer's acquire/produce boundary. Optional: when set, launchBrokenLineKernels uses each N-bin's
    // population to elide empty chunk/bin launches, with no D2H and no host wait of its own. Left null,
    // the elision no-ops and the fit runs to the cap (the merger refit path). Not owned; must stay valid
    // for the duration of the fit launch.
    void setHostTupleMultiplicityOffsets(const uint32_t *off) { hostTupleMultiplicityOffsets_ = off; }
    // Runtime switch for the GBL fit's in-fit smoothed-residual outlier drop (the outlierReject_ member of
    // Kernel_BLFitPhaseSolve / Kernel_BLFitPhaseOutlier). Every fit the pipeline runs has it on; with it
    // off the fit keeps every node it was handed.
    void setOutlierReject(bool on) { outlierReject_ = on; }
    // Fit correctness package (producer parameter useFitCorrections; see the block at the head of
    // RecoTracker/PixelTrackFitting/interface/alpaka/BrokenLine.h). Read only by the CA main fit; the
    // merger's GBL refit has its own scattering model and ignores it. It also gates the CA main fit's use
    // of the (Bz,Br) map set by setBFieldMap (see there): on, the fit's material AND its bending field are
    // the measured ones; off, the flat 0.06/16 material and the origin scalar field, i.e. upstream.
    void setFitCorrections(bool on) { fitCorrections_ = on; }
    // Fit-consistent curvature->pT conversion field. On: the GBL refit's effective bending field is
    // re-derived from the fit's own curvature-information weights (see blKernelWeightedBField in
    // BrokenLineFitKernels.h). Off: the field is the plain hit-count average of B_bend. Either way it
    // needs the (Bz,Br) map AND the GBL solve's influence vector, so it is inert on the CA main fit,
    // which is a factorized band solve and has no such vector: that fit always takes the hit-count
    // average.
    void setFieldKernelWeights(bool on) { fieldKernelWeights_ = on; }
    // Charge-symmetric corrections package: the arc of gblHelixAtPca's node-0 -> PCA step is signed
    // consistently with the fit's own transverse arc, and the node prep adds the bending-field PROFILE
    // deterministic offset. Off: an unsigned arc, and no profile offset. The first half needs no map;
    // the second is inert wherever setBFieldMap was not called.
    void setChargeSymmetric(bool on) { chargeSymmetric_ = on; }
    // Reference-trajectory corrections package: the GBL node builders seed the node-0 path length from
    // the reference helix and take the arc->azimuth sign of the measurement-less node from it, and the
    // field term carries its B_r lambda row beside the B_z one. Off: the node-0 path length is seeded 0,
    // the azimuth sign is taken unsigned, and only the B_z row is formed. The lambda row is inert
    // wherever setBFieldMap was not called.
    void setTrajectoryCorrections(bool on) { trajectoryCorrections_ = on; }
    // Highland's log evaluated at the track's TOTAL declared material rather than gap by gap. theta0^2 is
    // not additive over a chain of thin scatterers, so the single logarithm of the Highland form belongs at
    // the accumulated total; the resulting variance is apportioned to the gaps in proportion to their
    // thickness, leaving every gap's share and every endpoint partition weight untouched. Off: each gap
    // evaluates the logarithm at its own thickness. Read by both GBL node builders; map-independent.
    void setScatteringLogAtTotal(bool on) { scatteringLogAtTotal_ = on; }
    // Cumulative-column typical-loss law: the Landau family is stable under convolution, so the typical
    // loss of the charged column is the single-column law evaluated at the accumulated thickness (a median
    // statistic), and callers charge per-node increments of it. Off: each lump is charged its own Landau
    // MPV independently. Lump placement is the same either way. Read by both GBL node builders.
    void setCumulativeEloss(bool on) { cumulativeEloss_ = on; }
    // When true, refitMergedTwins propagates the GBL refit's single-outlier drop to the emitted
    // TrackHitSoA list (the fit-rejected hit is removed so nHits == nMeasFit for refit tracks; MTV
    // shared-fraction / the SoA->legacy converter / the cov-dedup then see the fit's actual hit set).
    // When false the rejected hit stays in the emitted list and only the fit ignores it. Merger final
    // refit only.
    void setDropOutlierFromHitList(bool on) { e3DropOutlierFromHitList_ = on; }
    // Core-protected outlier: when true, refitExtended threads a per-lane pixel-core flag table to the
    // outlier kernels so their worst-pull drop can only land on an appended extra, never on an original
    // pixel-core hit -- if the worst node is a core hit the fit abstains and drops nothing. When false the
    // drop lands on the worst node whatever it is. Merger final refit only. Composes with
    // setDropOutlierFromHitList: the protected drop is then also removed from the emitted list, so only
    // appended extras can be dropped AND removed.
    void setOutlierCoreProtect(bool on) { refitOutlierCoreProtect_ = on; }

    // Enable a one-shot device-side dump of the first N fitted tracks at the
    // end of each launchBrokenLineKernels call.  Prints (phi0, d0, kappa,
    // cotTheta, z0) as well as the derived pT [GeV] and eta.  OFF by default.
    void setVerboseDump(bool on, uint32_t nToPrint = 10) {
      verboseDump_ = on;
      verboseDumpN_ = nToPrint;
    }
    void launchRiemannKernels(const HitConstView &hv,
                              const ::reco::CAModulesConstView &fr,
                              uint32_t nhits,
                              uint32_t maxNumberOfTuples,
                              Queue &queue);
    // ONE sweep of the N-binned BLFastFit+BLFit kernels: the factorized fast BrokenLine fit that every
    // CA iteration and every topology runs on its own tracks. The General Broken Lines fit runs once per
    // track downstream, in the merger (refitExtended / refitMergedTwins below).
    void launchBrokenLineKernels(const HitConstView &hv,
                                 const ::reco::CAModulesConstView &fr,
                                 uint32_t nhits,
                                 uint32_t maxNumberOfTuples,
                                 Queue &queue);

    void allocate(TupleMultiplicity const *tupleMultiplicity,
                  OutputSoAView &helix_fit_results,
                  Tuples const *__restrict__ foundNtuplets);
    void deallocate();

    // Extended-N refit cap. N <= 12 keeps every refit launch's per-thread stack frame under the ceiling
    // at which the driver starts reserving local memory for every resident thread; do not raise it
    // without re-measuring the frame.
    static constexpr uint32_t kRefitMaxN = 12;

    // Extended-N refit concurrent-fit count = its own lane stride (see Map*S). The refit population
    // (a few thousand tracks per event at most, compacted per N-bin) is sized independently of the main
    // fit's 8192-wide stride; the hits/hits_ge Eigen maps and the gnodes/scratch per-lane buffers all
    // key off this. The fused ladder partitions these lanes across the ten N-bins per event, so this is
    // the number of tracks ONE PASS seats, not a cap: refitExtended repeats the pass until every track
    // has been seated (ceil(nTracksCap / kRefitStride) rounds always suffice, since a track takes at
    // most one lane). Raising it trades transient memory (~15 MB per in-flight call at 2048, roughly
    // linear) for fewer rounds; everything downstream (the maps, the fused grid width kFusedBlocks, the
    // round bound) is derived from it.
    static constexpr uint32_t kRefitStride = 2048;
    static_assert(kRefitStride <= riemannFit::maxNumberOfConcurrentFits);

    // Extended-N REFIT of the accepted-extended tracks (Phase2OTStubs only; a no-op otherwise). Runs
    // AFTER the merger's OT hit-attach rewrite: a full GBL fit of each accepted-extended track's rewritten
    // hit list (originals + attached extras, incl. tagged OT rechits) OVERWRITES its state/cov/chi2/
    // pt/eta/ndof (same writeback path as the standard fit). The OT lever arm shrinks the longitudinal
    // + pT covariance. hitContainer = the rewritten hit container; acceptedByTuple>=0 gates the
    // population (~1k/ev); otSource supplies the raw OT rechit positions/errors (null => merged-only).
    void refitExtended(const HitConstView &hv,
                       const ::reco::CAModulesConstView &fr,
                       caStructures::SequentialContainer const *hitContainer,
                       const int32_t *acceptedByTuple,
                       const caExtension::OTHitsSource *otSource,
                       uint32_t maxNumberOfTuples,
                       Queue &queue);

    // Merger-side twin refit (Phase2OTStubs only; no-op otherwise). Builds a SequentialContainer over
    // the MERGED track CSR (off <- tracks.hitOffsets, content aliases trackHits.id, no copy) and runs
    // refitExtended over the united-winner mask (>=0 for winners), OVERWRITING each winner's state/cov/
    // chi2/ndof with a full GBL fit of its post-union hit list (pixel + tagged OT extras). Sets tuples_/
    // outputSoa_ from the merged views directly (no allocate()); tupleMultiplicity_ is unused by the
    // refit path. mergedTracks == the output SoA (also read for hitOffsets); mergedHits supplies id.
    void refitMergedTwins(const HitConstView &hv,
                          const ::reco::CAModulesConstView &fr,
                          OutputSoAView mergedTracks,
                          OutputHitSoAView mergedHits,
                          const int32_t *unitedMask,
                          const caExtension::OTHitsSource *otSource,
                          uint32_t nTracksCap,
                          Queue &queue);

    // Dedup union refit + verdict (Phase2OTStubs only; no-op otherwise).
    // For each contested 0-shared fallback pair {loser i, partner j} captured by Kernel_dedupCovMark,
    // build the de-duplicated union of the two tracks' hit lists, run ONE batched GBL refit of the unions
    // (reusing refitExtended over a synthetic fixed-stride hit container + a scratch output SoA), and
    // DROP the loser (drop[i]=1) iff the merged fit CONFIRMS same-particle: converged (finite) AND union
    // hits outlier-dropped <= delta AND merged chi2/ndof <= max(chi2/ndof_i, chi2/ndof_j); else KEEP BOTH.
    // Unions whose fit-selected count exceeds kRefitMaxN are kept both (never truncated). tracks/trackHits
    // = the refined merged SoA (read-only: hit lists + per-track chi2/ndof); drop[] is adjusted in place.
    // contestedPairs holds 2 uint32 per slot {i, j}; an unfilled slot is tagged 0xffffffff. diag (nullable)
    // receives the verdict census in slots [12..15] (see Kernel_dedupCovMark).
    void refitDedupUnions(const HitConstView &hv,
                          const ::reco::CAModulesConstView &fr,
                          const ::reco::TrackSoAConstView &tracks,
                          const ::reco::TrackHitSoAConstView &trackHits,
                          const uint32_t *contestedPairs,
                          uint32_t nPairsCap,
                          const caExtension::OTHitsSource *otSource,
                          uint8_t *drop,
                          int delta,
                          uint32_t *diag,
                          Queue &queue);

  private:
    static constexpr uint32_t maxNumberOfConcurrentFits_ = riemannFit::maxNumberOfConcurrentFits;

    // fowarded
    Tuples const *tuples_ = nullptr;
    TupleMultiplicity const *tupleMultiplicity_ = nullptr;
    OutputSoAView outputSoa_;
    float bField_;
    const float *rhoMap_ = nullptr;          // BL material-map device grid (EventSetup condition; not owned)
    const float *bMap_ = nullptr;            // normalized (Bz,Br) r-z field map (EventSetup condition; not owned)
    bool outlierReject_ = true;              // in-fit smoothed-residual outlier drop
    bool fitCorrections_ = false;            // CA main fit's correctness package (useFitCorrections)
    bool fieldKernelWeights_ = false;        // fit-consistent curvature->pT conversion field
    bool chargeSymmetric_ = false;           // charge-symmetric corrections
    bool trajectoryCorrections_ = false;     // reference-trajectory corrections
    bool scatteringLogAtTotal_ = false;      // Highland log at the track total
    bool cumulativeEloss_ = false;           // cumulative-column typical loss
    bool e3DropOutlierFromHitList_ = false;  // remove the refit's dropped outlier from the emitted list
    bool refitOutlierCoreProtect_ = false;   // abstain when the largest-pull node is a pixel-core hit
    // Transient: per-merged-track dropped-hit-id buffer (owned by refitMergedTwins for the lifetime of
    // one refit; refitExtended forwards it to the outlier kernels, the post-refit compaction consumes it).
    uint32_t *e3DropHitId_ = nullptr;
    // Tuple-multiplicity per-N-bin cumulative offsets, pre-read by the caller into host memory (see
    // setHostTupleMultiplicityOffsets). Not owned; null on the refit path, which runs to the cap.
    const uint32_t *hostTupleMultiplicityOffsets_ = nullptr;

    // One-shot post-fit device dump of the first verboseDumpN_ tracks (see setVerboseDump). OFF by default.
    bool verboseDump_ = false;
    uint32_t verboseDumpN_ = 10;

    const bool fitNas4_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_HelixFit_h
