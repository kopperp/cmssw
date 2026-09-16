#ifndef RecoTracker_PixelTrackFitting_test_GblHostOracles_h
#define RecoTracker_PixelTrackFitting_test_GblHostOracles_h

// HOST-ONLY oracles for the GBL device tests: the DESY General-Broken-Lines library (external `gbl`,
// Kleinwort) and CMSSW's curvilinear->perigee conversion. Declared behind a POD interface and DEFINED IN
// GblHostOracles.cc, compiled into the same <bin> as the .dev.cc by the host compiler.
//
// WHY THE SPLIT. The .dev.cc is compiled by nvcc/hipcc for GPU backends. The DESY headers and CMSSW
// TrackingTools/MagneticField headers are pure host code (never compiled by nvcc in CMSSW). Rather than
// find out the hard way, they are confined to a plain .cc that scram compiles with the host compiler
// even inside an alpaka <bin> (Makefile.cuda selects only %.dev.cc for nvcc). NOTHING in this header
// pulls gbl, Eigen, alpaka or CMSSW: std types only, so the .dev.cc side stays clean.

#include <vector>
#include "Utilities/Cadna/interface/CadnaEigenTypes.h"

namespace gblHostOracles {

  // One GBL node, flattened. Mirrors generalBrokenLine::GblNodeData (both twins carry the same five
  // members in the same meaning); 5x5 and 2x2 blocks are ROW-MAJOR.
  struct Node {
    double jacToPrev[25] = {};    // curvilinear point-to-point Jacobian from the previous node (unused for node 0)
    double measPrec[4] = {};      // 2x2 measurement precision in the (U,V) offset frame
    double measResidual[2] = {};  // (U,V) residual: measured hit minus the reference-helix point
    double scatPrec[4] = {};      // 2x2 kink precision, (lambda, phi)
    bool hasMeas = false;
    bool hasScat = false;
  };

  //!< result of the DESY fit, read out at point 1 (== node 0, the PCA reference).
  struct DesyResult {
    double corr[5] = {};  // parameter corrections (q/p, lambda, phi, x_T, y_T)
    double cov[25] = {};  // their 5x5 covariance, row-major
    double chi2 = 0.;
    double lostWeight = 0.;
    int ndf = 0;
    bool ok = false;  // false = gbl::GblTrajectory::fit returned non-zero
  };

  // Run the DESY library over EXACTLY this node chain: one gbl::GblPoint per node carrying the node's own
  // jacToPrev, its measurement (residual + 2x2 precision, no projection -- the measurement directions ARE
  // the curvilinear u1,u2) and its scatterer (zero kink residual + 2x2 precision). The trajectory is built
  // with flagCurv/flagU1dir/flagU2dir = true and read out at point 1.
  //
  // The chain is an INPUT, not something this function derives: the device builds it, the test copies it
  // back, and DESY is asked only whether the SOLVER agrees. That is what makes the comparison a test of
  // gblFitPca rather than a second, independent (and inevitably stale) implementation of the builder.
  DesyResult desyFit(const std::vector<Node>& nodes);

  //!< sigma ratios ours/CMSSW for the five perigee parameters (1.0 == identical).
  struct PerigeeRatios {
    double_st d0 = 0., z0 = 0., phi = 0., theta = 0., ptRel = 0.;
    bool ok = false;
  };

  // Independent end-to-end check of the gblHelixAtPca extraction against CMSSW's trusted
  // FreeTrajectoryState -> PerigeeConversions, fed the same GBL curvilinear covariance. CMSSW's
  // ftsToPerigeeError gives the perigee AT the FTS point (the fast-fit PCA, node 0) while gblHelixAtPca
  // adds the node0 -> true-PCA propagation, so for prompt tracks the two coincide and every ratio is ~1.
  // \param fastFit (cx, cy, R, cotTheta-encoded 4th slot) -- the GBL reference helix.
  // \param corr / cov the GBL result at node 0; \param helixPar / helixCov what gblHelixAtPca produced.
  PerigeeRatios perigeeRatios(const double_st fastFit[4],
                              int qCharge,
                              double bField,
                              double_st sTransverse0,
                              double_st hitZ0,
                              const double_st corr[5],
                              const double_st cov[25],
                              const double_st helixPar[5],
                              const double_st helixCov[25]);

}  // namespace gblHostOracles

#endif
