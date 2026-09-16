#ifndef RecoTracker_PixelTrackFitting_interface_alpaka_BandedSolve_h
#define RecoTracker_PixelTrackFitting_interface_alpaka_BandedSolve_h

// Symmetric banded linear algebra for the device fits: root-free LDL^T factorization of a SYMMETRIC BAND
// matrix and in-place application of its inverse to a RHS. Pure scalar loops, no dependency beyond
// alpaka -- kept out of GeneralBrokenLine.h so the factorized fast-BL (BrokenLine.h) does not pull in
// the GBL numerics, the B-field map and the perigee transform.
//
// Lower-packed storage: Mb[(i*(kBand+1)+p)*lstride] = M(i, i-p), p = 0..min(i,kBand). bandFactor overwrites
// Mb with its root-free LDL^T (Mb[..+0] = D_i, Mb[..+p] = L(i,i-p)); bandSolveInPlace applies M^-1 in place.
// O(n) storage, O(n*band^2) work vs dense O(N^2)/O(N^3). Band half-width kBand (GBL: kGblBand=5; fast-BL:
// kBand=2); element stride lstride (1 = contiguous, lane count for [element][lane] fit scratch). Raw scalar
// loops -- no Eigen products in the solves.

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "Utilities/Cadna/interface/CadnaEigenTypes.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::generalBrokenLine {

  template <int kBand>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void bandFactor(double_st* Mb, int n, int lstride = 1) {
    const int rstride = (kBand + 1) * lstride;  // physical stride between packed rows
    for (int j = 0; j < n; ++j) {
      double_st* Mj = Mb + j * rstride;
      double_st Dj = Mj[0];
      const int pmax = (j < kBand) ? j : kBand;
      for (int p = 1; p <= pmax; ++p)
        Dj -= Mj[p * lstride] * Mj[p * lstride] * Mb[(j - p) * rstride];
      Mj[0] = Dj;
      const int imax = (j + kBand < n - 1) ? (j + kBand) : (n - 1);
      for (int i = j + 1; i <= imax; ++i) {
        double_st* Mi = Mb + i * rstride;
        double_st sum = Mi[(i - j) * lstride];  // A(i, j)
        const int kmin = (i - kBand > 0) ? (i - kBand) : 0;
        for (int k = kmin; k < j; ++k)
          sum -= Mi[(i - k) * lstride] * Mj[(j - k) * lstride] * Mb[k * rstride];
        Mi[(i - j) * lstride] = sum / Dj;  // L(i, j)
      }
    }
  }

  // iFirst/iStop are the SPARSITY WINDOW of a unit-column (or leading-zero) RHS; they skip steps whose
  // result is provably the +0 already in x: iFirst (caller guarantees x[i]==+0 for i<iFirst) starts the
  // forward and diagonal sweeps there; iStop (caller reads only x[i>=iStop]) ends the back sweep there,
  // since back substitution reads only strictly higher indices. A non-positive pivot -- impossible for the
  // SPD GBL matrix (a fit producing one is already non-finite-guarded downstream) -- would leave -0
  // instead of +0. Defaults 0/0 = the full solve.
  template <int kBand>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void bandSolveInPlace(
      const double_st* Mb, double_st* x, int n, int lstride = 1, int iFirst = 0, int iStop = 0) {
    const int rstride = (kBand + 1) * lstride;  // physical stride between packed rows
    for (int i = iFirst; i < n; ++i) {          // forward: L y = rhs
      const double_st* Mi = Mb + i * rstride;
      const int pmax = (i < kBand) ? i : kBand;
      double_st si = x[i * lstride];
      for (int p = 1; p <= pmax; ++p)
        si -= Mi[p * lstride] * x[(i - p) * lstride];
      x[i * lstride] = si;
    }
    const int iDiag = iFirst;
    for (int i = iDiag; i < n; ++i)  // diagonal: D z = y
      x[i * lstride] /= Mb[i * rstride];
    for (int i = n - 1; i >= iStop; --i) {  // back: L^T x = z
      const int kmax = (i + kBand < n - 1) ? (i + kBand) : (n - 1);
      double_st si = x[i * lstride];
      for (int k = i + 1; k <= kmax; ++k)
        si -= Mb[k * rstride + (k - i) * lstride] * x[k * lstride];
      x[i * lstride] = si;
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::generalBrokenLine

#endif  // RecoTracker_PixelTrackFitting_interface_alpaka_BandedSolve_h
