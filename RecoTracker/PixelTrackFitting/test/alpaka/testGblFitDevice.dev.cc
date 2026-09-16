// testGblFitDevice.dev.cc
//
// Device-twin check of the alpaka GeneralBrokenLine functions (curvilinearJacobian, gblFitPca,
// gblHelixAtPca): on every available backend they must reproduce the CPU twin
// (interface/GeneralBrokenLine.h, itself validated against the DESY GBL library by testGblOracleDevice)
// to 1e-9 on a small synthetic node chain. This compile-checks the device code under the alpaka toolchain
// and confirms device math == host math. Every measurement node carries a non-zero residual and the host
// references are asserted non-zero before the relative comparisons, so no check can pass vacuously.
//
// Build with scram b; one binary per enabled alpaka backend (also run by scram b runtests).

#include <cmath>
#include <cstdio>
#include <vector>

#include <alpaka/alpaka.hpp>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoTracker/PixelTrackFitting/interface/GeneralBrokenLine.h"         // CPU twin (host reference)
#include "RecoTracker/PixelTrackFitting/interface/alpaka/GeneralBrokenLine.h"  // device twin under test

#include "Utilities/Cadna/interface/CadnaEigenTypes.h"
#include "Utilities/Cadna/interface/CadnaOutput.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;
// NB: `using namespace ALPAKA_ACCELERATOR_NAMESPACE` pulls the device twin's generalBrokenLine into scope, so the
// bare name is ambiguous -- qualify the host (CPU twin) alias with the global scope `::`.
namespace gblh = ::generalBrokenLine;                              // host (CPU twin)
namespace gbld = ALPAKA_ACCELERATOR_NAMESPACE::generalBrokenLine;  // device twin

namespace {
  constexpr int kN = 4;  // hits -> nNodes = kN + 1
}

// runs both device functions once and writes the results (gblFitPca 5x5 then curvilinearJacobian 5x5) to out[0..49].
struct GblKernel {
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, gbld::GblNodeData const* nodes, double_st const* seg, double_st* out) const {
    for ([[maybe_unused]] auto i : cms::alpakatools::uniform_elements(acc, 1)) {
      gbld::Vector5d gcorr;
      double_st fd[2 * kN + 3];
      double_st chi2d = 0.;
      // The device gblFitPca takes its bordered-band solver scratch (Mb/cbor/w/ms) explicitly; the
      // production kernel passes a per-fit device-buffer block, the test a local one (STRIDE 1).
      double_st gblScratch[gbld::kGblScratchDoubles<kN>];
      const auto cov = gbld::gblFitPca<Acc1D, kN>(acc, nodes, gblScratch, &gcorr, fd, &chi2d);
      out[86 + 2 * kN + 3] = chi2d;  // native GBL chi2 (after the fullDelta block out[86..86+2kN+2])
      for (int a = 0; a < 5; ++a)
        for (int b = 0; b < 5; ++b)
          out[a * 5 + b] = cov(a, b);
      const gbld::Vector3d p1(seg[0], seg[1], seg[2]), p2(seg[3], seg[4], seg[5]);
      const gbld::Vector3d dx(seg[6], seg[7], seg[8]), h(seg[9], seg[10], seg[11]);
      const auto jac = gbld::curvilinearJacobian(acc, p1, p2, dx, seg[12], h, seg[13]);
      for (int a = 0; a < 5; ++a)
        for (int b = 0; b < 5; ++b)
          out[25 + a * 5 + b] = jac(a, b);
      // corrections (out[50..54]) + full gblHelixAtPca extraction (helixPar out[55..59], helixCov out[60..84]).
      for (int a = 0; a < 5; ++a)
        out[50 + a] = gcorr(a);
      Eigen::Vector<double_st, 4> ff(seg[14], seg[15], seg[16], seg[17]);
      gbld::Vector5d hp;
      gbld::Matrix5d hc;
      gbld::gblHelixAtPca(acc, ff, int(seg[18]), seg[19], seg[20], seg[21], gcorr, cov, hp, hc);
      for (int a = 0; a < 5; ++a)
        out[55 + a] = hp(a);
      for (int a = 0; a < 5; ++a)
        for (int b = 0; b < 5; ++b)
          out[60 + a * 5 + b] = hc(a, b);
      // the full solution vector fullDelta (out[86..86+2kN+2]).
      for (int a = 0; a < 2 * kN + 3; ++a)
        out[86 + a] = fd[a];
    }
  }
};

TEST_CASE("GeneralBrokenLine device twin vs CPU twin for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend",
          "[" EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) "]") {
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty())
    FAIL("No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, test skipped.");

  cadna_init(-1); // -1 enables default stochastic rounding

  // ---- one representative segment Jacobian (artificial but valid: invertible blocks, SPD system) ----
  Eigen::Vector<double_st, 3> p1(0.90, 0.10, 0.42);
  const int in_digits = p1.coeff(0).nb_significant_digit();
  p1.normalize();
  Eigen::Vector<double_st, 3> p2(0.88, 0.14, 0.42);
  p2.normalize();
  const Eigen::Vector<double_st, 3> dxv(-3.0, -0.3, -1.4);
  const Eigen::Vector<double_st, 3> hv(0., 0., 0.0113921);
  const double_st qbp = -1. / 50., sstep = 4.0;
  const auto jac = gblh::curvilinearJacobian(p1, p2, dxv, qbp, hv, sstep);

  Eigen::Matrix<double_st, 2, 2> measP;
  measP << 1.0e6, 0., 0., 1.0e4;
  Eigen::Matrix<double_st, 2, 2> scatP;
  scatP << 1.0e7, 0., 0., 1.2e7;

  // build identical node arrays for host (vector) and device (host buffer of the device struct).
  std::vector<gblh::GblNodeData> hostNodes(kN + 1);
  auto devNodes_h = cms::alpakatools::make_host_buffer<gbld::GblNodeData[], Platform>(kN + 1);
  for (int k = 0; k <= kN; ++k) {
    gbld::GblNodeData d;  // node 0 (PCA): default, no measurement/scatterer
    if (k >= 1) {
      // deterministic, node-dependent and non-zero on every measurement node (see the file head).
      const Eigen::Vector<double_st, 2> resid(1.5e-3 * static_cast<double_st>(k), -0.8e-3 * static_cast<double_st>(k + 1));
      hostNodes[k].jacToPrev = jac;
      hostNodes[k].measPrec = measP;
      hostNodes[k].measResidual = resid;
      hostNodes[k].hasMeas = true;
      d.jacToPrev = jac;
      d.measPrec = measP;
      d.measResidual = resid;
      d.hasMeas = true;
    }
    if (k >= 1 && k <= kN - 1) {
      hostNodes[k].scatPrec = scatP;
      hostNodes[k].hasScat = true;
      d.scatPrec = scatP;
      d.hasScat = true;
    }
    devNodes_h[k] = d;
  }
  gblh::Vector5d hostCorr;
  Eigen::Vector<double_st, Eigen::Dynamic> hostFd;
  double_st hostChi2 = 0.;
  const auto hostCov = gblh::gblFitPca(hostNodes, &hostCorr, &hostFd, &hostChi2);
  // host gblHelixAtPca on a made-up (but valid) fast-fit + reference, to validate the device value path vs the CPU.
  const Eigen::Vector4d ff(5.0, 3.0, 100.0, 2.4);
  const int qCh = 1;
  const double_st bF = 0.0113921, sT0 = 2.0, hZ0 = 1.0;
  gblh::Vector5d hostHp;
  gblh::Matrix5d hostHc;
  gblh::gblHelixAtPca(ff, qCh, bF, sT0, hZ0, hostCorr, hostCov, hostHp, hostHc);
  // POSITIVE CONTROLS: the relative comparisons below skip any reference that is numerically zero, so a
  // reference of all zeros passes them all. Assert that the fit actually moved before comparing to it.
  REQUIRE(hostCorr.cwiseAbs().maxCoeff() > 1e-30);
  REQUIRE(hostFd.cwiseAbs().maxCoeff() > 1e-30);
  REQUIRE(hostHp.cwiseAbs().maxCoeff() > 1e-30);
  REQUIRE(abs(hostChi2) > 1e-30);

  // seg inputs for the device curvilinearJacobian + gblHelixAtPca checks.
  constexpr int kSeg = 22;
  auto seg_h = cms::alpakatools::make_host_buffer<double_st[], Platform>(kSeg);
  for (int j = 0; j < 3; ++j) {
    seg_h[j] = p1[j];
    seg_h[3 + j] = p2[j];
    seg_h[6 + j] = dxv[j];
    seg_h[9 + j] = hv[j];
  }
  seg_h[12] = qbp;
  seg_h[13] = sstep;
  for (int j = 0; j < 4; ++j)
    seg_h[14 + j] = ff[j];
  seg_h[18] = qCh;
  seg_h[19] = bF;
  seg_h[20] = sT0;
  seg_h[21] = hZ0;

  for (auto const& device : cms::alpakatools::devices<Platform>()) {
    auto queue = Queue(device);
    constexpr int kOut = 86 + 2 * kN + 3 + 1;  // 86 prior + fullDelta(2kN+3) + chi2(1)
    auto nodes_d = cms::alpakatools::make_device_buffer<gbld::GblNodeData[]>(queue, kN + 1);
    auto seg_d = cms::alpakatools::make_device_buffer<double_st[]>(queue, kSeg);
    auto out_d = cms::alpakatools::make_device_buffer<double_st[]>(queue, kOut);
    auto out_h = cms::alpakatools::make_host_buffer<double_st[], Platform>(kOut);
    alpaka::memcpy(queue, nodes_d, devNodes_h);
    alpaka::memcpy(queue, seg_d, seg_h);
    auto div = cms::alpakatools::make_workdiv<Acc1D>(1, 1);
    alpaka::exec<Acc1D>(queue, div, GblKernel{}, nodes_d.data(), seg_d.data(), out_d.data());
    alpaka::memcpy(queue, out_h, out_d);
    alpaka::wait(queue);

    std::cout << std::endl;
    std::cout << "-- GENERAL BROKEN LINE (OVERALL FIT) --\n";
    print_cadna_metric("gblFitPca cov", hostCov(0), in_digits);
    print_cadna_metric("gblFitPca corr", hostCorr(0), in_digits);
    print_cadna_metric("curvJac", jac(0, 0), in_digits);
    print_cadna_metric("gblHelixAtPca par", hostHp(0), in_digits);
    print_cadna_metric("gblHelixAtPca cov", hostHc(0, 0), in_digits);
    print_cadna_metric("fullDelta", hostFd(0), in_digits);
    print_cadna_metric("chi2", hostChi2, in_digits);

    double_st covMaxRel = 0., jacMaxRel = 0., corrMaxRel = 0., hpMaxRel = 0., hcMaxRel = 0.;
    for (int a = 0; a < 5; ++a) {
      if (abs(static_cast<double>(hostCorr(a))) > 1e-30) {
        corrMaxRel = fmax(corrMaxRel, abs(out_h[50 + a] - hostCorr(a)) / abs(hostCorr(a)));
      }
      if (abs(static_cast<double>(hostHp(a))) > 1e-30)
        hpMaxRel = fmax(hpMaxRel, abs(out_h[55 + a] - hostHp(a)) / abs(hostHp(a)));
      for (int b = 0; b < 5; ++b) {
        const double_st rc = hostCov(a, b), rj = jac(a, b), rh = hostHc(a, b);
        if (abs(static_cast<double>(rc)) > 1e-30)
          covMaxRel = fmax(covMaxRel, abs(out_h[a * 5 + b] - rc) / abs(rc));
        if (abs(static_cast<double>(rj)) > 1e-30)
          jacMaxRel = fmax(jacMaxRel, abs(out_h[25 + a * 5 + b] - rj) / abs(rj));
        if (abs(static_cast<double>(rh)) > 1e-30)
          hcMaxRel = fmax(hcMaxRel, abs(out_h[60 + a * 5 + b] - rh) / abs(rh));
      }
    }
    double_st fdMaxRel = 0.;
    for (int a = 0; a < 2 * kN + 3; ++a)
      if (abs(hostFd(a)) > 1e-30)
        fdMaxRel = fmax(fdMaxRel, abs(out_h[86 + a] - hostFd(a)) / abs(hostFd(a)));
    const double_st chi2Rel =
        (abs(hostChi2) > 1e-30) ? abs(out_h[86 + 2 * kN + 3] - hostChi2) / abs(hostChi2) : static_cast<double_st>(0.);
    std::printf(
        "  %s: gblFitPca cov=%.2e corr=%.2e | curvJac=%.2e | gblHelixAtPca par=%.2e cov=%.2e | fullDelta=%.2e "
        "chi2=%.2e\n",
        alpaka::getName(device).c_str(),
        static_cast<double>(covMaxRel),
        static_cast<double>(corrMaxRel),
        static_cast<double>(jacMaxRel),
        static_cast<double>(hpMaxRel),
        static_cast<double>(hcMaxRel),
        static_cast<double>(fdMaxRel),
        static_cast<double>(chi2Rel));

    REQUIRE(covMaxRel < 1e-9);
    REQUIRE(jacMaxRel < 1e-9);
    REQUIRE(corrMaxRel < 1e-9);
    REQUIRE(hpMaxRel < 1e-9);
    REQUIRE(hcMaxRel < 1e-9);
    REQUIRE(fdMaxRel < 1e-9);
    // chi2 = chi2ref - delta.b is a cancellation of two large nearly-equal numbers when the fit is
    // good (here ~2e6), so it loses digits the covariance does not: the best relative precision any
    // arithmetic can carry is eps * 2e6 ~ 4e-10, which the serial backend sits at and CUDA (FMA
    // contraction) reaches within a few x. 1e-7 keeps ample headroom while catching any real divergence.
    REQUIRE(chi2Rel < 1e-7);

    cadna_end();
  }
}
