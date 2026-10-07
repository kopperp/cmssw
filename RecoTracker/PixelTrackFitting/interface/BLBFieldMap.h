// Solenoid (Bz,Br) r-z lattice for the BL curvature->pT conversion.
// The fit solves a geometric curvature of the transverse projection; converting to pT needs
//   B_bend(r,z) = Bz(r,z) - Br(r,z)*tanLambda*cos(alpha),
// where cos(alpha) is the track-radial cosine. Both terms matter for forward tracks: Bz falls ~7.5%
// at |z| = 2.7 m and |Br/Bz| reaches 1.8% in the OT endcap. Both components are stored normalized
// to Bz(0,0), sampled on the lattice below from the MagneticField EventSetup product.
#ifndef RecoTracker_PixelTrackFitting_BLBFieldMap_h
#define RecoTracker_PixelTrackFitting_BLBFieldMap_h

#include "Utilities/Cadna/interface/CadnaEigenTypes.h"

namespace blBFieldMap {
  // Sampling lattice, uniform in r and z (5 cm r, 10 cm z), covering the tracking volume; clamped at the boundary.
  constexpr int kNR = 24;
  constexpr int kNZ = 57;
  constexpr float kRMax = 115.0f;
  constexpr float kZMax = 280.0f;
  constexpr float kInvDR = float(kNR - 1) / float(kRMax);
  constexpr float kInvDZ = float(kNZ - 1) / (2. * float(kZMax));
  // Buffer layout: kNNodes normalized Bz values (r-major: index ir*kNZ + iz), then kNNodes normalized Br values.
  constexpr int kNNodes = kNR * kNZ;
  constexpr int kNValues = 2 * kNNodes;

  // Bilinear interpolation of one component inside the cell whose lower corner is the flat index i00.
  constexpr inline float bilinearAt(const float* comp, int i00, float tr, float tz) {
    const float b0 = float(comp[i00]) + tz * (float(comp[i00 + 1]) - float(comp[i00]));
    const float b1 = float(comp[i00 + kNZ]) + tz * (float(comp[i00 + kNZ + 1]) - float(comp[i00 + kNZ]));
    return b0 + tr * (b1 - b0);
  }

  // Normalized B_bend(r,z)/Bz(0,0): Bz - Br*tanLambda*cos(alpha), r/z clamped to the box. constexpr (device-callable).
  constexpr inline float bBendAt(const float* map, float r, float z, float tanLambdaCosAlpha) {
    const float rc = r < 0. ? 0. : (r > float(kRMax) ? float(kRMax) : r);
    const float zc = z < -float(kZMax) ? -float(kZMax) : (z > float(kZMax) ? float(kZMax) : z);
    const float fr = rc * kInvDR;
    const float fz = (zc + float(kZMax)) * kInvDZ;
    int ir = int(fr);
    ir = ir < 0 ? 0 : (ir > kNR - 2 ? kNR - 2 : ir);
    int iz = int(fz);
    iz = iz < 0 ? 0 : (iz > kNZ - 2 ? kNZ - 2 : iz);
    const float tr = fr - float(ir);
    const float tz = fz - float(iz);
    const int i00 = ir * kNZ + iz;
    return bilinearAt(map, i00, tr, tz) - bilinearAt(map + kNNodes, i00, tr, tz) * tanLambdaCosAlpha;
  }
  // B_bend and the normalized B_r; the index block repeats bBendAt (hot path), keep the two in sync.
  // The lambda slope of dT/ds = (q/|p|)(T x B) is -q/|p|*B_r*sin(alpha), and vanishes at B_r = 0.
  constexpr inline float bBendAndBrAt(const float* map, float r, float z, float tanLambdaCosAlpha, float& brNorm) {
    const float rc = r < 0. ? 0. : (r > float(kRMax) ? float(kRMax) : r);
    const float zc = z < -float(kZMax) ? -float(kZMax) : (z > float(kZMax) ? float(kZMax) : z);
    const float fr = rc * kInvDR;
    const float fz = (zc + float(kZMax)) * kInvDZ;
    int ir = int(fr);
    ir = ir < 0 ? 0 : (ir > kNR - 2 ? kNR - 2 : ir);
    int iz = int(fz);
    iz = iz < 0 ? 0 : (iz > kNZ - 2 ? kNZ - 2 : iz);
    const float tr = fr - float(ir);
    const float tz = fz - float(iz);
    const int i00 = ir * kNZ + iz;
    brNorm = bilinearAt(map + kNNodes, i00, tr, tz);
    return bilinearAt(map, i00, tr, tz) - brNorm * tanLambdaCosAlpha;
  }
}  // namespace blBFieldMap
#endif
