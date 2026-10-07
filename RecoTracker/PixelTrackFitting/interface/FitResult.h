#ifndef RecoTracker_PixelTrackFitting_interface_FitResult_h
#define RecoTracker_PixelTrackFitting_interface_FitResult_h

#include <cmath>
#include <cstdint>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#pragma GCC diagnostic pop

#include "Utilities/Cadna/interface/CadnaEigenTypes.h"

namespace riemannFit {

  using Vector2d = Eigen::Vector<double_st, 2>;
  using Vector3d = Eigen::Vector<double_st, 3>;
  using Vector4d = Eigen::Vector<double_st, 4>;
  using Vector5d = Eigen::Matrix<double_st, 5, 1>;
  using Matrix2d = Eigen::Matrix<double_st, 2, 2>;
  using Matrix3d = Eigen::Matrix<double_st, 3, 3>;
  using Matrix4d = Eigen::Matrix<double_st, 4, 4>;
  using Matrix5d = Eigen::Matrix<double_st, 5, 5>;
  using Matrix6d = Eigen::Matrix<double_st, 6, 6>;


  using Vector2f = Eigen::Vector<float_st, 2>;
  using Vector3f = Eigen::Vector<float_st, 3>;
  using Vector4f = Eigen::Vector<float_st, 4>;
  using Vector5f = Eigen::Matrix<float_st, 5, 1>;
  using Matrix2f = Eigen::Matrix<float_st, 2, 2>;
  using Matrix3f = Eigen::Matrix<float_st, 3, 3>;
  using Matrix4f = Eigen::Matrix<float_st, 4, 4>;
  using Matrix5f = Eigen::Matrix<float_st, 5, 5>;
  using Matrix6f = Eigen::Matrix<float_st, 6, 6>;

  template <int N>
  using Matrix3xNd = Eigen::Matrix<double_st, 3, N>;  // used for inputs hits

  template <int N>
  using Matrix3xNf = Eigen::Matrix<float_st, 3, N>;  // used for inputs hits

  struct CircleFit {
    Vector3f par;  //!< parameter: (X0,Y0,R)
    Matrix3f cov;
    /*!< covariance matrix: \n
      |cov(X0,X0)|cov(Y0,X0)|cov( R,X0)| \n
      |cov(X0,Y0)|cov(Y0,Y0)|cov( R,Y0)| \n
      |cov(X0, R)|cov(Y0, R)|cov( R, R)|
    */
    int32_t qCharge;  //!< particle charge
    float_st chi2;
  };

  struct LineFit {
    Vector2f par;  //!<(cotan(theta),Zip)
    Matrix2f cov;
    /*!<
      |cov(c_t,c_t)|cov(Zip,c_t)| \n
      |cov(c_t,Zip)|cov(Zip,Zip)|
    */
    double_st chi2;
  };

  struct HelixFit {
    Vector5f par;  //!<(phi,Tip,pt,cotan(theta)),Zip)
    Matrix5f cov;
    /*!< ()->cov() \n
      |(phi,phi)|(Tip,phi)|(p_t,phi)|(c_t,phi)|(Zip,phi)| \n
      |(phi,Tip)|(Tip,Tip)|(p_t,Tip)|(c_t,Tip)|(Zip,Tip)| \n
      |(phi,p_t)|(Tip,p_t)|(p_t,p_t)|(c_t,p_t)|(Zip,p_t)| \n
      |(phi,c_t)|(Tip,c_t)|(p_t,c_t)|(c_t,c_t)|(Zip,c_t)| \n
      |(phi,Zip)|(Tip,Zip)|(p_t,Zip)|(c_t,Zip)|(Zip,Zip)|
    */
    float_st chi2_circle;
    float_st chi2_line;
    //    Vector4d fast_fit;
    int32_t qCharge;  //!< particle charge
  };  // __attribute__((aligned(16)));

}  // namespace riemannFit

#endif  // RecoTracker_PixelTrackFitting_interface_FitResult_h
