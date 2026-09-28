#ifndef DataFormats_Cadna_CadnaEigenTypes_h
#define DataFormats_Cadna_CadnaEigenTypes_h

#include <cadna.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <Eigen/Core>
#pragma GCC diagnostic pop

namespace Eigen {

template<> struct NumTraits<double_st>
 : NumTraits<double> // permits to get the epsilon, dummy_precision, lowest, highest functions
{
  typedef double_st Real;
  typedef double_st NonInteger;
  typedef double_st Nested;

  enum {
    IsComplex = 0,
    IsInteger = 0,
    IsSigned = 1,
    RequireInitialization = 1,
    ReadCost = 1,
    AddCost = 3,
    MulCost = 3
  };
};

template<> struct NumTraits<float_st>
 : NumTraits<double> // permits to get the epsilon, dummy_precision, lowest, highest functions
{
  typedef float_st Real;
  typedef float_st NonInteger;
  typedef float_st Nested;

  enum {
    IsComplex = 0,
    IsInteger = 0,
    IsSigned = 1,
    RequireInitialization = 1,
    ReadCost = 1,
    AddCost = 3,
    MulCost = 3
  };
};

// Inform Eigen how to handle mixing standard literals with CADNA
template <typename BinaryOp>
struct ScalarBinaryOpTraits<double_st, double, BinaryOp> {
  typedef double_st ReturnType;
};

template <typename BinaryOp>
struct ScalarBinaryOpTraits<double, double_st, BinaryOp> {
  typedef double_st ReturnType;
};

template <typename BinaryOp>
struct ScalarBinaryOpTraits<double_st, int, BinaryOp> {
  typedef double_st ReturnType;
};

template <typename BinaryOp>
struct ScalarBinaryOpTraits<int, double_st, BinaryOp> {
  typedef double_st ReturnType;
};

} // namespace Eigen
#endif
