#ifndef DataFormats_Cadna_CadnaOutput_h
#define DataFormats_Cadna_CadnaOutput_h

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>
#include <concepts>
#include <type_traits>

#include <cadna.h>
#include <Eigen/Core>

template <typename T>
concept CadnaScalar = requires(std::remove_cvref_t<T> a) {
  { a.nb_significant_digit() } -> std::integral;
};

template <typename T>
concept EigenDense = std::derived_from<std::decay_t<T>, Eigen::DenseBase<std::decay_t<T>>>;

// --- Original Scalar Functions ---

template <CadnaScalar T, CadnaScalar U>
inline void print_cadna_metric(const std::string_view label,
                               const T& val1,
                               int in_digits1,
                               const U& val2,
                               int in_digits2,
                               int val_w = 25,
                               int dig_w = 2) {
  std::ostringstream oss1, oss2;
  oss1 << val1;
  oss2 << val2;

  const int sig1 = val1.nb_significant_digit();
  const int sig2 = val2.nb_significant_digit();

  std::cout << "[Accuracy] " << std::left << std::setw(20) << label << ": "
            << std::right << std::setw(val_w) << oss1.str()
            << " [" << std::setw(dig_w) << sig1 << " digits, lost "
            << std::setw(dig_w) << (in_digits1 >= 0 ? std::to_string(in_digits1 - sig1) : "--")
            << "], BrokenLine: "
            << std::right << std::setw(val_w) << oss2.str()
            << " [" << std::setw(dig_w) << sig2 << " digits, lost "
            << std::setw(dig_w) << (in_digits2 >= 0 ? std::to_string(in_digits2 - sig2) : "--")
            << "]\n";
}

template <CadnaScalar T>
inline void print_cadna_metric(const std::string_view label,
                               const T& val,
                               int in_digits,
                               int val_w = 25,
                               int dig_w = 2) {
  std::ostringstream oss;
  oss << val;

  const int sig = val.nb_significant_digit();

  std::cout << "[Accuracy] " << std::left << std::setw(20) << label << ": "
            << std::right << std::setw(val_w) << oss.str()
            << " [" << std::setw(dig_w) << sig << " digits, lost "
            << std::setw(dig_w) << (in_digits >= 0 ? std::to_string(in_digits - sig) : "--")
            << "]\n";
}

template <CadnaScalar T, std::floating_point F>
inline void print_cadna_metric(const std::string_view label,
                               const T& val,
                               F value,
                               int val_w = 25,
                               int dig_w = 2) {
  std::ostringstream oss;
  oss << val;

  const int sig = val.nb_significant_digit();

  std::cout << "[Accuracy] " << std::left << std::setw(20) << label << ": "
            << std::right << std::setw(val_w) << oss.str()
            << " [" << std::setw(dig_w) << sig << " digits, value "
            << std::setw(dig_w) << std::to_string(value)
            << "]\n";
}

// --- Eigen Overloads ---

template <EigenDense T, EigenDense U>
inline void print_cadna_metric(const std::string_view label,
                               const T& val1,
                               int in_digits1,
                               const U& val2,
                               int in_digits2,
                               int val_w = 25,
                               int dig_w = 2) {
  for (Eigen::Index row = 0; row < val1.rows(); ++row) {
    for (Eigen::Index col = 0; col < val1.cols(); ++col) {
      std::string elem_label = std::string(label) + "(" + std::to_string(row) + "," + std::to_string(col) + ")";
      print_cadna_metric(elem_label, val1(row, col), in_digits1, val2(row, col), in_digits2, val_w, dig_w);
    }
  }
}

template <EigenDense T>
inline void print_cadna_metric(const std::string_view label,
                               const T& val,
                               int in_digits,
                               int val_w = 25,
                               int dig_w = 2) {
  for (Eigen::Index row = 0; row < val.rows(); ++row) {
    for (Eigen::Index col = 0; col < val.cols(); ++col) {
      std::string elem_label = std::string(label) + "(" + std::to_string(row) + "," + std::to_string(col) + ")";
      print_cadna_metric(elem_label, val(row, col), in_digits, val_w, dig_w);
    }
  }
}

template <typename Derived>
void print_cadna_matrix(const std::string_view label,
                        const Eigen::DenseBase<Derived>& matrix,
                        int in_digits,
                        int val_w = 25,
                        int dig_w = 2) {
  for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
    for (Eigen::Index col = 0; col < matrix.cols(); ++col) {
      std::ostringstream oss;
      oss << matrix(row, col);
      const int sig = matrix(row, col).nb_significant_digit();

      std::cout << "[Accuracy] " << std::left << std::setw(20) << label << ": "
                << std::right << std::setw(val_w) << oss.str()
                << " [" << std::setw(dig_w) << sig << " digits, value "
                << std::setw(dig_w) << (in_digits >= 0 ? std::to_string(in_digits - sig) : "--")
                << "]\n";
      std::cout << matrix(row, col) << (col + 1 == matrix.cols() ? "" : " ");
    }
    std::cout << '\n';
  }
}

#endif
