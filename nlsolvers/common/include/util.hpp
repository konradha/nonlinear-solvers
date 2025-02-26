#ifndef UTIL_HPP
#define UTIL_HPP

#include <npy.hpp>
#include <iostream>

#define PROGRESS_BAR(i, total)                                                 \
  if ((i + 1) % (total / 100 + 1) == 0 || i + 1 == total) {                    \
    float progress = (float)(i + 1) / total;                                   \
    int barWidth = 70;                                                         \
    std::cout << "[";                                                          \
    int pos = barWidth * progress;                                             \
    for (int i = 0; i < barWidth; ++i) {                                       \
      if (i < pos)                                                             \
        std::cout << "=";                                                      \
      else if (i == pos)                                                       \
        std::cout << ">";                                                      \
      else                                                                     \
        std::cout << " ";                                                      \
    }                                                                          \
    std::cout << "] " << int(progress * 100.0) << "% \r";                      \
    std::cout.flush();                                                         \
    if (i + 1 == total)                                                        \
      std::cout << std::endl;                                                  \
  }


template <typename Float>
void save_to_npy(const std::string &filename, const Eigen::VectorX<Float> &data,
                 const std::vector<uint32_t> &shape) {
  std::vector<Float> vec(data.data(), data.data() + data.size());
  std::vector<uint64_t> shape_ul;
  for (const auto dim : shape)
    shape_ul.push_back(static_cast<uint64_t>(dim));
  npy::SaveArrayAsNumpy(filename, false, shape.size(), shape_ul.data(), vec);
}

template <typename Float>
Eigen::VectorX<Float> read_from_npy(const std::string &filename, std::vector<uint32_t>& shape) {
    std::vector<Float> data;
    std::vector<uint64_t> shape_ul;
    bool fortran_order;
    npy::LoadArrayFromNumpy(filename, shape_ul, fortran_order, data);
    shape.clear();
    for (const auto dim : shape_ul) {
        shape.push_back(static_cast<uint64_t>(dim));
    }
    return Eigen::Map<Eigen::VectorX<Float>>(data.data(), data.size());
}

#endif
