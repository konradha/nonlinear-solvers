#include "libnpy/include/npy.hpp"

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
  //for (const auto dim : shape)
  //  std::cout << dim << " ";
  //std::cout << "\n"
  //          << "shape size: " << shape.size() << "\n";
  npy::SaveArrayAsNumpy(filename, false, shape.size(), shape_ul.data(), vec);
}

