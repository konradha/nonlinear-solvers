#ifndef DEVICE_SPMV_HPP
#define DEVICE_SPMV_HPP

#include "pragmas.hpp"
// also includes relevant headers for now; no linking -- we just generate huge
// binaries for now
// TODO: refactor and introduce a nice build process

template <typename T> class DeviceSpMV {
private:
  cusparseHandle_t handle_;
  cusparseSpMatDescr_t mat_desc_;
  cusparseDnVecDescr_t vec_x_, vec_y_;
  T *d_x_, *d_y_;
  void *buffer_{nullptr};
  uint32_t n_;
  uint32_t nnz_;
  using cuda_scalar_t = std::conditional_t<
      std::is_same_v<T, std::complex<float>>, cuComplex,
      std::conditional_t<std::is_same_v<T, std::complex<double>>,
                         cuDoubleComplex, T>>;

  cudaDataType cuda_data_type() const {
    if constexpr (std::is_same_v<T, float>)
      return CUDA_R_32F;
    if constexpr (std::is_same_v<T, double>)
      return CUDA_R_64F;
    if constexpr (std::is_same_v<T, std::complex<float>>)
      return CUDA_C_32F;
    if constexpr (std::is_same_v<T, std::complex<double>>)
      return CUDA_C_64F;
    throw std::runtime_error("Unsupported type");
  }

public:
  // we pass host mem pointers that describe a matrix in CSR format
  // (ie. it's coalesced to CSR before constructing this matrix)
  // TODO: We might enhance performance by changing the order of the data
  DeviceSpMV(const int *d_offsets, const int *d_columns, const T *d_values,
             uint32_t n, uint32_t nnz)
      : n_(n), nnz_(nnz) {

    CHECK_CUSPARSE(cusparseCreate(&handle_));
    CHECK_CUSPARSE(cusparseCreateCsr(
        &mat_desc_, n, n, nnz, (void *)d_offsets, (void *)d_columns,
        const_cast<cuda_scalar_t *>(
            reinterpret_cast<const cuda_scalar_t *>(d_values)),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
        cuda_data_type()));

    CHECK_CUDA(cudaMalloc(&d_x_, n_ * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_y_, n_ * sizeof(T)));

    CHECK_CUSPARSE(cusparseCreateDnVec(&vec_x_, n, d_x_, cuda_data_type()));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vec_y_, n, d_y_, cuda_data_type()));
    size_t buffer_size;
    T alpha{1}, beta{0};
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
        reinterpret_cast<cuda_scalar_t *>(&alpha), mat_desc_, vec_x_,
        reinterpret_cast<cuda_scalar_t *>(&beta), vec_y_, cuda_data_type(),
        CUSPARSE_SPMV_CSR_ALG2, &buffer_size));
    // maybe too much mem, we don't care for now -- rather more mem than too
    // little :)
    CHECK_CUDA(cudaMalloc(&buffer_, buffer_size * sizeof(T)));
  }

  ~DeviceSpMV() {
    if (buffer_)
      CHECK_CUDA(cudaFree(buffer_));
    CHECK_CUDA(cudaFree(d_x_));
    CHECK_CUDA(cudaFree(d_y_));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vec_x_));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vec_y_));
    CHECK_CUSPARSE(cusparseDestroySpMat(mat_desc_));
    CHECK_CUSPARSE(cusparseDestroy(handle_));
  }

  void multiply(const T *d_x, T *d_y) {
    T alpha(1), beta(0);
    CHECK_CUSPARSE(cusparseDnVecSetValues(
        vec_x_, const_cast<cuda_scalar_t *>(
                    reinterpret_cast<const cuda_scalar_t *>(d_x))));
    CHECK_CUSPARSE(
        cusparseDnVecSetValues(vec_y_, reinterpret_cast<cuda_scalar_t *>(d_y)));

    CHECK_CUSPARSE(
        cusparseSpMV(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     reinterpret_cast<cuda_scalar_t *>(&alpha), mat_desc_,
                     vec_x_, reinterpret_cast<cuda_scalar_t *>(&beta), vec_y_,
                     cuda_data_type(), CUSPARSE_SPMV_CSR_ALG2, buffer_));
    // ALG2 seems to perform close-to-perfect
    // (hand-optimized kernel for 2d no-flux Laplacian with 5 diagonals takes
    // a comparable amount of time)
  }
};
#endif
