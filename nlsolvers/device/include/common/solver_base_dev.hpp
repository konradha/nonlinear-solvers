#ifndef SOLVER_BASE_DEV_HPP
#define SOLVER_BASE_DEV_HPP

#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace device {

template <typename Scalar_t, int Dim>
class SolverBase {
public:
    virtual ~SolverBase() = default;

    virtual void initialize(const std::vector<uint32_t>& grid_sizes,
                           const std::vector<Scalar_t>& domain_sizes) = 0;
    
    virtual void load_initial_condition(const std::string& input_file) = 0;
    
    virtual void step() = 0;
    
    virtual void apply_bc() = 0;
    
    virtual void run(Scalar_t T, uint32_t nt, uint32_t num_snapshots,
                    const std::string& output_file) = 0;
                    
    virtual void store_snapshot(uint32_t snapshot_idx) = 0;
};

} // namespace device

#endif // SOLVER_BASE_DEV_HPP
