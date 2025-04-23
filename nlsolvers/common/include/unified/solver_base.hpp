#ifndef SOLVER_BASE_UNIFIED_HPP
#define SOLVER_BASE_UNIFIED_HPP

#include <string>
#include <vector>
#include <optional>
#include <memory>

namespace nlsolvers {

template <typename Scalar_t>
class SolverBase {
public:
    virtual ~SolverBase() = default;

    virtual void initialize(const std::vector<uint32_t>& grid_sizes,
                           const std::vector<Scalar_t>& domain_sizes) = 0;
    
    virtual void load_initial_condition(const std::string& input_file) = 0;
    
    virtual void load_anisotropy(const std::string& anisotropy_file) {
        // Default implementation does nothing
        // Derived classes should override this if they support anisotropy
    }
    
    virtual void run(Scalar_t T, uint32_t nt, uint32_t num_snapshots,
                    const std::string& output_file) = 0;
};

template <typename Scalar_t>
class NLSESolverBase : public SolverBase<Scalar_t> {
public:
    virtual ~NLSESolverBase() = default;
    
    enum class Method {
        SS2,
        SEWI,
        GAUTSCHI
    };
    
    virtual void set_method(Method method) {
        method_ = method;
    }
    
    virtual Method get_method() const {
        return method_;
    }
    
    virtual void load_focusing(const std::string& focusing_file) {
        // Default implementation does nothing
        // Derived classes should override this if they support focusing parameter
    }
    
protected:
    Method method_ = Method::SS2;
};

template <typename Scalar_t>
class RealSpaceSolverBase : public SolverBase<Scalar_t> {
public:
    virtual ~RealSpaceSolverBase() = default;
    
    enum class Method {
        GAUTSCHI,
        STORMER_VERLET
    };
    
    virtual void set_method(Method method) {
        method_ = method;
    }
    
    virtual Method get_method() const {
        return method_;
    }
    
    virtual void load_initial_velocity(const std::string& velocity_file) {
        // Default implementation does nothing
        // Derived classes should override this
    }
    
    virtual void set_velocity_output_file(const std::string& velocity_output_file) {
        // Default implementation does nothing
        // Derived classes should override this
    }
    
    virtual void load_coupling(const std::string& coupling_file) {
        // Default implementation does nothing
        // Derived classes should override this if they support coupling parameter
    }
    
protected:
    Method method_ = Method::GAUTSCHI;
};

} // namespace nlsolvers

#endif // SOLVER_BASE_UNIFIED_HPP
