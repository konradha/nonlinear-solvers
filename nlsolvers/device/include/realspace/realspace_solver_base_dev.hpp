#ifndef REALSPACE_SOLVER_BASE_DEV_HPP
#define REALSPACE_SOLVER_BASE_DEV_HPP

#include "../common/solver_base_dev.hpp"

namespace device {

template <typename Scalar_t, int Dim>
class RealSpaceSolverBase : public SolverBase<Scalar_t, Dim> {
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
    
    virtual void load_initial_velocity(const std::string& velocity_file) = 0;
    
    virtual void set_velocity_output_file(const std::string& velocity_output_file) = 0;
    
    virtual void load_coupling_parameter(const std::string& m_file) = 0;
    
protected:
    Method method_ = Method::GAUTSCHI;
};

} // namespace device

#endif // REALSPACE_SOLVER_BASE_DEV_HPP
