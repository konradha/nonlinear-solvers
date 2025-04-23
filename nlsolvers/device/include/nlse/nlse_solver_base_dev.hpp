#ifndef NLSE_SOLVER_BASE_DEV_HPP
#define NLSE_SOLVER_BASE_DEV_HPP

#include "../common/solver_base_dev.hpp"
#include <complex>
#include <thrust/complex.h>

namespace device {

template <typename Scalar_t, int Dim>
class NLSESolverBase : public SolverBase<Scalar_t, Dim> {
public:
    virtual ~NLSESolverBase() = default;
    
    enum class Method {
        SS2,
        SEWI
    };
    
    virtual void set_method(Method method) {
        method_ = method;
    }
    
    virtual Method get_method() const {
        return method_;
    }
    
    virtual void load_focusing_parameter(const std::string& m_file) = 0;
    
protected:
    Method method_ = Method::SS2;
};

} // namespace device

#endif // NLSE_SOLVER_BASE_DEV_HPP
