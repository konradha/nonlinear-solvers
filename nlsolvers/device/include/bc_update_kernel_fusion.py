import torch
import torch._dynamo as dynamo


@torch.compile(backend="inductor", fullgraph=True, dynamic=True)
def neumann_bc(u: torch.Tensor, nx: int, ny: int) -> torch.Tensor:
    assert len(u.shape) == 2 
    # no corners
    u[0, 1:ny-1]    = u[1, 1:ny-1]
    u[nx-1, 1:ny-1] = u[nx-2, 1:ny-1]

    # with corners
    u[:, 0]    = u[:, 1]
    u[:, ny-1] = u[:, ny-2] 
    return u

if __name__ == '__main__':
    torch._inductor.config.debug = True
    torch._dynamo.config.verbose = True
    nx = ny = 300
    test_tensor = torch.rand((nx, ny), dtype=torch.float64, device="cuda" if torch.cuda.is_available() else "cpu")
    result = neumann_bc(test_tensor, nx, ny)
    exp = dynamo.explain(neumann_bc)(test_tensor, nx, ny)
    print("First explainer", exp)

    try:
        if hasattr(torch, "_inductor"): 
            torch._inductor.config.trace.enabled = True
            torch._inductor.config.trace.debug_cuda = True
            result = neumann_bc(test_tensor, nx, ny)
    except Exception as _:
        print("CUDA probably not avail")
