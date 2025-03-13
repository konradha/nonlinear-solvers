import torch
import triton
import triton.language as tl
import time
import os
import numpy as np

class LanczosStepTorch(torch.nn.Module):
    def __init__(self, dtype=torch.complex128):
        super().__init__()
        self.dtype = dtype
    
    def forward(self, buf1, V, T, j, m):
        buf1_out = buf1.clone()
        T_out = T.clone()
        
        if j > 0:
            dot_result = torch.vdot(V[j-1], buf1_out)
            T_out[j-1, j] = dot_result
            T_out[j, j-1] = dot_result.conj()
            buf1_out = buf1_out - dot_result * V[j-1]
        
        dot_result = torch.vdot(V[j], buf1_out)
        T_out[j, j] = dot_result
        buf1_out = buf1_out - dot_result * V[j]
        # TODO set this up as varying gemv
        h = torch.zeros(j+1, dtype=buf1_out.dtype, device=buf1_out.device)
        for i in range(j+1):
            h[i] = torch.vdot(V[i], buf1_out)
        
        for i in range(j+1):
            buf1_out = buf1_out - h[i] * V[i]
        
        beta = torch.norm(buf1_out)
        beta_val = beta.item() if beta.numel() == 1 else float(beta.detach().cpu().numpy())
        
        if beta_val > 0.0:
            beta_tensor = torch.tensor(complex(beta_val, 0.), dtype=T.dtype)
            T_out[j, j+1] = beta_tensor
            T_out[j+1, j] = beta_tensor
            buf1_out = buf1_out / beta
        
        return buf1_out, T_out, beta_val

@triton.jit
def complex_dot_kernel(
    buf1_ptr, v_ptr, dot_real_ptr, dot_imag_ptr,
    n, stride_buf1, stride_v,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
     
    offsets_bytes_real = offsets * 16
    offsets_bytes_imag = offsets_bytes_real + 8
    
    buf1_real = tl.load(buf1_ptr + offsets_bytes_real, mask=mask, other=0.0)
    buf1_imag = tl.load(buf1_ptr + offsets_bytes_imag, mask=mask, other=0.0)
    
    v_real = tl.load(v_ptr + offsets_bytes_real, mask=mask, other=0.0)
    v_imag = tl.load(v_ptr + offsets_bytes_imag, mask=mask, other=0.0)
    dot_real = v_real * buf1_real + v_imag * buf1_imag
    dot_imag = v_real * buf1_imag - v_imag * buf1_real
    
    dot_real_sum = tl.sum(dot_real, axis=0)
    dot_imag_sum = tl.sum(dot_imag, axis=0)
    
    if pid == 0:
        tl.atomic_add(dot_real_ptr, dot_real_sum)
        tl.atomic_add(dot_imag_ptr, dot_imag_sum)

@triton.jit
def complex_axpy_kernel(
    buf1_ptr, v_ptr, scale_real, scale_imag,
    n, stride_buf1, stride_v,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # let's try and see if Triton eats these offsets
    offsets_bytes_real = offsets * 16
    offsets_bytes_imag = offsets_bytes_real + 8
    
    # separate re + imag
    buf1_real = tl.load(buf1_ptr + offsets_bytes_real, mask=mask, other=0.0)
    buf1_imag = tl.load(buf1_ptr + offsets_bytes_imag, mask=mask, other=0.0)
    
    v_real = tl.load(v_ptr + offsets_bytes_real, mask=mask, other=0.0)
    v_imag = tl.load(v_ptr + offsets_bytes_imag, mask=mask, other=0.0)
    
    # (a+bi) * (c+di) = (ac-bd) + (ad+bc)i
    scale_times_v_real = scale_real * v_real - scale_imag * v_imag
    scale_times_v_imag = scale_real * v_imag + scale_imag * v_real
    result_real = buf1_real - scale_times_v_real
    result_imag = buf1_imag - scale_times_v_imag
    tl.store(buf1_ptr + offsets_bytes_real, result_real, mask=mask)
    tl.store(buf1_ptr + offsets_bytes_imag, result_imag, mask=mask)

class LanczosStepTriton:
    def __init__(self, dtype=torch.complex128):
        self.dtype = dtype
    
    def forward(self, buf1, V, T, j, m):
        n = buf1.shape[0]
        device = buf1.device
        BLOCK_SIZE = 128
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        
        buf1_out = buf1.clone()
        T_out = T.clone()
        
        if j > 0:
            dot_real = torch.zeros(1, dtype=torch.float64, device=device)
            dot_imag = torch.zeros(1, dtype=torch.float64, device=device)
            
            complex_dot_kernel[grid](
                buf1_out.data_ptr(), V[j-1].data_ptr(), 
                dot_real.data_ptr(), dot_imag.data_ptr(),
                n, buf1_out.stride(0), V.stride(1),
                BLOCK_SIZE
            )
            
            dot_result = torch.complex(dot_real.item(), dot_imag.item())
            T_out[j-1, j] = dot_result
            T_out[j, j-1] = dot_result.conj()
            
            complex_axpy_kernel[grid](
                buf1_out.data_ptr(), V[j-1].data_ptr(), 
                dot_real.item(), dot_imag.item(),
                n, buf1_out.stride(0), V.stride(1),
                BLOCK_SIZE
            )
        
        dot_real = torch.zeros(1, dtype=torch.float64, device=device)
        dot_imag = torch.zeros(1, dtype=torch.float64, device=device)
        
        complex_dot_kernel[grid](
            buf1_out.data_ptr(), V[j].data_ptr(), 
            dot_real.data_ptr(), dot_imag.data_ptr(),
            n, buf1_out.stride(0), V.stride(1),
            BLOCK_SIZE
        )
        
        dot_result = torch.complex(dot_real.item(), dot_imag.item())
        T_out[j, j] = dot_result
        
        complex_axpy_kernel[grid](
            buf1_out.data_ptr(), V[j].data_ptr(), 
            dot_real.item(), dot_imag.item(),
            n, buf1_out.stride(0), V.stride(1),
            BLOCK_SIZE
        )
        
        # For orthogonalization, fall back to PyTorch for simplicity and correctness
        # This avoids memory layout issues in Triton while keeping the performance benefits
        # of the optimized dot product and axpy operations
        h = torch.zeros(j+1, dtype=buf1_out.dtype, device=device)
        for i in range(j+1):
            h[i] = torch.vdot(V[i], buf1_out)
        
        for i in range(j+1):
            buf1_out = buf1_out - h[i] * V[i]
        
        beta = torch.norm(buf1_out).item()
        
        if beta > 0.0:
            T_out[j, j+1] = torch.complex(beta, 0.0)
            T_out[j+1, j] = torch.complex(beta, 0.0)
            buf1_out = buf1_out / beta
        
        return buf1_out, T_out, beta

def benchmark_lanczos_step(sizes=[(1000, 10, 5), (10000, 15, 8), (100000, 20, 10)], 
                           iterations=10, warmup=3):
    results = []
    
    cuda_available = torch.cuda.is_available()
    
    torch_cpu_model = LanczosStepTorch().cpu()
    
    for n, m, j in sizes:
        print(f"\nBenchmarking n={n}, m={m}, j={j}")
        
        cpu_buf1 = torch.randn(n, dtype=torch.complex128, device="cpu")
        cpu_V = torch.randn((m, n), dtype=torch.complex128, device="cpu")
        cpu_T = torch.zeros((m, m), dtype=torch.complex128, device="cpu")
        
        for _ in range(warmup):
            torch_cpu_model(cpu_buf1.clone(), cpu_V.clone(), cpu_T.clone(), j, m)
        
        start = time.time()
        for _ in range(iterations):
            buf1_cpu, T_cpu, beta_cpu = torch_cpu_model(
                cpu_buf1.clone(), cpu_V.clone(), cpu_T.clone(), j, m
            )
        cpu_time = (time.time() - start) / iterations
        
        result = {
            "n": n,
            "m": m,
            "j": j,
            "cpu_time_ms": cpu_time * 1000,
        }
        
        print(f"  CPU (PyTorch): {cpu_time * 1000:.2f} ms")
        
        results.append(result)
        
        if cuda_available:
            torch_gpu_model = LanczosStepTorch().cuda()
            triton_model = LanczosStepTriton()
            
            gpu_buf1 = torch.randn(n, dtype=torch.complex128, device="cuda")
            gpu_V = torch.randn((m, n), dtype=torch.complex128, device="cuda")
            gpu_T = torch.zeros((m, m), dtype=torch.complex128, device="cuda")
            
            try:
                for _ in range(warmup):
                    torch_gpu_model(gpu_buf1.clone(), gpu_V.clone(), gpu_T.clone(), j, m)
                    try:
                        triton_model.forward(gpu_buf1.clone(), gpu_V.clone(), gpu_T.clone(), j, m)
                    except Exception as e:
                        print(f"  Warning: Triton warmup failed: {e}")
                        print("  Falling back to PyTorch GPU only")
                        break
                
                torch.cuda.synchronize()
                start = time.time()
                for _ in range(iterations):
                    buf1_gpu, T_gpu, beta_gpu = torch_gpu_model(
                        gpu_buf1.clone(), gpu_V.clone(), gpu_T.clone(), j, m
                    )
                torch.cuda.synchronize()
                gpu_time = (time.time() - start) / iterations
                
                result.update({
                    "gpu_time_ms": gpu_time * 1000,
                })
                
                print(f"  GPU (PyTorch): {gpu_time * 1000:.2f} ms")
                try:
                    torch.cuda.synchronize()
                    start = time.time()
                    for _ in range(iterations):
                        buf1_triton, T_triton, beta_triton = triton_model.forward(
                            gpu_buf1.clone(), gpu_V.clone(), gpu_T.clone(), j, m
                        )
                    torch.cuda.synchronize()
                    triton_time = (time.time() - start) / iterations
                    
                    triton_l2_error_buf1 = torch.norm(buf1_gpu - buf1_triton) / torch.norm(buf1_gpu)
                    triton_l2_error_T = torch.norm(T_gpu - T_triton) / torch.norm(T_gpu)
                    
                    result.update({
                        "triton_time_ms": triton_time * 1000,
                        "triton_speedup_vs_gpu": gpu_time / triton_time,
                        "triton_l2_error_buf1": triton_l2_error_buf1.item(),
                        "triton_l2_error_T": triton_l2_error_T.item(),
                    })
                    
                    print(f"  GPU (Triton): {triton_time * 1000:.2f} ms")
                    print(f"  Triton vs GPU speedup: {gpu_time / triton_time:.2f}x")
                    print(f"  Triton L2 Error buf1: {triton_l2_error_buf1.item():.2e}")
                    print(f"  Triton L2 Error T: {triton_l2_error_T.item():.2e}")
                except Exception as e:
                    print(f"  Triton benchmark failed: {e}")
            except Exception as e:
                print(f"  GPU benchmark failed: {e}")
    
    return results

def analyze_performance(results):
    print("\nPerformance Analysis Summary:")
    print("-------------------------------")
    
    for result in results:
        n, m, j = result["n"], result["m"], result["j"]
        print(f"\nMatrix Size: {n}x{n}, Krylov subspace size: {m}, Current step: {j}")
        
        print(f"CPU Performance: {result['cpu_time_ms']:.2f} ms")
        
        if "gpu_time_ms" in result:
            print("GPU Performance:")
            print(f"  - PyTorch: {result['gpu_time_ms']:.2f} ms")
            
            if "triton_time_ms" in result:
                print(f"  - Triton: {result['triton_time_ms']:.2f} ms")
                print(f"  - Speedup: {result['triton_speedup_vs_gpu']:.2f}x")
                print(f"  - L2 Errors: buf1={result['triton_l2_error_buf1']:.2e}, T={result['triton_l2_error_T']:.2e}")

def main():
    print("Benchmarking Lanczos Step implementations...")
    results = benchmark_lanczos_step()
    
    analyze_performance(results)
    
    print("\nAll benchmarks completed!")

if __name__ == "__main__":
    main()
