import torch

    def u_xx_yy(buf, a, dx, dy): uxx_yy = buf uxx_yy[1: - 1, 1: - 1] = ((a[2:, 1: - 1] + a[: - 2, 1: - 1] - 2 * a[1: - 1, 1: - 1]) / (dx ** 2) + (a[1: - 1, 2:] + a[1: - 1, : - 2] - 2 * a[1: - 1, 1: - 1]) / (dy ** 2)) return uxx_yy

        def neumann_bc(u): u[0, 1: - 1] = u[1, 1: - 1] u[- 1, 1: - 1] = u[- 2, 1: - 1] u[:, 0] = u[:, 1] u[:, - 1] = u[:, - 2]

            if __name__ == '__main__':
                nx, ny = 3, 3 L = 1e-2 dx = dy = torch.tensor(2 * L / (nx - 1), dtype=torch.complex128) u = torch.ones((nx, ny), dtype=torch.complex128) buffer = torch.zeros_like(u)

               print("Norms pre:", u.real.norm(), u.imag.norm()) lapl = u_xx_yy(buffer, u, dx, dy) neumann_bc(lapl) print(u) print(lapl) print("Norms post:", lapl.real.norm(), lapl.imag.norm())
