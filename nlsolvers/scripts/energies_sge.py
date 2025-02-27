import numpy as np

def energy_sge_hyperbolic(un, dt, dx, dy, m):
    # NOTE: dt will not be the actual dt, as we don't actually have access
    # to u_i and u_{i+1} but rather u_{i} + u{i + snapshot_freq}

    # take first-order approximant of velocity as we don't have
    # direct access to it here

    nt, nx, ny = un.shape
    energy = np.zeros(nt)
    
    v = np.zeros_like(un)
    v[0] = (un[1] - un[0]) / dt
    v[1:-1] = (un[2:] - un[:-2]) / (2 * dt)
    v[-1] = (un[-1] - un[-2]) / dt
    
    for t in range(nt):
        u_interior = un[t, 1:-1, 1:-1]
        v_interior = v[t, 1:-1, 1:-1]
        m_interior = m[1:-1, 1:-1]
        
        # gradient scaled to grid
        ux = (un[t, 2:, 1:-1] - un[t, :-2, 1:-1]) / (2 * dx)
        uy = (un[t, 1:-1, 2:] - un[t, 1:-1, :-2]) / (2 * dy)
        
        kinetic = 0.5 * v_interior**2
        gradient = 0.5 * (ux**2 + uy**2)
        potential = m_interior * np.cosh(u_interior)
        
        energy[t] = np.sum((kinetic + gradient + potential) * dx * dy)
    
    return energy


