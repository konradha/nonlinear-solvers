import numpy as np
import matplotlib.pyplot as plt

def make_grid(n, L):
    xn = yn = np.linspace(-L, L, n)
    X, Y = np.meshgrid(xn, yn)
    return X, Y


def make_periodic_boxes(n, L, factor,
        set_zero=False,
        box_length=.1, num_boxes_per_dim=3, wall_dist=.1):
    X, Y = make_grid(n, L)
    box_length *= L
    wall_dist *= L
    available_space = 2*L - 2*wall_dist
    total_boxes_length = num_boxes_per_dim * box_length
    spacing = (available_space - total_boxes_length)/(num_boxes_per_dim - 1) if num_boxes_per_dim > 1 else 0
    
    start_x = start_y = -L + wall_dist
    m = np.ones_like(X)
    
    masks = []
    for i in range(num_boxes_per_dim):
        for j in range(num_boxes_per_dim):
            x0 = start_x + i*(box_length + spacing)
            x1 = x0 + box_length
            y0 = start_y + j*(box_length + spacing)
            y1 = y0 + box_length
            mask = np.logical_and.reduce((
                X >= x0, X <= x1,
                Y >= y0, Y <= y1
            ))
            m[mask] *= -factor 
    if set_zero:
        # only apply nonlinearity in boxes!
        for mask in masks:
            m[mask] = 0.
    return m

def make_step(n, L, factor=-1.):
    X, Y = make_grid(n, L)
    mask = X < 0.
    m = np.ones_like(X)
    m[mask] *= factor
    return m

if __name__ == '__main__':
    m = make_periodic_boxes(200, 10., factor=3., num_boxes_per_dim=5)
    plt.imshow(m)
    plt.show()

    m = make_step(200, 10., factor=3.)
    plt.imshow(m)
    plt.show()
