import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sys import argv
from PIL import Image as img

def u_analytical(x, y, t):
    a = [1., 0, 3, 2, 4, 0, 0, 6, 0]
    a1, a2, a3, a4, a5, a6, a7, a8, a9 = a

    c1 = a1*a5/a3
    c2 = 0

    x_shifted = x - c1*t
    y_shifted = y - c2*t

    term1 = a1 * x_shifted + a2 * x_shifted + a4
    term2 = a5 * x_shifted

    denom = (term1**2 + term2**2 +
             6*a3**2*(a1**4 + 2*a1**2*a5**2 + a5**4)/(a5*(a1**2 - a3**2))**2)
    num1 = 2*(2*a1**2 + 2*a5**2)
    num2 = 2*(2*term1*a1 + 2*term2*a5)**2

    return num1/denom - num2/denom**2



def animate(X, Y, data, nt, save=None):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, data[0], cmap='viridis',)
    def update(frame):
        ax.clear()
        ax.plot_surface(X, Y,
                #data[frame],
                #np.imag(data[frame]),
                np.abs(data[frame]),
                cmap='coolwarm')
    fps = 10
    ani = FuncAnimation(fig, update, frames=nt, interval=nt / fps, )

    if save and save == "gif":
        fname = f"evolution.gif"
        ani.save(fname,)
        #img.open(fname).save(fname, optimize=True)

    elif save and save ==  "mp4":
        fname = f"evolution.mp4"
        ani.save(fname,)
    else:
        plt.show()

def animate_comparison(x, y, data, nt,):
    fig, axs = plt.subplots(ncols=2,subplot_kw={"projection":'3d'})
    dt = 5 / 100 
    def update(frame):
        axs[0].clear()
        axs[1].clear()

        axs[0].plot_surface(X, Y,
                data[frame],
                cmap='viridis')

        axs[1].plot_surface(X, Y,
                u_analytical(X, Y, frame * dt),
                cmap='viridis') 

    fps = 10
    ani = FuncAnimation(fig, update, frames=nt, interval=nt / fps, )
    plt.show()

if __name__ == '__main__':
    fname = str(argv[1])
    L = float(argv[2])
    nx = int(argv[3])
    ny = int(argv[4])
    save_ani = str(argv[5])

    data = np.load(fname)
    nt = data.shape[0]
    assert data.shape[1] == nx and data.shape[2] == ny

    xn = np.linspace(-L, L, nx)
    yn = np.linspace(-L, L, ny)
    X, Y = np.meshgrid(xn, yn)

    #animate_comparison(X, Y, data, nt)
    animate(X, Y, data, nt, save_ani)
