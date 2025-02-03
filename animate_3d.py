import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from sys import argv


def animate_3d(X, Y, Z, data, nt, save=None):
   fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6), 
                                      subplot_kw={'projection': '3d'})
   
   scatters = []
   for ax in (ax1, ax2, ax3):
       scatter = ax.scatter([], [], [], c=[], cmap='viridis', alpha=0.5)
       scatters.append(scatter)
       ax.set_xlim([X.min(), X.max()])
       ax.set_ylim([Y.min(), Y.max()])
       ax.set_zlim([Z.min(), Z.max()])
   
   #ax1.set_title('Absolute Value')
   #ax2.set_title('Real Part')
   #ax3.set_title('Imaginary Part')
   
   def update(frame):
       data_frame = data[frame]
       abs_data = np.abs(data_frame)
       real_data = np.real(data_frame)
       imag_data = np.imag(data_frame)
       
       threshold = 0.3 * np.max(abs_data)
       points = np.where(abs_data > threshold)
       
       for ax, scatter, component in zip((ax1, ax2, ax3), scatters, 
                                       (abs_data, real_data, imag_data)):
           scatter._offsets3d = (X[points], Y[points], Z[points])
           scatter.set_array(component[points])
           #ax.view_init(elev=20, azim=frame % 360)
       
       return scatters
   
   fps = 30
   ani = FuncAnimation(fig, update, frames=nt, interval=1000/fps)

   if save == "gif":
       ani.save("evolution_3d.gif", writer='pillow', fps=fps)
   elif save == "mp4":
       ani.save("evolution_3d.mp4", writer='ffmpeg', fps=fps)
   else:
       plt.show()

if __name__ == '__main__':
    fname = str(argv[1])
    L = float(argv[2])
    nx = int(argv[3])
    ny = int(argv[4])
    nz = int(argv[5])
    save_ani = str(argv[6])

    data = np.load(fname)
    nt = data.shape[0]
    assert data.shape[1] == nx and data.shape[2] == ny and data.shape[3] == nz

    xn = np.linspace(-L, L, nx)
    yn = np.linspace(-L, L, ny)
    zn = np.linspace(-L, L, nz)
    X, Y, Z = np.meshgrid(xn, yn, zn, indexing='ij')

    animate_3d(X, Y, Z, data, nt, save_ani)
