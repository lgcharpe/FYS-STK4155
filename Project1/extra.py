import numpy as np
#from scipy.misc import imread
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def surface_plot(surface,title, surface1=None):
    M,N = surface.shape

    ax_rows = np.arange(M)
    ax_cols = np.arange(N)

    [X,Y] = np.meshgrid(ax_cols, ax_rows)

    fig = plt.figure()
    if surface1 is not None:
        ax = fig.add_subplot(1,2,1,projection='3d')
        ax.plot_surface(X,Y,surface,cmap="viridis",linewidth=0)
        plt.title(title)

        ax = fig.add_subplot(1,2,2,projection='3d')
        ax.plot_surface(X,Y,surface1,cmap="viridis",linewidth=0)
        plt.title(title)
    else:
        ax = fig.gca(projection='3d')
        ax.plot_surface(X,Y,surface,cmap="viridis",linewidth=0)
        plt.title(title)


def predict(rows, cols, beta):
    out = np.zeros((np.size(rows), np.size(cols)))

    for i,y_ in enumerate(rows):
        for j,x_ in enumerate(cols):
            data_vec = np.array([1, x_, y_, x_**2, x_*y_, y_**2, \
                                x_**3, x_**2*y_, x_*y_**2, y_**3, \
                                x_**4, x_**3*y_, x_**2*y_**2, x_*y_**3,y_**4, \
                                x_**5, x_**4*y_, x_**3*y_**2, x_**2*y_**3,x_*y_**4,y_**5])#,\
                            #    x_**6, x_**5*y_, x_**4*y_**2, x_**3*y_**3,x_**2*y_**4, x_*y_**5, y_**6, \
                            #    x_**7, x_**6*y_, x_**5*y_**2, x_**4*y_**3,x_**3*y_**4, x_**2*y_**5, x_*y_**6, y_**7, \
                            #    x_**8, x_**7*y_, x_**6*y_**2, x_**5*y_**3,x_**4*y_**4, x_**3*y_**5, x_**2*y_**6, x_*y_**7,y_**8, \
                            #    x_**9, x_**8*y_, x_**7*y_**2, x_**6*y_**3,x_**5*y_**4, x_**4*y_**5, x_**3*y_**6, x_**2*y_**7,x_*y_**8, y_**9])
            out[i,j] = data_vec @ beta

    return out

from sklearn.metrics import mean_squared_error

if __name__ == '__main__':

    # Load the terrain
    terrain1 = imread('SRTM_data_Norway_1.tif')
    [n,m] = terrain1.shape

    ## Find some random patches within the dataset and perform a fit

    patch_size_row = 100
    patch_size_col = 50

    # Define their axes
    rows = np.linspace(0,1,patch_size_row)
    cols = np.linspace(0,1,patch_size_col)

    [C,R] = np.meshgrid(cols,rows)

    x = C.reshape(-1,1)
    y = R.reshape(-1,1)

    num_data = patch_size_row*patch_size_col

    # Find the start indices of each patch

    num_patches = 5

    np.random.seed(4155)

    row_starts = np.random.randint(0,n-patch_size_row,num_patches)
    col_starts = np.random.randint(0,m-patch_size_col,num_patches)

    for i,row_start, col_start in zip(np.arange(num_patches),row_starts, col_starts):
        row_end = row_start + patch_size_row
        col_end = col_start + patch_size_col

        patch = terrain1[row_start:row_end, col_start:col_end]

        z = patch.reshape(-1,1)

        # Perform OLS fit
        data = np.c_[np.ones((num_data,1)), x, y, \
                     x**2, x*y, y**2, \
                     x**3, x**2*y, x*y**2, y**3, \
                     x**4, x**3*y, x**2*y**2, x*y**3,y**4, \
                     x**5, x**4*y, x**3*y**2, x**2*y**3,x*y**4, y**5]#, \
                     #x**6, x**5*y, x**4*y**2, x**3*y**3,x**2*y**4, x*y**5, y**6, \
                     #x**7, x**6*y, x**5*y**2, x**4*y**3,x**3*y**4, x**2*y**5, x*y**6, y**7, \
                     #x**8, x**7*y, x**6*y**2, x**5*y**3,x**4*y**4, x**3*y**5, x**2*y**6, x*y**7,y**8, \
                     #x**9, x**8*y, x**7*y**2, x**6*y**3,x**5*y**4, x**4*y**5, x**3*y**6, x**2*y**7,x*y**8, y**9]

        beta_ols = np.linalg.inv(data.T @ data) @ data.T @ z

        fitted_patch = predict(rows, cols, beta_ols)

        mse = np.sum( (fitted_patch - patch)**2 )/num_data
        R2 = 1 - np.sum( (fitted_patch - patch)**2 )/np.sum( (patch - np.mean(patch))**2 )
        var = np.sum( (fitted_patch - np.mean(fitted_patch))**2 )/num_data
        bias = np.sum( (patch - np.mean(fitted_patch))**2 )/num_data

        print("patch %d, from (%d, %d) to (%d, %d)"%(i+1, row_start, col_start, row_end,col_end))
        print("mse: %g\nR2: %g"%(mse, R2))
        print("variance: %g"%var)
        print("bias: %g\n"%bias)

        surface_plot(fitted_patch,'Fitted terrain surface',patch)

    # Perform fit over the whole dataset
    print("The whole dataset")

    rows = np.linspace(0,1,n)
    cols = np.linspace(0,1,m)

    [C,R] = np.meshgrid(cols,rows)

    x = C.reshape(-1,1)
    y = R.reshape(-1,1)

    num_data = n*m

    data = np.c_[np.ones((num_data,1)), x, y, \
                 x**2, x*y, y**2, \
                 x**3, x**2*y, x*y**2, y**3, \
                 x**4, x**3*y, x**2*y**2, x*y**3,y**4, \
                 x**5, x**4*y, x**3*y**2, x**2*y**3,x*y**4, y**5]

    z = terrain1.flatten()
    
    beta_ols = np.linalg.inv(data.T @ data) @ data.T @ z

    fitted_terrain = predict(rows, cols, beta_ols)

    mse = np.sum( (fitted_terrain - terrain1)**2 )/num_data
    R2 = 1 - np.sum( (fitted_terrain - terrain1)**2 )/np.sum( (terrain1- np.mean(terrain1))**2 )
    var = np.sum( (fitted_terrain - np.mean(fitted_terrain))**2 )/num_data
    bias = np.sum( (terrain1 - np.mean(fitted_terrain))**2 )/num_data

    print("mse: %g\nR2: %g"%(mse, R2))
    print("variance: %g"%var)
    print("bias: %g\n"%bias)

    plt.show()