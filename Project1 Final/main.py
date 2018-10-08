import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.ticker import FormatStrFormatter
from Regression_functions import OLS, Ridge, Lasso, cross_validation, bootstrap

# Load the terrain
terrain1 = imread('SRTM_data_Norway_1.tif')
# Show the terrain
plt.figure()
plt.title('Terrain over Norway 1')
plt.imshow(terrain1, cmap='coolwarm')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar()
plt.show()

[n, m] = terrain1.shape
print(n)
print(m)

rows = np.linspace(0,1,n)
columns = np.linspace(0,1,m)

[X, Y] = np.meshgrid(columns, rows)

print(X.shape)
print(Y.shape)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, terrain1, cmap= 'coolwarm', linewidth= 0, antialiased= False)
ax.set_zlim(0, 2000)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink= 0.5, aspect= 0.5)
plt.show()

# OLS

#Effect of complexity on MSE and R2, also finding best x_k

nMSE = np.zeros(5)
nR2 = np.zeros(5)

for x_k in range(1,6):
    MSE, R2, var = OLS(X, Y, terrain1, X, Y, terrain1, k=x_k, graph=False)
    nMSE[x_k-1] = MSE
    nR2[x_k-1] = R2
    print("Done!")

plt.figure()
plt.xlabel('Complexity (Order of polynomial)')
plt.ylabel('MSE')
plt.title('Change in MSE depending on the complexity of the model')
plt.plot(range(1,6), nMSE)

plt.figure()
plt.xlabel('Complexity (Order of polynomial)')
plt.ylabel('R2')
plt.title('Change in R2 depending on the complexity of the model')
plt.plot(range(1,6), nR2)

#Printing the graph of each patch at the optimal x_k

print("Fifth-order polynomial Model")
MSE_5, R2_5, var_5 = OLS(X, Y, terrain1, X, Y, terrain1, 5)
print("The MSE is: %.05f; and the R2 is: %.02f" % (MSE_5, R2_5))
for i in range(var_5.size):
    print("The variance of beta_%d is: %.08f" % (i, var_5[i]))

#Doing Bootstrapping

print("Bootstraping 5th order polynomial")
bMSE, bR2 = bootstrap(X, Y, terrain1, x_k= 5, method='OLS')

plt.figure()
plt.hist(bMSE)
plt.xlabel('MSE')
plt.ylabel('Frequency')
plt.title('Frequency of MSE results in bootstrap (5th order polynomial)')
plt.show()

plt.figure()
plt.hist(bR2)
plt.xlabel('R2')
plt.ylabel('Frequency')
plt.title('Frequency of R2 results in bootstrap (5th order polynomial)')
plt.show()

# Doing Cross-Validation

k = 10
print("%d-fold cross-validation 5th order polynomial" % k)
cvMSE_test, cvR2_test, cvMSE_train, cvR2_train = cross_validation(X,Y,terrain1,k=k,x_k=5,method='OLS')


plt.figure()
plt.xlabel('Fold')
plt.ylabel('MSE')
plt.title('MSE of each fold (5th order polynomial)')
plt.plot(range(k), cvMSE_test, 'bo', label='Test')
plt.plot(range(k), cvMSE_train, 'ro', label='Train')
plt.legend(loc=4, frameon=False)

plt.figure()
plt.xlabel('Fold')
plt.ylabel('R2')
plt.title('R2 of each fold (5th order polynomial)')
plt.plot(range(k), cvR2_test, 'bo', label='Test')
plt.plot(range(k), cvR2_train, 'ro', label='Train')
plt.legend(loc=4, frameon=False)

plt.show()

# Ridge

#Finding the optimal lambda
lambdas = np.arange(0.0001, 0.0101, 0.0001)
MSEs = np.zeros(lambdas.size)
R2s = np.zeros(lambdas.size)
min_lambda_MSE = 0
min_lambda_R2 = 0 
place = 0
for i in lambdas:
    MSE_5, R2_5, var_5 = Ridge(X, Y, terrain1, X, Y, terrain1, lam=i, k=5, graph=False)
    MSEs[place] = MSE_5
    R2s[place] = R2_5
    place += 1
    if(min_lambda_MSE == 0 or MSE_5 < MSEs[int(min_lambda_MSE * 10000 - 1)]):
        min_lambda_MSE = i
    if(min_lambda_R2 == 0 or R2_5 > R2s[int(min_lambda_R2 * 10000 - 1)]):
        min_lambda_R2 = i
    if place % 10 == 0:
        print("We have done %d iterations" % place)

opt_lam = (min_lambda_MSE + min_lambda_R2) / 2

print("The best lambda for the MSE is: %.04f; and the best for the R2 is: %.04f" % (min_lambda_MSE, min_lambda_R2))

plt.figure()
plt.plot(lambdas,MSEs)
plt.xlabel('Lambda')
plt.ylabel('MSE')
plt.title('Variation of MSE depending on lambda (5th order polynomial)')

plt.figure()
plt.plot(lambdas,R2s)
plt.xlabel('Lambda')
plt.ylabel('R2')
plt.title('Variation of R2 depending on lambda (5th order polynomial)')
plt.show()

#Effect of complexity on MSE and R2, also finding best x_k

nMSE = np.zeros(5)
nR2 = np.zeros(5)

bk = 1

for x_k in range(1,6):
    MSE, R2, var = Ridge(X, Y, terrain1, X, Y, terrain1, k=x_k, lam=opt_lam, graph=False)
    nMSE[x_k-1] = MSE
    nR2[x_k-1] = R2
    if(nR2[bk] < nR2[x_k-1]):
        bk = x_k
    print("Done!")

plt.figure()
plt.xlabel('Complexity (Order of polynomial)')
plt.ylabel('MSE')
plt.title('Change in MSE depending on the complexity of the model')
plt.plot(range(1,6), nMSE)

plt.figure()
plt.xlabel('Complexity (Order of polynomial)')
plt.ylabel('R2')
plt.title('Change in R2 depending on the complexity of the model')
plt.plot(range(1,6), nR2)

#Printing the graph of each patch at the optimal lambda and x_k

print("Fifth-order polynomial Model")
MSE_5, R2_5, var_5 = Ridge(X, Y, terrain1, X, Y, terrain1, lam=opt_lam, k=bk)
print("The MSE is: %.05f; and the R2 is: %.02f" % (MSE_5, R2_5))
for i in range(var_5.size):
    print("The variance of beta_%d is: %.08f" % (i, var_5[i]))

#Doing Bootstrapping

print("Bootstraping 5th order polynomial")
bMSE, bR2 = bootstrap(X, Y, terrain1, lam=opt_lam, x_k= bk, method='Ridge')

plt.figure()
plt.hist(bMSE)
plt.xlabel('MSE')
plt.ylabel('Frequency')
plt.title('Frequency of MSE results in bootstrap (5th order polynomial)')
plt.show()

plt.figure()
plt.hist(bR2)
plt.xlabel('R2')
plt.ylabel('Frequency')
plt.title('Frequency of R2 results in bootstrap (5th order polynomial)')
plt.show()

# Doing Cross-Validation

k = 10
print("%d-fold cross-validation 5th order polynomial" % k)
cvMSE_test, cvR2_test, cvMSE_train, cvR2_train = cross_validation(X,Y,terrain1,lam=opt_lam,k=k,x_k=bk, method='Ridge')


plt.figure()
plt.xlabel('Fold')
plt.ylabel('MSE')
plt.title('MSE of each fold (5th order polynomial)')
plt.plot(range(k), cvMSE_test, 'bo', label='Test')
plt.plot(range(k), cvMSE_train, 'ro', label='Train')
plt.legend(loc=4, frameon=False)

plt.figure()
plt.xlabel('Fold')
plt.ylabel('R2')
plt.title('R2 of each fold (5th order polynomial)')
plt.plot(range(k), cvR2_test, 'bo', label='Test')
plt.plot(range(k), cvR2_train, 'ro', label='Train')
plt.legend(loc=4, frameon=False)

plt.show()

#Lasso using batches

patch_size_row = 100
patch_size_col = 100

# Define their axes
rows = np.linspace(0,1,patch_size_row)
cols = np.linspace(0,1,patch_size_col)

[C,R] = np.meshgrid(cols,rows)

num_data = patch_size_row*patch_size_col

# Find the start indices of each patch

num_patches = 5

np.random.seed(4155)

row_starts = np.random.randint(0,n-patch_size_row,num_patches)
col_starts = np.random.randint(0,m-patch_size_col,num_patches)

#Finding the optimal lambda

lambdas = np.arange(0.0001, 0.0101, 0.0001)
MSEs = np.zeros(lambdas.size)
R2s = np.zeros(lambdas.size)

for i,row_start, col_start in zip(np.arange(num_patches),row_starts, col_starts):
    row_end = row_start + patch_size_row
    col_end = col_start + patch_size_col

    patch = terrain1[row_start:row_end, col_start:col_end]

    lambdas = np.arange(0.0001, 0.0101, 0.0001)
    MSEs = np.zeros(lambdas.size)
    R2s = np.zeros(lambdas.size)
    place = 0
    for j in lambdas:
        MSE_5, R2_5, var_5 = Lasso(C, R, patch, C, R, patch, lam=j, k=5, graph=False)
        MSEs[place] += MSE_5
        R2s[place] += R2_5
        place += 1
        if place % 10 == 0:
            print("We have done %d iterations" % place)

MSEs = MSEs / num_patches
R2s = R2s / num_patches

min_lambda_MSE = lambdas[np.argmin(MSEs)]
min_lambda_R2 = lambdas[np.argmax(R2s)]

print("The best lambda for the MSE is: %.04f; and the best for the R2 is: %.04f" % (min_lambda_MSE, min_lambda_R2))

plt.figure()
plt.plot(lambdas,MSEs)
plt.xlabel('Lambda')
plt.ylabel('MSE')
plt.title('Variation of MSE depending on lambda (5th order polynomial)')

plt.figure()
plt.plot(lambdas,R2s)
plt.xlabel('Lambda')
plt.ylabel('R2')
plt.title('Variation of R2 depending on lambda (5th order polynomial)')
plt.show()

opt_lam = (min_lambda_MSE + min_lambda_R2)/2

#Effect of complexity on MSE and R2, also finding best x_k

nMSE = np.zeros(5)
nR2 = np.zeros(5)

bk = 1 # best x_k

for i,row_start, col_start in zip(np.arange(num_patches),row_starts, col_starts):
    row_end = row_start + patch_size_row
    col_end = col_start + patch_size_col

    patch = terrain1[row_start:row_end, col_start:col_end]

    for x_k in range(1,6):
        MSE, R2, var = Lasso(C, R, patch, C, R, patch, k=x_k, lam=opt_lam, graph=False)
        nMSE[x_k-1] += MSE
        nR2[x_k-1] += R2
        if(nR2[bk-1] < nR2[x_k-1]):
            bk = x_k
        print("Done!")
    
nMSE = nMSE / num_patches
nR2 = nR2 / num_patches

plt.figure()
plt.xlabel('Complexity (Order of polynomial)')
plt.ylabel('MSE')
plt.title('Change in MSE depending on the complexity of the model')
plt.plot(range(1,6), nMSE)

plt.figure()
plt.xlabel('Complexity (Order of polynomial)')
plt.ylabel('R2')
plt.title('Change in R2 depending on the complexity of the model')
plt.plot(range(1,6), nR2)

#Printing the graph of each patch at the optimal lambda and x_k

for i,row_start, col_start in zip(np.arange(num_patches),row_starts, col_starts):
    row_end = row_start + patch_size_row
    col_end = col_start + patch_size_col

    patch = terrain1[row_start:row_end, col_start:col_end]

    print("%dth order polynomial Model, patch %d" % (bk,i))
    MSE_5, R2_5, var_5 = Lasso(C, R, patch, C, R, patch, lam=opt_lam, k=bk)
    print("The MSE is: %.05f; and the R2 is: %.02f" % (MSE_5, R2_5))
    for j in range(var_5.size):
        print("The variance of beta_%d is: %.08f" % (j, var_5[j]))

#Doing Bootstrapping

bMSE = 0
bR2 = 0

for i,row_start, col_start in zip(np.arange(num_patches),row_starts, col_starts):
    row_end = row_start + patch_size_row
    col_end = col_start + patch_size_col

    patch = terrain1[row_start:row_end, col_start:col_end]

    print("Bootstraping %dth order polynomial, patch %d" % (bk,i))
    tbMSE, tbR2 = bootstrap(C, R, patch, lam=opt_lam, x_k= bk)
    bMSE += tbMSE
    bR2 += tbR2

bMSE = bMSE / num_patches
bR2 = bR2 / num_patches

plt.figure()
plt.hist(bMSE)
plt.xlabel('MSE')
plt.ylabel('Frequency')
plt.title(('Frequency of MSE results in bootstrap (%dth order polynomial)' % bk))
plt.show()

plt.figure()
plt.hist(bR2)
plt.xlabel('R2')
plt.ylabel('Frequency')
plt.title(('Frequency of R2 results in bootstrap (%dth order polynomial)' % bk))
plt.show()

# Doing Cross-Validation

k = 10 #number of folds
cvMSE_test = 0
cvR2_test = 0
cvMSE_train = 0
cvR2_train = 0

for i,row_start, col_start in zip(np.arange(num_patches),row_starts, col_starts):
    row_end = row_start + patch_size_row
    col_end = col_start + patch_size_col

    patch = terrain1[row_start:row_end, col_start:col_end]

    print("%d-fold cross-validation %dth order polynomial" % (k, bk))
    tcvMSE_test, tcvR2_test, tcvMSE_train, tcvR2_train = cross_validation(C,R,patch,lam=opt_lam,k=k,x_k=bk)
    cvMSE_test += tcvMSE_test
    cvR2_test += tcvR2_test
    cvMSE_train += tcvMSE_train
    cvR2_train += tcvR2_train

cvMSE_test = cvMSE_test / num_patches
cvR2_test = cvR2_test / num_patches
cvMSE_train = cvMSE_train / num_patches
cvR2_train = cvR2_train / num_patches

plt.figure()
plt.xlabel('Fold')
plt.ylabel('MSE')
plt.title(('MSE of each fold (%dth order polynomial)' % bk))
plt.plot(range(k), cvMSE_test, 'bo', label='Test')
plt.plot(range(k), cvMSE_train, 'ro', label='Train')
plt.legend(loc=4, frameon=False)

plt.figure()
plt.xlabel('Fold')
plt.ylabel('R2')
plt.title(('R2 of each fold (%dth order polynomial)' % bk))
plt.plot(range(k), cvR2_test, 'bo', label='Test')
plt.plot(range(k), cvR2_train, 'ro', label='Train')
plt.legend(loc=4, frameon=False)

plt.show()