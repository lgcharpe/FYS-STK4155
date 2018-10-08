import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib.ticker import FormatStrFormatter
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import resample

#Creating the Franke Funtction

np.random.seed(403)

def FrankeFunction(x,y):
    term1 = 0.75 * np.exp(-(0.25 * (9*x - 2)**2) - 0.25 * ((9*y - 2)**2))
    term2 = 0.75 * np.exp(-((9*x + 1)**2)/49.0 - 0.1 * (9*y + 1))
    term3 = 0.5 * np.exp(-((9*x - 7)**2)/4.0 - 0.25 * ((9*y - 3)**2))
    term4 = -0.2 * np.exp(-(9*x - 4)**2 - (9*y - 7)**2)
    return term1 + term2 + term3 + term4

#Inititalizing the x and y variables

# x = np.random.rand(100, 1)
# y = np.random.rand(100, 1)
# f = FrankeFunction(x,y)

# x = np.arange(0, 1, 0.05)
# y = np.arange(0, 1, 0.05)
# x, y = np.meshgrid(x, y)
# z = FrankeFunction(x,y)

# #Plot the Franke Function

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(x, y, z, cmap= 'coolwarm', linewidth= 0, antialiased= False)

# #Customize z axis
# ax.set_zlim(-0.10, 1.40)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# #Add colour bar
# fig.colorbar(surf, shrink= 0.5, aspect= 0.5)

# plt.show()

#Inititalizing the x and y variables plus the Franke Function with noise

# x = np.random.rand(100, 1)
# y = np.random.rand(100, 1)
# X, Y = np.meshgrid(x, y)
# f = FrankeFunction(X,Y) + np.random.normal(0,1,(100,100))

# x = X.ravel()
# y = Y.ravel()

# #Setting up OLS

# xb = np.c_[np.ones((10000, 1)), x, y, x**2, x*y, y**2, x**3, (x**2)*y, x*(y**2), y**3, x**4, (x**3)*y, (x**2)*(y**2), x*(y**3), 
#             y**4, x**5, (x**4)*y, (x**3)*(y**2), (x**2)*(y**3), x*(y**4), y**5]
# beta = np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(f.ravel())

# #Doing OLS

# xnew = np.arange(0, 1, 0.01)
# ynew = np.arange(0, 1, 0.01)
# xnew2 = np.arange(0, 1, 0.01)
# ynew2 = np.arange(0, 1, 0.01)
# Xnew, Ynew = np.meshgrid(xnew, ynew)
# fn = FrankeFunction(Xnew,Ynew)

# xnew = Xnew.ravel()
# ynew = Ynew.ravel()

# xbnew = np.c_[np.ones((10000, 1)), xnew, ynew, xnew**2, xnew*ynew, ynew**2, xnew**3, (xnew**2)*ynew, xnew*(ynew**2), ynew**3, xnew**4, (xnew**3)*ynew, (xnew**2)*(ynew**2), xnew*(ynew**3), 
#             ynew**4, xnew**5, (xnew**4)*ynew, (xnew**3)*(ynew**2), (xnew**2)*(ynew**3), xnew*(ynew**4), ynew**5]
# fpredict = xbnew.dot(beta)
# fpredict = fpredict.reshape((100,100))

# # fpredict2 = np.zeros((100, 100), dtype='float')
# # for i in xnew2:
# #     for j in ynew2:
# #         posx = int(i / 0.01)
# #         posy = int(j / 0.01)
# #         vec = np.array([1, i, j, i**2, i*j, j**2, i**3, (i**2)*j, i*(j**2), j**3, i**4, (i**3)*j, (i**2)*(j**2), i*(j**3), j**4, i**5, (i**4)*j, (i**3)*(j**2), (i**2)*(j**3), i*(j**4), j**5])
# #         fpredict2[posx, posy] = vec.dot(beta)
# #         if(not(fpredict[posx, posy] == fpredict2[posx, posy])):
# #             print("Wrong placement of coordinates: (%.02f, %.02f)" % (i, j))
# #             print("fpredict: %.06f, frpedict2: %.06f" % (fpredict[posx, posy], fpredict2[posx, posy]))

# MSE = mean_squared_error(fn, fpredict)
# R2 = r2_score(fn, fpredict)

# sigma2 = 0

# for i in range(np.size(fn, 0)):
#     for j in range(np.size(fn,1)):
#         sigma2 = sigma2 + (fpredict[i,j] - fn[i,j])**2

# sigma2 = sigma2 / (fn.size - beta.size)

# var = np.diag(np.linalg.inv(xbnew.T.dot(xbnew))) * sigma2



# print("The Mean Squared Error is: %.05f" % MSE)
# print("The R-squared score is: %.02f" % R2)

# z_score = beta / np.sqrt(var)
# for i in range(len(beta)):
#     print("The variance of beta_%d is: %.02f and its z-score is: %.04f" % (i, var[i], z_score[i]))

# #Plot the Franke Function

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(Xnew, Ynew, fpredict, cmap= 'coolwarm', linewidth= 0, antialiased= False)

# #Customize z axis
# ax.set_zlim(-0.10, 1.40)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# #Add colour bar
# fig.colorbar(surf, shrink= 0.5, aspect= 0.5)

# plt.show()

def gen_def_matrix(x, y, k=1):

    xb = np.ones((x.size, 1))
    for i in range(1, k+1):
        for j in range(i+1):
            xb = np.c_[xb, (x**(i-j))*(y**j)]
    # xb = np.c_[np.ones((x.size, 1)), x, y, x**2, x*y, y**2, x**3, (x**2)*y, x*(y**2), y**3, x**4, (x**3)*y, (x**2)*(y**2), x*(y**3), 
    #         y**4, x**5, (x**4)*y, (x**3)*(y**2), (x**2)*(y**3), x*(y**4), y**5]
    return xb

def gen_beta(xb, f):

    beta = np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(f)
    return beta

def OLS_FF(X, Y, F, k=1, graph=True):

    x_data = X.ravel()
    y_data = Y.ravel()
    f_data = F.ravel()

    xb = gen_def_matrix(x_data, y_data, k)
    beta = gen_beta(xb, f_data)

    xnew = np.linspace(0, 1, np.size(X, 0))
    ynew = np.linspace(0, 1, np.size(X, 1))
    Xnew, Ynew = np.meshgrid(xnew, ynew)
    F_true = FrankeFunction(Xnew, Ynew)

    xn = Xnew.ravel()
    yn = Ynew.ravel()

    xb_new = gen_def_matrix(xn, yn, k)
    f_predict = xb_new.dot(beta)
    F_predict = f_predict.reshape(F.shape)

    MSE = mean_squared_error(F_true, F_predict)
    R2 = r2_score(F_true, F_predict)

    #print("The MSE is: %.05f; the R^2-score is: %.02f" % (MSE, R2))

    sigma2 = 0

    for i in range(np.size(F_true, 0)):
        for j in range(np.size(F_true,1)):
            sigma2 = sigma2 + (F_predict[i,j] - F_true[i,j])**2
    #print("Sigma before division: %.05f" % sigma2)
    sigma2 = sigma2 / (f_predict.size - beta.size)

    var = np.diag(np.linalg.inv(xb_new.T.dot(xb_new))) * sigma2

    var2 = np.var(F_predict)
    bias = np.mean(F_true - np.mean(F_predict))**2
    mse = np.mean((F_true-F_predict)**2)

    # print(var2)
    # print(bias)
    # print(var2 + bias)
    # print(mse)

    # for i in range(len(beta)):
    #     print("The variance of beta_%d is: %.08f" % (i, var[i]))

    if(graph):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(Xnew, Ynew, F_predict, cmap= 'coolwarm', linewidth= 0, antialiased= False)
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        plt.title("Linear Regression polynomial order %d" % k)
        plt.xlabel('X')
        plt.ylabel('Y')
        fig.colorbar(surf, shrink= 0.5, aspect= 0.5)
        plt.show()
    
    return MSE, R2, var

# Bootstraping

def boot_OLS_FF(X, Y, F, it=1000, x_k=1):

    # x = X.ravel()
    # y = Y.ravel()
    # f = F.ravel()
    aMSE = np.zeros(it)
    aR2 = np.zeros(it)
    tMSE = 0
    tR2 = 0
    tvar = 0
    for i in range(it):
        # x_temp, y_temp, f_temp = resample(x, y, f)
        # Xt = np.reshape(x_temp, X.shape)
        # Yt = np.reshape(y_temp, Y.shape)
        # Ft = np.reshape(f_temp, F.shape)
        Xt, Yt, Ft = resample(X, Y, F)

        MSE, R2, var = OLS_FF(Xt, Yt, Ft, x_k, False)
        tMSE = tMSE + MSE
        tR2 = tR2 + R2
        tvar = tvar + var
        aMSE[i] = MSE
        aR2[i] = R2
    
    tMSE = tMSE / it
    tR2 = tR2 / it
    tvar = tvar / it

    print("The average MSE is: %.05f; and the average R2 is: %.02f" % (tMSE, tR2))
    for i in range(tvar.size):
        print("The average variance for beta_%d is: %.08f" % (i, tvar[i]))
    return aMSE, aR2

#k-fold Cross-Validation

def k_fold_CV_OLS_FF(X,Y,F,k=5,x_k=1):

    F_true = FrankeFunction(X, Y)
    
    x2 = X.ravel()
    y2 = Y.ravel()
    f_data = F.ravel()
    f_true = F_true.ravel()

    shf = np.random.permutation(x2.size)

    x2 = x2[shf]
    y2 = y2[shf]
    f_data = f_data[shf]
    f_true = f_true[shf]

    size_fold = int(np.ceil(f_data.size / k))
    tMSE_test = 0
    tR2_test = 0
    aMSE_test = np.zeros(k)
    aR2_test = np.zeros(k)
    tMSE_train = 0
    tR2_train = 0
    aMSE_train = np.zeros(k)
    aR2_train = np.zeros(k)
    for i in range(k):
        start_val = size_fold*i
        end_val = min(size_fold*(i+1), x2.size)
        x_test = x2[start_val:end_val]
        y_test = y2[start_val:end_val]
        f_test = f_true[start_val:end_val]
        x_train = np.append(x2[0:start_val], x2[end_val:])
        y_train = np.append(y2[0:start_val], y2[end_val:])
        f_train = np.append(f_data[0:start_val], f_data[end_val:])

        xb = gen_def_matrix(x_train, y_train, x_k)
        beta = gen_beta(xb, f_train)

        xb_test = gen_def_matrix(x_test, y_test, x_k)
        f_predict = xb_test.dot(beta)

        f_predict_train = xb.dot(beta)
        f_train_test = np.append(f_true[0:start_val], f_true[end_val:])

        MSE = mean_squared_error(f_test, f_predict)
        R2 = r2_score(f_test, f_predict)
        tMSE_test = tMSE_test + MSE
        tR2_test = tR2_test + R2
        aMSE_test[i] = MSE
        aR2_test[i] = R2

        MSEt = mean_squared_error(f_train_test, f_predict_train)
        R2t = r2_score(f_train_test, f_predict_train)
        tMSE_train = tMSE_train + MSEt
        tR2_train = tR2_train + R2t
        aMSE_train[i] = MSEt
        aR2_train[i] = R2t

        #print("The MSE for fold %d is: %.05f; its R^2-score is: %.02f" % (i, MSE, R2))
    
    tMSE_test = tMSE_test / k
    tR2_test = tR2_test / k
    tMSE_train = tMSE_train / k
    tR2_train = tR2_train / k
    print("The test average MSE is: %.05f; the test average R^2-score is: %.02f" % (tMSE_test, tR2_test))
    print("The train average MSE is: %.05f; the train average R^2-score is: %.02f" % (tMSE_train, tR2_train))
    return aMSE_test, aR2_test, aMSE_train, aR2_train

# Graphing data
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)

x,y = np.meshgrid(x,y)
f = FrankeFunction(x,y)

#Plot the Franke Function

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x, y, f, cmap= 'coolwarm', linewidth= 0, antialiased= False)

#Customize z axis
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

#Labeling axes and title
ax.set_title('Franke function without noise')
ax.set_xlabel('X')
ax.set_ylabel('Y')


#Add colour bar
fig.colorbar(surf, shrink= 0.5, aspect= 0.5)

plt.show()

f += np.random.normal(0, 1, (100, 100))

#Plot the Franke Function

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x, y, f, cmap= 'coolwarm', linewidth= 0, antialiased= False)

#Customize z axis
ax.set_zlim(-3, 4.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

#Labeling axes and title
ax.set_title('Franke function with noise')
ax.set_xlabel('X')
ax.set_ylabel('Y')

#Add colour bar
fig.colorbar(surf, shrink= 0.5, aspect= 0.5)

plt.show()

# Creation of virtual data
xp = np.random.rand(100, 1)
yp = np.random.rand(100, 1)
noise = np.random.normal(0, 1, (xp.size, yp.size))

X, Y = np.meshgrid(xp, yp)
F = FrankeFunction(X, Y) + noise

nMSE = np.zeros(5)
nR2 = np.zeros(5)

for x_k in range(1,6):
    MSE, R2, var = OLS_FF(X, Y, F, k=x_k, graph=False)
    nMSE[x_k-1] = MSE
    nR2[x_k-1] = R2

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

# print("Linear Model")
# MSE_1, R2_1, var_1 = OLS_FF(X, Y, F)
# print("The MSE is: %.05f; and the R2 is: %.02f" % (MSE_1, R2_1))
# for i in range(var_1.size):
#     print("The variance of beta_%d is: %.08f" % (i, var_1[i]))

# print("Second-order polynomial Model")
# MSE_2, R2_2, var_2 = OLS_FF(X, Y, F, 2)
# print("The MSE is: %.05f; and the R2 is: %.02f" % (MSE_2, R2_2))
# for i in range(var_2.size):
#     print("The variance of beta_%d is: %.08f" % (i, var_2[i]))

# print("Third-order polynomial Model")
# MSE_3, R2_3, var_3 = OLS_FF(X, Y, F, 3)
# print("The MSE is: %.05f; and the R2 is: %.02f" % (MSE_3, R2_3))
# for i in range(var_3.size):
#     print("The variance of beta_%d is: %.08f" % (i, var_3[i]))

# print("Fourth-order polynomial Model")
# MSE_4, R2_4, var_4 = OLS_FF(X, Y, F, 4)
# print("The MSE is: %.05f; and the R2 is: %.02f" % (MSE_4, R2_4))
# for i in range(var_4.size):
#     print("The variance of beta_%d is: %.08f" % (i, var_4[i]))

print("Fifth-order polynomial Model")
MSE_5, R2_5, var_5 = OLS_FF(X, Y, F, 5)
print("The MSE is: %.05f; and the R2 is: %.02f" % (MSE_5, R2_5))
for i in range(var_5.size):
    print("The variance of beta_%d is: %.08f" % (i, var_5[i]))

print("Bootstraping 5th order polynomial")
bMSE, bR2 = boot_OLS_FF(X, Y, F, x_k= 5)

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


k = 10
print("%d-fold cross-validation 5th order polynomial" % k)
cvMSE_test, cvR2_test, cvMSE_train, cvR2_train = k_fold_CV_OLS_FF(X,Y,F,k,5)


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
