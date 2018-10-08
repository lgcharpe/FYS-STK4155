import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib.ticker import FormatStrFormatter
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import resample
import sklearn.linear_model as lm
from numba import jit
import sklearn.metrics as skm


@jit
def gen_def_matrix(x, y, k=1):

    xb = np.ones((x.size, 1))
    for i in range(1, k+1):
        for j in range(i+1):
            xb = np.c_[xb, (x**(i-j))*(y**j)]
    return xb

def gen_beta(xb, f, lam=0):

    dim = np.shape(xb)[1]
    beta = np.linalg.inv(xb.T.dot(xb) + lam * np.eye(dim)).dot(xb.T.dot(f))
    return beta

@jit
def resample3(x, y, f):
    comb = np.random.randint(0,f.size,f.size)
    x_train = np.zeros(x.size)
    y_train = np.zeros(y.size)
    f_train = np.zeros(f.size)
    for i in range(f.size):
        x_train[i] = x[comb[i]]
        y_train[i] = y[comb[i]]
        f_train[i] = f[comb[i]]
    
    return x_train, y_train, f_train

def cal_MSE(f_predict, f_true):
    MSE = np.sum((f_true - f_predict)**2, dtype=np.float64) / f_true.size
    return MSE

@jit
def cal_R2(f_predict, f_true):
    # ft = f_true.ravel()
    # fp = f_predict.ravel()
    # print(np.sum(ft))
    # print(ft.size)
    # f_mean = (np.sum(ft)) / ft.size
    # print(f_mean)
    # f_mean2 = np.mean(f_true)
    # print(f_mean2)
    # temp = ft - f_mean2
    # SS_tot = np.sum((temp)**2)
    # SS_res = np.sum((ft - fp)**2)
    # res_tot = np.sum(((ft-fp)/temp)**2)
    SS_tot = np.float64(0)
    f_bar = np.mean(f_true, dtype=np.float64)
    for i in range(len(f_true)):
        for j in range(len(f_true[i])):
            SS_tot += (np.float64(f_true[i,j] - f_bar))**2
    SS_res = np.float64(0)
    for i in range(len(f_true)):
        for j in range(len(f_true[i])):
            SS_res += (np.float64(f_predict[i,j] - f_true[i,j]))**2
    R2 = 1 - SS_res/SS_tot #np.sum((f_true - np.mean(f_true))**2)

    return R2

@jit
def OLS(X_train, Y_train, F_train, X_test, Y_test, F_test, k=1, graph=True):

    x_train = X_train.ravel()
    y_train = Y_train.ravel()
    f_train = F_train.ravel()

    xb = gen_def_matrix(x_train, y_train, k)
    beta = gen_beta(xb, f_train)

    x_test = X_test.ravel()
    y_test = Y_test.ravel()

    xb_test = gen_def_matrix(x_test, y_test, k)
    sklOLS = lm.LinearRegression()
    sklOLS.fit(xb, f_train)

    f_predict = xb_test.dot(beta)
    F_predict = f_predict.reshape(F_test.shape)
    sklpre = sklOLS.predict(xb_test)
    
    # print(np.max(sklpre - f_predict))
    # print(np.min(sklpre - f_predict))

    MSE = mean_squared_error(F_test, F_predict)
    # MSE2 = cal_MSE(F_predict, F_test)
    R2 = r2_score(F_test, F_predict)
    # R22 = cal_R2(F_predict, F_test)

    # print(MSE2 - MSE)
    # print(R22 - R2)

    sigma2 = 0

    for i in range(np.size(F_test, 0)):
        for j in range(np.size(F_test,1)):
            sigma2 = sigma2 + (F_predict[i,j] - F_test[i,j])**2
    sigma2 = sigma2 / (f_predict.size - beta.size)

    varb = np.diag(np.linalg.inv(xb_test.T.dot(xb_test))) * sigma2



    var = np.sum( (F_predict - np.mean(F_predict, dtype=np.float64))**2, dtype=np.float64 )/F_test.size
    bias = np.mean((F_test - np.mean(F_predict))**2)
    # print(bias)
    # print(beta[0])
    # print(np.mean(F_predict, dtype=np.float64))

    # print(np.var(f_predict))

    bias2 = (np.mean(np.absolute(F_predict - F_test)))**2
    # print(np.min(np.absolute(F_predict - F_test)))
    var2 = (np.mean(F_predict**2)) - ((np.mean(F_predict))**2)

    # print(bias2)
    # print(var2)
    # print(bias2 + var2)
    

    if(graph):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X_test, Y_test, F_predict, cmap= 'coolwarm', linewidth= 0, antialiased= False)
        ax.set_zlim(0, 2000)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink= 0.5, aspect= 0.5)
        plt.show()
    
    return MSE, R2, varb

@jit
def Ridge(X_train, Y_train, F_train, X_test, Y_test, F_test, lam=0.01, k=1, graph=True):

    x_train = X_train.ravel()
    y_train = Y_train.ravel()
    f_train = F_train.ravel()

    xb = gen_def_matrix(x_train, y_train, k)
    beta = gen_beta(xb, f_train, lam)

    x_test = X_test.ravel()
    y_test = Y_test.ravel()

    # sklridge = lm.Ridge(alpha=lam, fit_intercept=False)
    # sklridge.fit(xb, f_train)

    xb_test = gen_def_matrix(x_test, y_test, k)
    f_predict = xb_test.dot(beta)
    F_predict = f_predict.reshape(F_test.shape)
    # sklridge_pred = sklridge.predict(xb_test)
    # sklridge_pred = sklridge_pred.reshape(F_test.shape)

    MSE = mean_squared_error(F_test, F_predict)
    R2 = r2_score(F_predict, F_test, F_predict, )

    # skl_MSE = mean_squared_error(F_test, sklridge_pred)
    # skl_R2 = r2_score(F_test, sklridge_pred)

    # diff_MSE = MSE - skl_MSE
    # diff_R2 = R2 - skl_R2

    sigma2 = 0

    for i in range(np.size(F_test, 0)):
        for j in range(np.size(F_test,1)):
            sigma2 = sigma2 + (F_predict[i,j] - F_test[i,j])**2
    sigma2 = sigma2 / (f_predict.size - beta.size)

    varb = np.diag(np.linalg.inv(xb_test.T.dot(xb_test))) * sigma2

    var = np.sum( (F_predict - np.mean(F_predict, dtype=np.float64))**2, dtype=np.float64 )/F_test.size
    bias = np.sum( (F_test - np.mean(F_predict, dtype=np.float64))**2, dtype=np.float64 )/F_test.size

    if(graph):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X_test, Y_test, F_predict, cmap= 'coolwarm', linewidth= 0, antialiased= False)
        ax.set_zlim(0, 2000)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink= 0.5, aspect= 0.5)
        plt.show()
    
    return MSE, R2, varb

@jit
def Lasso(X_train, Y_train, F_train, X_test, Y_test, F_test, lam=0.0001, k=1, graph=True):

    x_train = X_train.ravel()
    y_train = Y_train.ravel()
    f_train = F_train.ravel()

    xb = gen_def_matrix(x_train, y_train, k)
    lasso_reg = lm.Lasso(alpha=lam, fit_intercept=False, max_iter=10000)
    lasso_reg.fit(xb, f_train,)

    x_test = X_test.ravel()
    y_test = Y_test.ravel()

    xb_test = gen_def_matrix(x_test, y_test, k)
    f_predict = lasso_reg.predict(xb_test)
    F_predict = f_predict.reshape(F_test.shape)

    MSE = mean_squared_error(F_test, F_predict)
    R2 = r2_score(F_test, F_predict)

    beta = lasso_reg.coef_

    sigma2 = 0


    for i in range(np.size(F_test, 0)):
        for j in range(np.size(F_test,1)):
            sigma2 = sigma2 + (F_predict[i,j] - F_test[i,j])**2
    sigma2 = sigma2 / (f_predict.size - beta.size)

    varb = np.diag(np.linalg.inv(xb_test.T.dot(xb_test))) * sigma2

    var = np.sum( (F_predict - np.mean(F_predict, dtype=np.float64))**2, dtype=np.float64 )/F_test.size
    bias = np.sum( (F_test - np.mean(F_predict, dtype=np.float64))**2, dtype=np.float64 )/F_test.size

    if(graph):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X_test, Y_test, F_predict, cmap= 'coolwarm', linewidth= 0, antialiased= False)
        ax.set_zlim(0, 2000)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink= 0.5, aspect= 0.5)
        plt.show()
    
    return MSE, R2, varb

def cross_validation(X, Y, F, k=5, lam=0.001, x_k=1, method='OLS'):

    x = X.ravel()
    y = Y.ravel()
    f = F.ravel()

    shf = np.random.permutation(x.size)

    Xs = x[shf].reshape(X.shape)
    Ys = y[shf].reshape(Y.shape)
    Fs = f[shf].reshape(F.shape)

    size_fold = int(np.ceil(len(Fs) / k))
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
        end_val = min(size_fold*(i+1), len(Xs))
        print(end_val)
        X_test = Xs[start_val:end_val]
        print(X_test)
        Y_test = Ys[start_val:end_val]
        F_test = Fs[start_val:end_val]
        X_train = np.r_[Xs[0:start_val], Xs[end_val:]]
        print(X_train.shape)
        Y_train = np.r_[Ys[0:start_val], Ys[end_val:]]
        F_train = np.r_[Fs[0:start_val], Fs[end_val:]]

        if(method == 'OLS'):
            MSE, R2, var = OLS(X_train, Y_train, F_train, X_test, Y_test, F_test, x_k, False)
            MSEt, R2t, vart = OLS(X_train, Y_train, F_train, X_train, Y_train, F_train, x_k, False)
        elif(method == 'Ridge'):
            MSE, R2, var = Ridge(X_train, Y_train, F_train, X_test, Y_test, F_test, lam, x_k, False)
            MSEt, R2t, vart = Ridge(X_train, Y_train, F_train, X_train, Y_train, F_train, lam, x_k, False)
        elif(method == 'Lasso'):
            MSE, R2, var = Lasso(X_train, Y_train, F_train, X_test, Y_test, F_test, lam, x_k, False)
            MSEt, R2t, vart = Lasso(X_train, Y_train, F_train, X_train, Y_train, F_train, lam, x_k, False)

        tMSE_test = tMSE_test + MSE
        tR2_test = tR2_test + R2
        aMSE_test[i] = MSE
        aR2_test[i] = R2

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

def bootstrap(X, Y, F, it=100, lam=0.001, x_k=1, method='OLS'):

    aMSE = np.zeros(it)
    aR2 = np.zeros(it)
    tMSE = 0
    tR2 = 0
    tvar = 0
    for i in range(it):
        Xt, Yt, Ft = resample(X, Y, F)

        if(method == 'OLS'):
            MSE, R2, var = OLS(Xt, Yt, Ft, X, Y, F, x_k, False)
        elif(method == 'Ridge'):
            MSE, R2, var = Ridge(Xt, Yt, Ft, X, Y, F, lam, x_k, False)
        elif(method == 'Lasso'):
            MSE, R2, var = Lasso(Xt, Yt, Ft, X, Y, F, lam, x_k, False)
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
