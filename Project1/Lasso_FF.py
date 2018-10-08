import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib.ticker import FormatStrFormatter
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import resample
from sklearn.linear_model import Lasso

#Creating the Franke Funtction

np.random.seed(403)

def FrankeFunction(x,y):
    term1 = 0.75 * np.exp(-(0.25 * (9*x - 2)**2) - 0.25 * ((9*y - 2)**2))
    term2 = 0.75 * np.exp(-((9*x + 1)**2)/49.0 - 0.1 * (9*y + 1))
    term3 = 0.5 * np.exp(-((9*x - 7)**2)/4.0 - 0.25 * ((9*y - 3)**2))
    term4 = -0.2 * np.exp(-(9*x - 4)**2 - (9*y - 7)**2)
    return term1 + term2 + term3 + term4


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

def Lasso_FF(X, Y, F, lam=0.0001, k=1, graph=True):

    x_data = X.ravel()
    y_data = Y.ravel()
    f_data = F.ravel()

    xb = gen_def_matrix(x_data, y_data, k)
    lasso_reg = Lasso(alpha=lam, fit_intercept=False, max_iter=100000)
    lasso_reg.fit(xb, f_data,)
    # num_coef = 0
    # for i in lasso_reg.coef_:
    #     if i != 0:
    #         num_coef += 1
    # print(num_coef)
    
    xnew = np.linspace(0, 1, np.size(X, 0))
    ynew = np.linspace(0, 1, np.size(X, 1))
    Xnew, Ynew = np.meshgrid(xnew, ynew)
    F_true = FrankeFunction(Xnew, Ynew)

    xn = Xnew.ravel()
    yn = Ynew.ravel()

    xb_new = gen_def_matrix(xn, yn, k)
    f_predict = lasso_reg.predict(xb_new)
    F_predict = f_predict.reshape(F.shape)

    MSE = mean_squared_error(F_true, F_predict)
    R2 = r2_score(F_true, F_predict)

    beta = lasso_reg.coef_

    sigma2 = 0


    for i in range(np.size(F_true, 0)):
        for j in range(np.size(F_true,1)):
            sigma2 = sigma2 + (F_predict[i,j] - F_true[i,j])**2
    sigma2 = sigma2 / (f_predict.size - beta.size)

    var = np.diag(np.linalg.inv(xb_new.T.dot(xb_new))) * sigma2

    if(graph):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(Xnew, Ynew, F_predict, cmap= 'coolwarm', linewidth= 0, antialiased= False)
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink= 0.5, aspect= 0.5)
        plt.show()
    
    return MSE, R2, var

# Bootstraping

def boot_Lasso_FF(X, Y, F, lam=0.0001, it=1000, x_k=1):

    aMSE = np.zeros(it)
    aR2 = np.zeros(it)
    tMSE = 0
    tR2 = 0
    tvar = 0
    for i in range(it):
        Xt, Yt, Ft = resample(X, Y, F)

        MSE, R2, var = Lasso_FF(Xt, Yt, Ft, lam, x_k, False)
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

def k_fold_CV_Lasso_FF(X,Y,F,lam=0.0001,k=5,x_k=1):

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
        lasso_reg = Lasso(alpha=lam, fit_intercept=False, max_iter=100000)
        lasso_reg.fit(xb, f_train)
        num_coef = 0
        for j in lasso_reg.coef_:
            if j != 0:
                num_coef += 1
        print(num_coef)

        xb_test = gen_def_matrix(x_test, y_test, x_k)
        f_predict = lasso_reg.predict(xb_test)

        f_predict_train = lasso_reg.predict(xb)
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
    
    tMSE_test = tMSE_test / k
    tR2_test = tR2_test / k
    tMSE_train = tMSE_train / k
    tR2_train = tR2_train / k
    print("The test average MSE is: %.05f; the test average R^2-score is: %.02f" % (tMSE_test, tR2_test))
    print("The train average MSE is: %.05f; the train average R^2-score is: %.02f" % (tMSE_train, tR2_train))
    return aMSE_test, aR2_test, aMSE_train, aR2_train


# Moving the randomizer to get the same noise as in OLS
np.random.normal(0,1,(100,100))

# Creating the data
xp = np.random.rand(100, 1)
yp = np.random.rand(100, 1)
noise = np.random.normal(0, 1, (xp.size, yp.size))

X, Y = np.meshgrid(xp, yp)
F = FrankeFunction(X, Y) + noise


# print("Linear Model")
# MSE_1, R2_1, var_1 = Lasso_FF(X, Y, F)
# print("The MSE is: %.05f; and the R2 is: %.02f" % (MSE_1, R2_1))
# for i in range(var_1.size):
#     print("The variance of beta_%d is: %.08f" % (i, var_1[i]))

# print("Second-order polynomial Model")
# MSE_2, R2_2, var_2 = Lasso_FF(X, Y, F, k=2)
# print("The MSE is: %.05f; and the R2 is: %.02f" % (MSE_2, R2_2))
# for i in range(var_2.size):
#     print("The variance of beta_%d is: %.08f" % (i, var_2[i]))

# print("Third-order polynomial Model")
# MSE_3, R2_3, var_3 = Lasso_FF(X, Y, F, k=3)
# print("The MSE is: %.05f; and the R2 is: %.02f" % (MSE_3, R2_3))
# for i in range(var_3.size):
#     print("The variance of beta_%d is: %.08f" % (i, var_3[i]))

# print("Fourth-order polynomial Model")
# MSE_4, R2_4, var_4 = Lasso_FF(X, Y, F, k=4)
# print("The MSE is: %.05f; and the R2 is: %.02f" % (MSE_4, R2_4))
# for i in range(var_4.size):
#     print("The variance of beta_%d is: %.08f" % (i, var_4[i]))

# print("Fifth-order polynomial Model")
# MSE_5, R2_5, var_5 = Lasso_FF(X, Y, F, k=5)
# print("The MSE is: %.05f; and the R2 is: %.02f" % (MSE_5, R2_5))
# for i in range(var_5.size):
#     print("The variance of beta_%d is: %.08f" % (i, var_5[i]))

# print("Bootstraping 4th order polynomial")
# boot_Lasso_FF(X, Y, F, x_k= 4)

# print("%d-fold cross-validation 4th order polynomial" % k)
# k_fold_CV_Lasso_FF(X,Y,F,k,4)

#Finding the optimal lambda
lambdas = np.arange(0.0001, 1.00001, 0.0001)
MSEs = np.zeros(lambdas.size)
R2s = np.zeros(lambdas.size)
min_lambda_MSE = 0
min_lambda_R2 = 0 
place = 0
for i in lambdas:
    MSE_5, R2_5, var_5 = Lasso_FF(X, Y, F, lam=i, k=5, graph=False)
    MSEs[place] = MSE_5
    R2s[place] = R2_5
    place += 1
    if(min_lambda_MSE == 0 or MSE_5 < MSEs[int(min_lambda_MSE * 10000 - 1)]):
        min_lambda_MSE = i
    if(min_lambda_R2 == 0 or R2_5 > R2s[int(min_lambda_R2 * 10000 - 1)]):
        min_lambda_R2 = i
    if place % 1000 == 0:
        print("We have done %d iterations" % place)

print(MSEs)

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

nMSE = np.zeros(5)
nR2 = np.zeros(5)

bk = 1

for x_k in range(1,6):
    MSE, R2, var = Lasso_FF(X, Y, F, k=x_k, lam=opt_lam, graph=False)
    nMSE[x_k-1] = MSE
    nR2[x_k-1] = R2
    if(nR2[bk] < nR2[x_k-1]):
        bk = x_k

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

print("%dth order polynomial Model" % bk)
MSE_5, R2_5, var_5 = Lasso_FF(X, Y, F, lam=opt_lam, k=bk)
print("The MSE is: %.05f; and the R2 is: %.02f" % (MSE_5, R2_5))
for i in range(var_5.size):
    print("The variance of beta_%d is: %.08f" % (i, var_5[i]))

print("Bootstraping %dth order polynomial" % bk)
bMSE, bR2 = boot_Lasso_FF(X, Y, F, lam=opt_lam, x_k= bk)

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


k = 10
print("%d-fold cross-validation %dth order polynomial" % (k, 5))
cvMSE_test, cvR2_test, cvMSE_train, cvR2_train = k_fold_CV_Lasso_FF(X,Y,F,lam=0.0001,k=k,x_k=5)


plt.figure()
plt.xlabel('Fold')
plt.ylabel('MSE')
plt.title(('MSE of each fold (%dth order polynomial)' % 5))
plt.plot(range(k), cvMSE_test, 'bo', label='Test')
plt.plot(range(k), cvMSE_train, 'ro', label='Train')
plt.legend(loc=4, frameon=False)

plt.figure()
plt.xlabel('Fold')
plt.ylabel('R2')
plt.title(('R2 of each fold (%dth order polynomial)' % 5))
plt.plot(range(k), cvR2_test, 'bo', label='Test')
plt.plot(range(k), cvR2_train, 'ro', label='Train')
plt.legend(loc=4, frameon=False)

plt.show()