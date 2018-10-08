import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import sklearn.linear_model as lm
from sklearn.preprocessing import PolynomialFeatures
from Regression_functions import OLS, Ridge, Lasso, gen_def_matrix

def test_gen_matrix():
    x = np.ones(10)
    y = np.ones(10)
    f = np.ones((10,3))
    c = gen_def_matrix(x,y)
    assert np.all(f == c)

def test_OLS():
    x = np.linspace(20, 50, 10)
    y = np.linspace(20, 50, 10)
    X,Y = np.meshgrid(x,y)
    x = X.ravel()
    y = Y.ravel()
    F = 5*X + 2*Y + 4
    MSE, R2, var = OLS(X,Y,F,X,Y,F,graph=False)
    Xv = gen_def_matrix(x,y)
    slf2 = lm.LinearRegression()
    slf2.fit(Xv,F.ravel())
    Fp = slf2.predict(Xv).reshape(F.shape)
    MSEt = mean_squared_error(F, Fp)
    R2t = r2_score(F,Fp)
    assert abs(MSE-MSEt) < 1e-6 and abs(R2-R2t) < 1e-6

def test_Ridge():
    x = np.linspace(20, 50, 10)
    y = np.linspace(20, 50, 10)
    X,Y = np.meshgrid(x,y)
    x = X.ravel()
    y = Y.ravel()
    F = 5*X + 2*Y + 4
    MSE, R2, var = Ridge(X,Y,F,X,Y,F,graph=False)
    Xv = gen_def_matrix(x,y)
    slf2 = lm.Ridge(alpha=0.01)
    slf2.fit(Xv,F.ravel())
    Fp = slf2.predict(Xv).reshape(F.shape)
    MSEt = mean_squared_error(F, Fp)
    R2t = r2_score(F,Fp)
    assert abs(MSE-MSEt) < 1e-4 and abs(R2-R2t) < 1e-4

def test_Lasso():
    x = np.linspace(20, 50, 10)
    y = np.linspace(20, 50, 10)
    X,Y = np.meshgrid(x,y)
    x = X.ravel()
    y = Y.ravel()
    F = 5*X + 2*Y + 4
    MSE, R2, var = Lasso(X,Y,F,X,Y,F,graph=False)
    Xv = gen_def_matrix(x,y)
    slf2 = lm.Lasso(alpha=0.0001)
    slf2.fit(Xv,F.ravel())
    Fp = slf2.predict(Xv).reshape(F.shape)
    MSEt = mean_squared_error(F, Fp)
    R2t = r2_score(F,Fp)
    assert abs(MSE-MSEt) < 1e-4 and abs(R2-R2t) < 1e-4