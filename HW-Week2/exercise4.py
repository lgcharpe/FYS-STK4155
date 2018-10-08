import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

x = np.random.rand(100,1)
y = 5*(x**2) + 0.1*np.random.rand(100,1)


xb = np.c_[np.ones((100,1)), x, x**2]
beta = np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(y)
print(beta)

lam = 10
RSS = (y - xb.dot(beta)).T.dot(y - xb.dot(beta)) + lam * beta.T.dot(beta)
beta_ridge = np.linalg.inv(xb.T.dot(xb) + lam * np.eye(3)).dot(xb.T.dot(y))
beta_test = (1 + lam)**(-1) * beta
reg = beta_ridge / (beta)**2
print(beta_ridge)
print(beta_test)
print(beta)
print(reg)

xnew = np.linspace(0, 1, 1000)
yn = 5*(xnew**2)
xbnew = np.c_[np.ones((1000, 1)), xnew, xnew**2]
ypredict = xbnew.dot(beta_ridge)

print("Variance of beta variables:")
print(np.linalg.inv(xb.T.dot(xb) + lam * np.eye(3)))

plt.plot(xnew, ypredict, "r-")
plt.plot(x, y, 'ro')
plt.axis([0,1,0,5.5])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Linear Regression')
plt.show()