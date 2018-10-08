import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

x = np.random.rand(100,1)*10
y = 5*(x**2) + 100*np.random.rand(100,1)


xb = np.c_[np.ones((100,1)), x, x**2]
beta = np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(y)

xnew = np.linspace(0, 10, 10000)
yn = 5*(xnew**2)
xbnew = np.c_[np.ones((10000, 1)), xnew, xnew**2]
ypredict = xbnew.dot(beta)

plt.plot(xnew, ypredict, "r-")
plt.plot(x, y, 'ro')
plt.axis([0,10,0,500])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Linear Regression')
plt.show()

#Sickit-learn

poly2 = PolynomialFeatures(degree=2)
X = poly2.fit_transform(x)
slf2 = LinearRegression()
slf2.fit(X,y)

Xplot = poly2.fit_transform(xnew[:,np.newaxis])
plt.figure(1)
poly2_plot = plt.plot(xnew, slf2.predict(Xplot), label='Square Fit')
plt.plot(xnew, yn, color='red', label='True Square')
plt.scatter(x, y, label='Data', color='orange', s=15)
plt.legend()
plt.show()

print(r2_score(yn, slf2.predict(Xplot)))
print(mean_squared_error(yn, slf2.predict(Xplot)))