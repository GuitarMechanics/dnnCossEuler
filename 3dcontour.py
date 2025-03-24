import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load CSV data
file_path = '/home/guitarmechanics/git_repos/DnnCossEuler/curvature_reginfos.csv'
data = pd.read_csv(file_path)

# Extract columns
x_data = data['LR_ratio'].values
y_data = data['tipang'].values
z_data = data['Kratio'].values

# Create grid for data surface
xi = np.linspace(x_data.min(), x_data.max(), 100)
yi = np.linspace(y_data.min(), y_data.max(), 100)
xi, yi = np.meshgrid(xi, yi)
zi = griddata((x_data, y_data), z_data, (xi, yi), method='linear')

# Create grid for math function surface (use same range for easy comparison)

# Stack x and y for polynomial feature generation
XY = np.vstack((x_data, y_data)).T

# Generate polynomial features (2차 다항식까지)
poly = PolynomialFeatures(degree=3)
XY_poly = poly.fit_transform(XY)

# Fit regression model
# Fit regression model without fitting intercept
model = LinearRegression()
model.fit(XY_poly, z_data)

# 모델 계수 출력
coeff_names = poly.get_feature_names_out(['x', 'y'])
for name, coef in zip(coeff_names, model.coef_):
    print(f'{name}: {coef:.5f}')

# 예측용 그리드 생성
xfit = np.linspace(x_data.min(), x_data.max(), 100)
yfit = np.linspace(y_data.min(), y_data.max(), 100)
xfit, yfit = np.meshgrid(xfit, yfit)
XYfit = np.vstack((xfit.ravel(), yfit.ravel())).T

# Predict with fixed intercept added back
Zfit = model.predict(poly.transform(XYfit)).reshape(xfit.shape)

fig = plt.figure(figsize = (12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot data-based surface
surf1 = ax.plot_surface(xi, yi, zi, cmap='viridis', alpha=0.8, edgecolor='none', label='Data Surface')

# Plot math function surface (offset upward for clarity if needed)
surf2 = ax.plot_surface(xfit, yfit, Zfit, cmap='plasma', alpha=0.8, edgecolor='none', label='Math Surface')

# Overlay original data points
ax.scatter(x_data, y_data, z_data, c='k', s=5)

# Labels and title
ax.set_xlabel('LR_ratio')
ax.set_ylabel('tipang')
ax.set_zlabel('Kratio / sin(x)*cos(y)')
ax.set_title('Comparison: Data Surface vs. Mathematical Surface')

# Add colorbar for one surface (optional)
fig.colorbar(surf1, ax=ax, shrink=0.5, aspect=10, label='Kratio')

plt.show()
