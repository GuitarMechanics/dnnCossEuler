import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Load the CSV data
file_path = 'curvature_reginfos.csv'
data = pd.read_csv(file_path)

# Extract x, y, z
x = data['LR_ratio'].values
y = data['tipang'].values
z = data['Kratio'].values

# Adjust target to remove fixed intercept
# Stack x and y for polynomial feature generation
XY = np.vstack((x, y)).T

# Generate polynomial features (2차 다항식까지)
poly = PolynomialFeatures(degree=2)
XY_poly = poly.fit_transform(XY)

# Fit regression model without fitting intercept
model = LinearRegression()
model.fit(XY_poly, z)

# 모델 계수 출력
coeff_names = poly.get_feature_names_out(['x', 'y'])
for name, coef in zip(coeff_names, model.coef_):
    print(f'{name}: {coef:.3f}')

# 예측용 그리드 생성
xfit = np.linspace(x.min(), x.max(), 100)
yfit = np.linspace(y.min(), y.max(), 100)
xfit, yfit = np.meshgrid(xfit, yfit)
XYfit = np.vstack((xfit.ravel(), yfit.ravel())).T

# Predict with fixed intercept added back
Zfit = model.predict(poly.transform(XYfit)).reshape(xfit.shape)

print(f'R2 score: {r2_score(z, model.predict(XY_poly)):.5f}')
# 시각화
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 원래 데이터
ax.scatter(x, y, z, c='k', s=5)

# 피팅된 곡면
ax.plot_surface(xfit, yfit, Zfit, alpha=0.6, cmap='coolwarm', edgecolor='none')

# 축 라벨
ax.set_xlabel('LR_ratio')
ax.set_ylabel('tipang')
ax.set_zlabel('Kratio')
ax.set_title('Fitted Polynomial Surface (degree=2, Intercept fixed = 1)')

plt.show()