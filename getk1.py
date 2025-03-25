import sympy as sp

Lb, r, t, k1, ka = sp.symbols('L, r, t, \kappa_1, \kappa_a')

lr = Lb/r
th = t/r
ka = t/(Lb*r)

kr = k1/ka

eqn = 3.56e-3*lr + 7.93e-2*th - 4e-5 * lr ** 2 + 4.03e-3 * lr  * th - 2.46e-3 * th**2 + 0.953 - kr

sol = sp.solve(eqn)
for val in sol:
    print(sp.latex(val))
