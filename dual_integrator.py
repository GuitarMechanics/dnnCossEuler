import sympy as sp

thp, thd, tp, td, lp, ld, alp, r = sp.symbols(r'\theta_p \theta_d t_p t_d l_p l_d \alpha r')

kda = td / (ld * r)
sp.pprint(kda)