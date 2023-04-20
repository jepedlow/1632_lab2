import lab2_setup
import numpy as np
import sympy
import time


phi, p, theta, q, psi, r = sympy.symbols('phi, p, theta, q, psi, r')
f = sympy.Matrix([
    p + sympy.sin(phi)*sympy.tan(theta)*q + sympy.cos(phi)*sympy.tan(theta)*r,
    0,
    sympy.cos(phi)*p - sympy.sin(phi)*q,
    0,
    sympy.sin(phi)*sympy.sec(theta)*q + sympy.cos(phi)*sympy.sec(theta)*r,
    0
])

fjac = f.jacobian([phi, p, theta, q, psi, r])
print(fjac)


