from sympy import symbols, Eq, solve

# 定义变量
a, b, c, x, y, z = symbols('a b c x y z')

# 定义方程
eq1 = Eq(a*x**2 + b*y - c*y, 1)
eq2 = Eq(x**2 - b*y*x + c*y, 2)
# 求解方程
solution = solve((eq1, eq2), (x, y))

print(solution)