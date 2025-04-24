import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from solver.linear_system import LinearSystemSolver

@pytest.fixture
def simple_system():
    # 3x + 2y = 5
    #  x + 2y = 5
    A = [[3, 2],
         [1, 2]]
    b = [5, 5]
    # точное решение x=0, y=2.5
    return A, b, [0, 2.5]

@pytest.fixture
def spd_system():
    # SPD-система, решение x=[1,-1,2]
    A = [[4, 1, 1],
         [1, 3, -1],
         [1, -1, 2]]
    # b = A·x
    b = [5, -4, 6]
    return A, b, [1, -1, 2]

@pytest.fixture
def dd_system():
    # диагонально-доминируемая, решение x=[1,1,1]
    A = [[5,2,1],
         [1,4,1],
         [2,1,6]]
    b = [8,6,9]
    return A, b, [1,1,1]

def as_vec(x):
    # если вернулся скаляр, превращаем в вектор одинаковых значений
    return x if isinstance(x, list) else [x]*2

def test_direct_methods_simple(simple_system):
    A, b, expected = simple_system
    solver = LinearSystemSolver(A, b)

    xg = solver.solve(method=solver.GAUSS)
    assert as_vec(xg) == pytest.approx(expected, rel=1e-6)

    xl = solver.solve(method=solver.LU)
    assert as_vec(xl) == pytest.approx(expected, rel=1e-6)

    xi = solver.solve(method=solver.INVERSE)
    assert as_vec(xi) == pytest.approx(expected, rel=1e-6)

def test_cholesky_and_cg(spd_system):
    A, b, expected = spd_system
    solver = LinearSystemSolver(A, b)

    xc = solver.solve(method=solver.CHOLESKY)
    assert xc == pytest.approx(expected, rel=1e-6)

    xcg, norms = solver.solve(method=solver.CONJUGATE_GRADIENT, collect_norms=True, tol=1e-10, max_iter=500)
    assert xcg == pytest.approx(expected, rel=1e-6)
    assert all(norms[i] >= norms[i+1] for i in range(len(norms)-1))
    assert norms[-1] < 1e-8

def test_iterative_on_dd(dd_system):
    A, b, expected = dd_system
    solver = LinearSystemSolver(A, b)

    for method in (solver.JACOBI, solver.GAUSS_SEIDEL, solver.SIMPLE_ITERATION, solver.SOR):
        x, norms = solver.solve(method=method, collect_norms=True, tol=1e-10, max_iter=1000)
        assert x == pytest.approx(expected, rel=1e-6)
        assert norms[0] > norms[-1]
        assert norms[-1] < 1e-8

def test_singular_raises():
    A = [[2,4],[1,2]]
    b = [6,3]
    solver = LinearSystemSolver(A,b)
    with pytest.raises(ValueError):
        solver.solve(method=solver.GAUSS)

def test_jacobi_requires_dd():
    A = [[1,2],[3,4]]
    b = [3,7]
    solver = LinearSystemSolver(A,b)
    with pytest.raises(ValueError):
        solver.solve(method=solver.JACOBI)

def test_cg_requires_spd():
    A = [[1,2],[2,1]]
    b = [3,3]
    solver = LinearSystemSolver(A,b)
    with pytest.raises(ValueError):
        solver.solve(method=solver.CONJUGATE_GRADIENT)

def test_simple_fallback_to_sor():
    A = [[1,0.9],[0.9,1]]
    b = [1.9,1.9]
    solver = LinearSystemSolver(A,b)
    x, norms = solver.solve(method=solver.SIMPLE_ITERATION, collect_norms=True, max_iter=50, tol=1e-6)
    assert x == pytest.approx([1,1], rel=1e-4)
    assert norms[-1] < 1e-4
