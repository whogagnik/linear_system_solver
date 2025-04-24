import pytest
import sys, os

# Чтобы pytest видел папку-репозиторий
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from solver.linear_system import LinearSystemSolver

# ─── FIXTURES ──────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_system():
    A = [[3, 2], [1, 2]]
    b = [5, 5]
    expected = [0, 2.5]
    return A, b, expected

@pytest.fixture
def spd_system():
    A = [
        [4, 1, 1],
        [1, 3, -1],
        [1, -1, 2]
    ]
    b = [5, -4, 6]
    expected = [1, -1, 2]
    return A, b, expected

@pytest.fixture
def dd_system():
    A = [
        [5, 2, 1],
        [1, 4, 1],
        [2, 1, 6]
    ]
    b = [8, 6, 9]
    expected = [1, 1, 1]
    return A, b, expected

@pytest.fixture
def diagonal_system():
    A = [[2,0,0],[0,3,0],[0,0,4]]
    b = [2,6,12]
    expected = [1,2,3]
    return A, b, expected

# ─── TEST DIRECT METHODS ───────────────────────────────────────────────────────

def test_gauss(simple_system):
    A, b, expected = simple_system
    solver = LinearSystemSolver(A, b)
    x = solver.solve(method=solver.GAUSS)
    assert x == pytest.approx(expected, rel=1e-6)

def test_lu(simple_system):
    A, b, expected = simple_system
    solver = LinearSystemSolver(A, b)
    x = solver.solve(method=solver.LU)
    assert x == pytest.approx(expected, rel=1e-6)

def test_inverse(diagonal_system):
    A, b, expected = diagonal_system
    solver = LinearSystemSolver(A, b)
    x = solver.solve(method=solver.INVERSE)
    assert x == pytest.approx(expected, rel=1e-6)

# ─── TEST SPD METHODS ──────────────────────────────────────────────────────────

def test_cholesky(spd_system):
    A, b, expected = spd_system
    solver = LinearSystemSolver(A, b)
    x = solver.solve(method=solver.CHOLESKY)
    assert x == pytest.approx(expected, rel=1e-6)

def test_conjugate_gradient(spd_system):
    A, b, expected = spd_system
    solver = LinearSystemSolver(A, b)
    x, norms = solver.solve(
        method=solver.CONJUGATE_GRADIENT,
        collect_norms=True, tol=1e-10, max_iter=500
    )
    assert x == pytest.approx(expected, rel=1e-6)
    # проверяем монотонное убывание невязки
    assert all(norms[i] >= norms[i+1] for i in range(len(norms)-1))
    assert norms[-1] < 1e-8

# ─── TEST ITERATIVE ON DD ──────────────────────────────────────────────────────

@pytest.mark.parametrize("method", [
    LinearSystemSolver.JACOBI,
    LinearSystemSolver.GAUSS_SEIDEL,
    LinearSystemSolver.SIMPLE_ITERATION,
    LinearSystemSolver.SOR
])
def test_iterative_methods(dd_system, method):
    A, b, expected = dd_system
    solver = LinearSystemSolver(A, b)
    x, norms = solver.solve(
        method=method, collect_norms=True, tol=1e-10, max_iter=1000
    )
    assert x == pytest.approx(expected, rel=1e-6)
    assert norms[0] > norms[-1]
    assert norms[-1] < 1e-8

def test_simple_iteration_fallback():
    A = [[1, 0.99], [0.99, 1]]
    b = [1.99, 1.99]
    solver = LinearSystemSolver(A, b)
    x, norms = solver.solve(
        method=solver.SIMPLE_ITERATION, collect_norms=True,
        tol=1e-6, max_iter=10
    )
    # должен откатиться на SOR и всё же сойтись
    assert x == pytest.approx([1,1], rel=1e-2)
    assert norms[-1] < norms[0]

# ─── TEST AUTO SELECTION ───────────────────────────────────────────────────────

def test_auto_spd_uses_cholesky(spd_system):
    A, b, expected = spd_system
    solver = LinearSystemSolver(A, b)
    x = solver.solve(method=solver.AUTO)
    assert x == pytest.approx(expected, rel=1e-6)

def test_auto_dd_uses_gauss_seidel(dd_system):
    A, b, expected = dd_system
    solver = LinearSystemSolver(A, b)
    x = solver.solve(method=solver.AUTO)
    assert x == pytest.approx(expected, rel=1e-6)

def test_auto_simple_uses_lu(simple_system):
    A, b, expected = simple_system
    solver = LinearSystemSolver(A, b)
    x = solver.solve(method=solver.AUTO)
    assert x == pytest.approx(expected, rel=1e-6)

# ─── TEST MISC ─────────────────────────────────────────────────────────────────

def test_compute_residual_zero(diagonal_system):
    A, b, expected = diagonal_system
    solver = LinearSystemSolver(A, b)
    residual, norm = solver.compute_residual(expected)
    assert all(abs(r) < 1e-12 for r in residual)
    assert norm == pytest.approx(0.0, abs=1e-12)

def test_invalid_method(simple_system):
    A, b, _ = simple_system
    solver = LinearSystemSolver(A, b)
    with pytest.raises(ValueError):
        solver.solve(method="not_a_method")

def test_singular_raises():
    A = [[2,4],[1,2]]
    b = [6,3]
    solver = LinearSystemSolver(A, b)
    with pytest.raises(ValueError):
        solver.solve(method=solver.GAUSS)

def test_jacobi_requires_dd():
    A = [[1,2],[3,4]]
    b = [3,7]
    solver = LinearSystemSolver(A, b)
    with pytest.raises(ValueError):
        solver.solve(method=solver.JACOBI)

def test_cg_requires_spd():
    A = [[1,2],[2,1]]
    b = [3,3]
    solver = LinearSystemSolver(A, b)
    with pytest.raises(ValueError):
        solver.solve(method=solver.CONJUGATE_GRADIENT)
