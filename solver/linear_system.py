# solver/linear_system.py

import math
import time

class LinearSystemSolver:
    GAUSS = 'gauss'
    JACOBI = 'jacobi'
    GAUSS_SEIDEL = 'gauss_seidel'
    LU = 'lu'
    CHOLESKY = 'cholesky'
    SIMPLE_ITERATION = 'simple_iteration'
    SOR = 'sor'
    CONJUGATE_GRADIENT = 'conjugate_gradient'
    INVERSE = 'inverse'
    AUTO = 'auto'

    def __init__(self, A, b):
        self.A = [row[:] for row in A]
        self.b = b[:]
        self.n = len(A)
        self.last_norms = None
        self.last_method = None

    def is_diagonally_dominant(self):
        return all(
            abs(self.A[i][i]) >= sum(abs(self.A[i][j]) for j in range(self.n) if j != i)
            for i in range(self.n)
        )

    def is_symmetric(self):
        return all(
            self.A[i][j] == self.A[j][i]
            for i in range(self.n) for j in range(i+1, self.n)
        )

    def is_positive_definite(self):
        try:
            self._solve_cholesky()
            return True
        except:
            return False

    def compute_residual(self, x):
        res = [sum(self.A[i][j]*x[j] for j in range(self.n)) - self.b[i] for i in range(self.n)]
        try:
            norm = math.sqrt(sum(r*r for r in res))
        except OverflowError:
            norm = float('inf')
        return res, norm

    def auto_select_method(self):
        # Для SPD систем — Cholesky
        if self.is_symmetric() and self.is_positive_definite():
            return self.CHOLESKY
        # Для диагонально-доминируемых больших систем — Gauss-Seidel
        if self.is_diagonally_dominant() and self.n > 2:
            return self.GAUSS_SEIDEL
        # Во всех остальных случаях — LU
        return self.LU

    def solve(self, method=AUTO, tol=1e-10, max_iter=1000,
              log=False, x0=None, collect_norms=False, sor_omega=1.25):
        if method == self.AUTO:
            method = self.auto_select_method()
            if log:
                print(f"[auto] selected {method}")

        if method in (self.JACOBI, self.GAUSS_SEIDEL) and not self.is_diagonally_dominant():
            raise ValueError(f"{method} requires diagonally dominant matrix")
        if method == self.CONJUGATE_GRADIENT and not (self.is_symmetric() and self.is_positive_definite()):
            raise ValueError("Conjugate Gradient requires SPD matrix")

        if x0 is None:
            x0 = [0.0]*self.n

        iterative = {
            self.JACOBI, self.GAUSS_SEIDEL,
            self.SIMPLE_ITERATION, self.SOR,
            self.CONJUGATE_GRADIENT
        }

        start = time.time()
        if method == self.GAUSS:
            x = self._solve_gauss()
        elif method == self.LU:
            x = self._solve_lu()
        elif method == self.CHOLESKY:
            x = self._solve_cholesky()
        elif method == self.INVERSE:
            x = self._solve_inverse()
        elif method in iterative:
            # запускаем итерационный метод, получаем либо x, либо (x, norms)
            try:
                if method == self.SIMPLE_ITERATION:
                    result = self._solve_simple_iteration(tol, max_iter, log, x0, collect_norms)
                elif method == self.SOR:
                    result = self._solve_sor(tol, max_iter, log, x0, collect_norms, sor_omega)
                else:
                    func = {
                        self.JACOBI: self._solve_jacobi,
                        self.GAUSS_SEIDEL: self._solve_gauss_seidel,
                        self.CONJUGATE_GRADIENT: self._solve_conjugate_gradient
                    }[method]
                    result = func(tol, max_iter, log, x0, collect_norms)
            except RuntimeError:
                # fallback для простейшей итерации
                if method == self.SIMPLE_ITERATION:
                    if log:
                        print("Simple iteration failed, falling back to SOR")
                    result = self._solve_sor(tol, max_iter, log, x0, collect_norms, sor_omega)
                    method = self.SOR
                else:
                    raise

            # Распаковываем в зависимости от флага collect_norms
            if collect_norms:
                x, norms = result
                self.last_norms = norms
                self.last_method = method
            else:
                x = result
                self.last_norms = None
                self.last_method = None
        else:
            raise ValueError(f"Unknown method: {method}")

        if log:
            print(f"Time: {time.time() - start:.6f}s")

        return (x, self.last_norms) if collect_norms and method in iterative else x

    # — Direct methods —

    def _solve_gauss(self, eps=1e-12):
        A = [row[:] for row in self.A]
        b = self.b[:]
        for i in range(self.n):
            pivot = max(range(i, self.n), key=lambda r: abs(A[r][i]))
            if abs(A[pivot][i]) < eps:
                raise ValueError("Matrix is singular")
            A[i], A[pivot] = A[pivot], A[i]
            b[i], b[pivot] = b[pivot], b[i]
            for j in range(i+1, self.n):
                f = A[j][i] / A[i][i]
                for k in range(i, self.n):
                    A[j][k] -= f * A[i][k]
                b[j] -= f * b[i]
        x = [0.0]*self.n
        for i in reversed(range(self.n)):
            s = sum(A[i][j]*x[j] for j in range(i+1, self.n))
            x[i] = (b[i] - s) / A[i][i]
        return x

    def _solve_lu(self):
        n = self.n
        L = [[0.0]*n for _ in range(n)]
        U = [[0.0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                U[i][j] = self.A[i][j] - sum(L[i][k]*U[k][j] for k in range(i))
            L[i][i] = 1.0
            for j in range(i+1, n):
                L[j][i] = (self.A[j][i] - sum(L[j][k]*U[k][i] for k in range(i))) / U[i][i]
        y = [0.0]*n
        for i in range(n):
            y[i] = self.b[i] - sum(L[i][j]*y[j] for j in range(i))
        x = [0.0]*n
        for i in reversed(range(n)):
            x[i] = (y[i] - sum(U[i][j]*x[j] for j in range(i+1, n))) / U[i][i]
        return x

    def _solve_cholesky(self):
        if not self.is_symmetric():
            raise ValueError("Matrix not symmetric")
        n = self.n
        L = [[0.0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1):
                s = sum(L[i][k]*L[j][k] for k in range(j))
                if i == j:
                    v = self.A[i][i] - s
                    if v <= 0:
                        raise ValueError("Matrix not positive definite")
                    L[i][i] = math.sqrt(v)
                else:
                    L[i][j] = (self.A[i][j] - s) / L[j][j]
        y = [0.0]*n
        for i in range(n):
            y[i] = (self.b[i] - sum(L[i][j]*y[j] for j in range(i))) / L[i][i]
        x = [0.0]*n
        for i in reversed(range(n)):
            x[i] = (y[i] - sum(L[j][i]*x[j] for j in range(i+1, n))) / L[i][i]
        return x

    def _solve_inverse(self):
        n = self.n
        L = [[0.0]*n for _ in range(n)]
        U = [[0.0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                U[i][j] = self.A[i][j] - sum(L[i][k]*U[k][j] for k in range(i))
            L[i][i] = 1.0
            for j in range(i+1, n):
                L[j][i] = (self.A[j][i] - sum(L[j][k]*U[k][i] for k in range(i))) / U[i][i]
        inv = [[0.0]*n for _ in range(n)]
        for k in range(n):
            y = [0.0]*n
            for i in range(n):
                y[i] = ((1.0 if i == k else 0.0) - sum(L[i][j]*y[j] for j in range(i))) / L[i][i]
            x_col = [0.0]*n
            for i in reversed(range(n)):
                x_col[i] = (y[i] - sum(U[i][j]*x_col[j] for j in range(i+1, n))) / U[i][i]
            for i in range(n):
                inv[i][k] = x_col[i]
        return [sum(inv[i][j]*self.b[j] for j in range(n)) for i in range(n)]

    # — Iterative methods —

    def _solve_jacobi(self, tol, max_iter, log, x0, collect_norms):
        x = x0[:]; norms = []
        for it in range(max_iter):
            x_new = [
                (self.b[i] - sum(self.A[i][j]*x[j] for j in range(self.n) if j != i))
                / self.A[i][i]
                for i in range(self.n)
            ]
            diff = math.sqrt(sum((x_new[i] - x[i])**2 for i in range(self.n)))
            if collect_norms: norms.append(diff)
            if log: print(f"Jacobi {it+1}: {diff:.3e}")
            if diff < tol: return (x_new, norms) if collect_norms else x_new
            x = x_new
        raise RuntimeError("Jacobi did not converge")

    def _solve_gauss_seidel(self, tol, max_iter, log, x0, collect_norms):
        x = x0[:]; norms = []
        for it in range(max_iter):
            for i in range(self.n):
                s1 = sum(self.A[i][j]*x[j] for j in range(i))
                s2 = sum(self.A[i][j]*x0[j] for j in range(i+1, self.n))
                x[i] = (self.b[i] - s1 - s2) / self.A[i][i]
            diff = math.sqrt(sum((x[i] - x0[i])**2 for i in range(self.n)))
            if collect_norms: norms.append(diff)
            if log: print(f"Gauss-Seidel {it+1}: {diff:.3e}")
            if diff < tol: return (x[:], norms) if collect_norms else x[:]
            x0 = x[:]
        raise RuntimeError("Gauss-Seidel did not converge")

    def _solve_simple_iteration(self, tol, max_iter, log, x0, collect_norms):
        row_sums = [sum(abs(self.A[i][j]) for j in range(self.n)) for i in range(self.n)]
        tau = 1.0 / max(row_sums)
        x = x0[:]; norms = []
        for it in range(max_iter):
            x_new = [
                x[i] - tau*(sum(self.A[i][j]*x[j] for j in range(self.n)) - self.b[i])
                for i in range(self.n)
            ]
            diff = math.sqrt(sum((x_new[i] - x[i])**2 for i in range(self.n)))
            if collect_norms: norms.append(diff)
            if log: print(f"SimpleIter {it+1}: {diff:.3e}, tau={tau:.3e}")
            if diff < tol: return (x_new, norms) if collect_norms else x_new
            x = x_new
        raise RuntimeError("Simple iteration did not converge")

    def _solve_sor(self, tol, max_iter, log, x0, collect_norms, omega):
        x = x0[:]; norms = []
        for it in range(max_iter):
            for i in range(self.n):
                s = sum(self.A[i][j]*x[j] for j in range(self.n) if j != i)
                x_new_i = (self.b[i] - s) / self.A[i][i]
                x[i] = x[i] + omega*(x_new_i - x[i])
            diff = math.sqrt(sum((x[i] - x0[i])**2 for i in range(self.n)))
            if collect_norms: norms.append(diff)
            if log: print(f"SOR {it+1}: {diff:.3e}")
            if diff < tol: return (x[:], norms) if collect_norms else x[:]
            x0 = x[:]
        raise RuntimeError("SOR did not converge")

    def _solve_conjugate_gradient(self, tol, max_iter, log, x0, collect_norms):
        if not self.is_symmetric() or not self.is_positive_definite():
            raise ValueError("CG requires SPD matrix")
        x = x0[:]
        r = [self.b[i] - sum(self.A[i][j]*x[j] for j in range(self.n)) for i in range(self.n)]
        p = r[:]
        rs_old = sum(r_i*r_i for r_i in r)
        norms = []
        for it in range(max_iter):
            Ap = [sum(self.A[i][j]*p[j] for j in range(self.n)) for i in range(self.n)]
            alpha = rs_old / sum(p[i]*Ap[i] for i in range(self.n))
            x = [x[i] + alpha*p[i] for i in range(self.n)]
            r = [r[i] - alpha*Ap[i] for i in range(self.n)]
            rs_new = sum(r_i*r_i for r_i in r)
            diff = math.sqrt(rs_new)
            if collect_norms: norms.append(diff)
            if log: print(f"CG {it+1}: {diff:.3e}")
            if diff < tol: return (x[:], norms) if collect_norms else x[:]
            p = [r[i] + (rs_new/rs_old)*p[i] for i in range(self.n)]
            rs_old = rs_new
        raise RuntimeError("Conjugate Gradient did not converge")
