# gui/controller.py
import random
import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog, ttk
import matplotlib.pyplot as plt
from solver.linear_system import LinearSystemSolver
from gui.view import LinearSystemView
from gui.callbacks import callback

class LinearSystemController:
    def __init__(self):
        # Create root and view
        self.root = tk.Tk()
        self.view = LinearSystemView(master=self.root)
        self.last_A = None
        self.last_b = None
        self.last_norms = None
        self.last_method = None
        self.grid_entries = []
        self._setup()

    def _setup(self):
        methods = [
            LinearSystemSolver.GAUSS,
            LinearSystemSolver.JACOBI,
            LinearSystemSolver.GAUSS_SEIDEL,
            LinearSystemSolver.LU,
            LinearSystemSolver.CHOLESKY,
            LinearSystemSolver.SIMPLE_ITERATION,
            LinearSystemSolver.SOR,
            LinearSystemSolver.CONJUGATE_GRADIENT,
            LinearSystemSolver.INVERSE,
            LinearSystemSolver.AUTO
        ]
        self.view.method_box['values'] = methods
        self.view.method_box.set(LinearSystemSolver.AUTO)

        # Bind events
        self.view.solve_button.config(command=self.on_solve)
        self.view.conv_button.config(command=self.show_convergence)
        self.view.heatmap_button.config(command=self.show_heatmap)
        self.view.mode_var.trace_add('write', lambda *_: self.switch_mode())

        self.switch_mode()

    def switch_mode(self):
        mode = self.view.mode_var.get()
        # hide both
        self.view.manual_frame.pack_forget()
        self.view.grid_frame.pack_forget()

        if mode == 'manual':
            self.view.manual_frame.pack(fill='x', padx=10, pady=5, after=self.view.mode_frame)
        elif mode == 'grid':
            n = simpledialog.askinteger("Grid Size", "Enter n:", minvalue=2, maxvalue=15)
            if not n:
                return
            # Clear previous grid entries
            for w in self.view.grid_frame.winfo_children():
                w.destroy()
            self.grid_entries = []
            for i in range(n):
                row_entries = []
                for j in range(n):
                    e = ttk.Entry(self.view.grid_frame, width=5)
                    e.grid(row=i, column=j, padx=2, pady=2)
                    row_entries.append(e)
                b_e = ttk.Entry(self.view.grid_frame, width=5)
                b_e.grid(row=i, column=n, padx=5, pady=2)
                row_entries.append(b_e)
                self.grid_entries.append(row_entries)
            self.view.grid_frame.pack(fill='x', padx=10, pady=5, after=self.view.mode_frame)
        elif mode == 'file':
            path = filedialog.askopenfilename(filetypes=[("Text","*.txt")])
            if not path:
                return
            with open(path) as f:
                lines = [l.strip() for l in f if l.strip()]
            A = [list(map(float, l.split())) for l in lines[:-1]]
            b = list(map(float, lines[-1].split()))
            self.last_A, self.last_b = A, b
            self.view.display_matrix(A, b)
        else:  # random or dominant
            n = simpledialog.askinteger("Size", "Enter n:", minvalue=2, maxvalue=15)
            if not n:
                return
            A = [[random.uniform(-10,10) for _ in range(n)] for _ in range(n)]
            b = [random.uniform(-10,10) for _ in range(n)]
            if mode == 'dominant':
                for i in range(n):
                    A[i][i] = sum(abs(A[i][j]) for j in range(n) if j!=i)+1
            self.last_A, self.last_b = A, b
            self.view.display_matrix(A, b)

    def _get_input(self):
        mode = self.view.mode_var.get()
        if mode == 'manual':
            raw_A = self.view.entry_A.get().strip()
            raw_b = self.view.entry_b.get().strip()
            if not raw_A or not raw_b:
                raise ValueError("Manual input is empty")
            A = []
            for row in raw_A.split(';'):
                parts = [x for x in row.replace(',', ' ').split()]
                if not parts:
                    raise ValueError("Empty row in matrix A")
                A.append([float(x) for x in parts])
            b_parts = [x for x in raw_b.replace(',', ' ').split()]
            if len(b_parts) != len(A):
                raise ValueError("Length of b must match number of rows in A")
            b = [float(x) for x in b_parts]
        elif mode == 'grid':
            if not self.grid_entries:
                raise ValueError("Grid not initialized")
            A = []
            b = []
            for row in self.grid_entries:
                vals = [e.get().strip() for e in row]
                if any(v == '' for v in vals):
                    raise ValueError("Empty cell in grid input")
                row_vals = [float(v) for v in vals[:-1]]
                A.append(row_vals)
                b.append(float(vals[-1]))
        else:
            if self.last_A is None or self.last_b is None:
                raise ValueError("No matrix/vector loaded")
            A, b = self.last_A, self.last_b
        return A, b

    @callback('solve_button')
    def on_solve(self):
        try:
            A, b = self._get_input()
            self.view.display_matrix(A, b)
            method = self.view.method_box.get()
            solver = LinearSystemSolver(A, b)
            iterative = {solver.JACOBI, solver.GAUSS_SEIDEL,
                         solver.SIMPLE_ITERATION, solver.SOR,
                         solver.CONJUGATE_GRADIENT}
            if method in iterative:
                x, norms = solver.solve(
                    method=method, tol=1e-10,
                    max_iter=500, log=False,
                    x0=None, collect_norms=True)
                self.last_norms, self.last_method = norms, method
            else:
                x = solver.solve(method=method)
                self.last_norms, self.last_method = None, None
            _, res_norm = solver.compute_residual(x)
            self.view.display_solution(x, res_norm)
            self.last_A = A
        except Exception as e:
            messagebox.showerror("Error", str(e))

    @callback('conv_button')
    def show_convergence(self):
        if not getattr(self, 'last_norms', None):
            messagebox.showinfo("No Data", "No convergence data.")
            return
        fig, ax = plt.subplots()
        ax.plot(self.last_norms, marker='o')
        ax.set_yscale('log')
        ax.set_title(f"Convergence: {self.last_method}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Residual norm")
        self.view.embed_figure(fig)

    @callback('heatmap_button')
    def show_heatmap(self):
        if not getattr(self, 'last_A', None):
            messagebox.showinfo("No Data", "No matrix A.")
            return
        fig, ax = plt.subplots()
        im = ax.imshow(self.last_A, cmap='viridis', aspect='auto')
        fig.colorbar(im, ax=ax)
        ax.set_title("Heatmap of A")
        self.view.embed_figure(fig)

    def start(self):
        self.root.mainloop()