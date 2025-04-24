# gui/view.py
import tkinter as tk
from tkinter import ttk, simpledialog, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class LinearSystemView(ttk.Frame):
    """
    Представление: рисует все виджеты и предоставляет методы для отображения данных.
    """
    def __init__(self, master=None):
        self.master = master or tk.Tk()
        super().__init__(self.master)
        self.master.title("Linear System Solver MVC")
        self.master.geometry("1000x750")
        self.pack(fill="both", expand=True)
        self._build_widgets()

    def _build_widgets(self):
        # -- Input Mode Frame --
        self.mode_frame = ttk.LabelFrame(self, text="Input Mode")
        self.mode_frame.pack(fill="x", padx=10, pady=5)
        self.mode_var = tk.StringVar(value="manual")
        for text, mode in [("Manual","manual"), ("File","file"),
                           ("Random","random"), ("Dominant","dominant"),
                           ("Grid","grid")]:
            rb = ttk.Radiobutton(self.mode_frame, text=text,
                                 value=mode, variable=self.mode_var)
            rb.pack(side="left", padx=5)

        # -- Manual Input Frame (hidden by default) --
        self.manual_frame = ttk.Frame(self)
        ttk.Label(self.manual_frame, text="A (`;`-rows):").grid(row=0, column=0, sticky="w")
        self.entry_A = ttk.Entry(self.manual_frame, width=70)
        self.entry_A.grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(self.manual_frame, text="b (comma-separated):").grid(row=1, column=0, sticky="w")
        self.entry_b = ttk.Entry(self.manual_frame, width=70)
        self.entry_b.grid(row=1, column=1, padx=5, pady=2)
        self.manual_frame.pack_forget()

        # -- Grid Input Frame (hidden by default) --
        self.grid_frame = ttk.Frame(self)
        self.grid_frame.pack_forget()

        # -- Method Selection Frame --
        self.method_frame = ttk.Frame(self)
        self.method_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(self.method_frame, text="Method:").pack(side="left")
        self.method_box = ttk.Combobox(self.method_frame, state="readonly")
        self.method_box.pack(side="left", padx=5)

        # -- Buttons Frame --
        self.btn_frame = ttk.Frame(self)
        self.btn_frame.pack(fill="x", padx=10, pady=5)
        self.solve_button = ttk.Button(self.btn_frame, text="Solve")
        self.solve_button.pack(side="left", padx=5)
        self.conv_button = ttk.Button(self.btn_frame, text="Convergence")
        self.conv_button.pack(side="left", padx=5)
        self.heatmap_button = ttk.Button(self.btn_frame, text="Heatmap")
        self.heatmap_button.pack(side="left", padx=5)

        # -- Output Text Widget --
        self.output_text = tk.Text(self, height=10)
        self.output_text.pack(fill="both", padx=10, pady=5)

        # -- Plot Frame --
        self.plot_frame = ttk.Frame(self)
        self.plot_frame.pack(fill="both", expand=True, padx=10, pady=5)

    def display_matrix(self, A, b):
        """Отобразить матрицу A и вектор b."""
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, "Matrix A:")
        for row in A:
            self.output_text.insert(tk.END, f"{[round(v,4) for v in row]}")
        self.output_text.insert(tk.END, "\nVector b:")
        self.output_text.insert(tk.END, f"{[round(v,4) for v in b]}\n")

    def display_solution(self, x, residual_norm):
        """Отобразить решение x и норму невязки."""
        self.output_text.insert(tk.END, "Solution x:\n")
        for i, val in enumerate(x):
            self.output_text.insert(tk.END, f"    x[{i}] = {val:.6f}\n")
        self.output_text.insert(tk.END, f"Residual norm = {residual_norm:.2e}\n")

    def embed_figure(self, fig):
        """Встроить график из matplotlib в plot_frame."""
        for child in self.plot_frame.winfo_children():
            child.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)