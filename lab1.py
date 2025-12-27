import sys
import math
import numpy as np
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QTextEdit, QSizePolicy, QGroupBox)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# === ФУНКЦІЯ: x^3 - 3 - 10*ln(x) = 0 ===
def f(x):
    # ln(x) визначений тільки для x > 0
    if x <= 0:
        raise ValueError("f(x) невизначена для x <= 0 (ln x)")
    return x**3 - 3 - 10 * math.log(x)

# === ПОХІДНА: 3x^2 - 10/x ===
def df(x):
    if x <= 0:
        raise ValueError("df(x) невизначена для x <= 0")
    return 3 * x**2 - 10 / x

# === Допоміжні функції пошуку ===
def find_bracket(a, b, n_sub=200):
    """Шукає підінтервал, де функція змінює знак."""
    xs = np.linspace(a, b, n_sub + 1)
    
    # Фільтруємо недопустимі значення (x <= 0)
    valid_xs = [x for x in xs if x > 1e-9]
    if not valid_xs:
        return None
        
    try:
        fa = f(valid_xs[0])
        for i in range(len(valid_xs) - 1):
            fb = f(valid_xs[i + 1])
            if fa * fb <= 0:
                return float(valid_xs[i]), float(valid_xs[i + 1])
            fa = fb
    except ValueError:
        pass
    return None

def bisection(a, b, tol, max_iter=1000):
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        found = find_bracket(a, b, n_sub=500)
        if found is None:
            raise ValueError("Не знайдено зміни знаку f(x) на [a,b].")
        a, b = found
        fa, fb = f(a), f(b)

    n = 0
    while abs(b - a) > tol and n < max_iter:
        n += 1
        c = (a + b) / 2.0
        try:
            fc = f(c)
        except ValueError:
            # Якщо потрапили в недопустиму область, зміщуємось
            a = c + tol
            continue
            
        if abs(fc) < tol:
            return c, n
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return (a + b) / 2.0, n

def newton(x0, tol, max_iter=1000):
    x = x0
    for n in range(max_iter):
        try:
            fx = f(x)
            dfx = df(x)
        except ValueError:
             raise ValueError("Вихід за межі області визначення (x<=0).")

        if abs(dfx) < 1e-14:
            raise ValueError("Похідна ≈ 0.")
            
        x_new = x - fx / dfx
        
        # Захист від від'ємних значень для логарифма
        if x_new <= 0:
            x_new = 1e-3 # Повертаємось в допустиму область
            
        if abs(x_new - x) < tol:
            return x_new, n + 1
        x = x_new
    raise ValueError("Метод Ньютона не збігся.")

def iteration(x0, lam, tol, max_iter=10000):
    x = x0
    for n in range(max_iter):
        try:
            fx = f(x)
        except ValueError:
            raise ValueError("x <= 0")
            
        x_new = x - lam * fx
        
        if x_new <= 0:
             raise ValueError("Метод ітерацій вийшов в x <= 0.")

        if abs(x_new - x) < tol:
            return x_new, n + 1
        x = x_new
    raise ValueError("Метод ітерацій не збігся.")

# === ІНТЕРФЕЙС ===
class Solver(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Розв’язання: 10 ln x = x^3 - 3')
        self.setGeometry(100, 100, 1000, 700)
        
        # Головний вертикальний лейаут
        main_layout = QVBoxLayout(self)

        # === 1. ВЕРХНЯ ПАНЕЛЬ (НАЛАШТУВАННЯ) ===
        controls_group = QGroupBox("Параметри рівняння та методу")
        controls_layout = QHBoxLayout()
        
        # Створюємо поля
        self.a_edit = QLineEdit('0.1')   # Початок (має бути > 0)
        self.b_edit = QLineEdit('3.0')   # Кінець
        self.x0_edit = QLineEdit('2.0')  # Початкове наближення (для 2.22)
        self.lam_edit = QLineEdit('0.01') # Лямбда
        self.tol_edit = QLineEdit('1e-4') # Точність

        # Додаємо їх в лейаут
        for text, widget in [('a:', self.a_edit), ('b:', self.b_edit),
                             ('x0:', self.x0_edit), ('λ:', self.lam_edit),
                             ('ε (точність):', self.tol_edit)]:
            vbox = QVBoxLayout()
            lbl = QLabel(text)
            lbl.setStyleSheet("font-weight: bold;")
            vbox.addWidget(lbl)
            vbox.addWidget(widget)
            controls_layout.addLayout(vbox)
        
        controls_group.setLayout(controls_layout)
        main_layout.addWidget(controls_group)

        # === 2. КНОПКИ (Під налаштуваннями) ===
        btn_layout = QHBoxLayout()
        self.plot_btn = QPushButton('Побудувати графік')
        self.solve_btn = QPushButton('РОЗВ’ЯЗАТИ')
        self.plot_btn.setMinimumHeight(40)
        self.solve_btn.setMinimumHeight(40)
        self.solve_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        
        self.plot_btn.clicked.connect(self.plot_graph)
        self.solve_btn.clicked.connect(self.solve)
        
        btn_layout.addWidget(self.plot_btn)
        btn_layout.addWidget(self.solve_btn)
        main_layout.addLayout(btn_layout)

        # === 3. НИЖНЯ ПАНЕЛЬ (ГРАФІК + РЕЗУЛЬТАТ) ===
        bottom_layout = QHBoxLayout()
        
        # Текстове поле (ліворуч, вужче)
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setFixedWidth(300)
        self.output.setPlaceholderText("Тут будуть результати обчислень...")
        
        # Графік (праворуч, на всю ширину)
        self.fig = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        bottom_layout.addWidget(self.output)
        bottom_layout.addWidget(self.canvas)
        
        # Додаємо нижню панель в головний лейаут
        main_layout.addLayout(bottom_layout)

        self.last_xs = None
        self.last_ys = None

    def plot_graph(self):
        try:
            a = float(self.a_edit.text())
            b = float(self.b_edit.text())
            if a <= 0: 
                a = 0.01 # Коригуємо для логарифма
                self.output.append("Увага: a змінено на 0.01 (x > 0)")
            if a >= b:
                raise ValueError("Потрібно a < b.")
        except Exception as e:
            self.output.setText(f"Помилка даних: {e}")
            return

        # Генеруємо точки
        xs = np.linspace(a, b, 800)
        ys = []
        valid_xs = []
        for x in xs:
            try:
                val = f(x)
                ys.append(val)
                valid_xs.append(x)
            except:
                pass # Пропускаємо недопустимі точки

        self.last_xs = valid_xs
        self.last_ys = ys

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.plot(valid_xs, ys, label=r'$f(x) = x^3 - 3 - 10\ln(x)$', color='blue')
        ax.axhline(0, color='black', linewidth=1)
        ax.grid(True, linestyle='--')
        
        # Підпис осей та легенда
        ax.set_title("Графік функції")
        ax.legend()
        
        self.canvas.draw()
        self.output.append("Графік оновлено.")

    def solve(self):
        # Якщо графік старий або відсутній, оновимо його
        self.plot_graph()
        
        try:
            a = float(self.a_edit.text())
            b = float(self.b_edit.text())
            x0 = float(self.x0_edit.text())
            lam = float(self.lam_edit.text())
            tol = float(self.tol_edit.text())
            
            # Корекція для логарифма
            if a <= 0: a = 0.001
        except Exception as e:
            self.output.setText(f"Помилка параметрів: {e}")
            return

        res_text = "=== РЕЗУЛЬТАТИ ===\n"
        
        # 1. Бісекція
        try:
            r_bis, n_bis = bisection(a, b, tol)
            res_text += f"Бісекція:\n  x ≈ {r_bis:.6f}\n  (ітер: {n_bis})\n\n"
        except Exception as e:
            r_bis = None
            res_text += f"Бісекція: Помилка ({e})\n\n"

        # 2. Ньютон
        try:
            r_new, n_new = newton(x0, tol)
            res_text += f"Ньютон:\n  x ≈ {r_new:.6f}\n  (ітер: {n_new})\n\n"
        except Exception as e:
            r_new = None
            res_text += f"Ньютон: Помилка ({e})\n\n"

        # 3. Ітерації
        try:
            r_it, n_it = iteration(x0, lam, tol)
            res_text += f"Ітерації:\n  x ≈ {r_it:.6f}\n  (ітер: {n_it})\n"
        except Exception as e:
            r_it = None
            res_text += f"Ітерації: Помилка ({e})\n"

        self.output.setText(res_text)

        # Малюємо знайдені точки на графіку
        ax = self.fig.gca()
        # Очищаємо старі маркери (якщо є, але простіше перемалювати все)
        if self.last_xs:
            ax.clear()
            ax.plot(self.last_xs, self.last_ys, label=r'$x^3 - 3 - 10\ln(x)$')
            ax.axhline(0, color='black', linewidth=1)
            ax.grid(True, linestyle='--')

        # Додаємо точки
        found_roots = []
        if r_bis: found_roots.append(('Bisection', r_bis, 'ro'))
        if r_new: found_roots.append(('Newton', r_new, 'gs'))
        if r_it:  found_roots.append(('Iter', r_it, 'b^'))

        # Щоб не накладати легенду багато разів
        for name, rx, style in found_roots:
            try:
                ry = f(rx)
                ax.plot(rx, ry, style, markersize=8, label=f'{name}: {rx:.4f}')
            except: pass
            
        ax.legend()
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Solver()
    win.show()
    sys.exit(app.exec())