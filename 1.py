import numpy as np
import matplotlib.pyplot as plt

def f_orig(x):
    if 0 <= x < 2:
        return x/2
    elif 2 <= x <= 4:
        return 1
    else:
        return None

def S_general_exact(x):
    x_mod = x % 4
    # Приводим к [0,4)
    x_mod = np.where(x_mod < 0, x_mod + 4, x_mod)
    # Используем np.select для трёх условий
    condlist = [x_mod < 2, (x_mod >= 2) & (x_mod < 4), np.isclose(x_mod, 0) | np.isclose(x_mod, 4)]
    choicelist = [x_mod/2, 1, 0.5]
    return np.select(condlist, choicelist, default=0.5)

def S_general_partial(x, N):
    s = 0.75 * np.ones_like(x)
    for n in range(1, N+1):
        arg = n * np.pi * x / 2
        s += (-1/(n*np.pi)) * np.sin(arg)
        if n % 2 == 1:
            a_n = -2/(n*n * np.pi*np.pi)
            s += a_n * np.cos(arg)
    return s

def S_cos_exact(x):
    x_mod = x % 8
    x_mod = np.where(x_mod < 0, x_mod + 8, x_mod)
    x_mod = np.where(x_mod > 4, 8 - x_mod, x_mod)
    condlist = [x_mod < 2, (x_mod >= 2) & (x_mod <= 4)]
    choicelist = [x_mod/2, 1]
    return np.select(condlist, choicelist, default=1)

def S_cos_partial(x, N):
    s = 0.75 * np.ones_like(x)
    for n in range(1, N+1):
        arg = n * np.pi * x / 4
        if n % 4 == 0:
            an = 0
        elif n % 4 == 2:
            an = -8/(n*n * np.pi*np.pi)
        else:
            an = -4/(n*n * np.pi*np.pi)
        s += an * np.cos(arg)
    return s

def S_sin_exact(x):
    x_mod = x % 8
    x_mod = np.where(x_mod < 0, x_mod + 8, x_mod)
    sign = np.where(x_mod > 4, -1, 1)
    x_mod = np.where(x_mod > 4, 8 - x_mod, x_mod)
    val = np.select([x_mod < 2, (x_mod >= 2) & (x_mod < 4)], [x_mod/2, 1], default=1)
    val = np.where(np.isclose(x_mod, 0) | np.isclose(x_mod, 4), 0, val)
    return sign * val

def S_sin_partial(x, N):
    s = np.zeros_like(x)
    for n in range(1, N+1):
        arg = n * np.pi * x / 4
        if n % 2 == 0:
            bn = -2/(n * np.pi)
        else:
            k = (n-1)//2
            bn = 2/(n*np.pi) + 4*(-1)**k/(n*n * np.pi*np.pi)
        s += bn * np.sin(arg)
    return s

def plot_fourier_series(title, x_vals, exact_func, partial_func, N_list, ylim, save_name):
    plt.figure(figsize=(10, 6))
    y_exact = exact_func(x_vals)   # теперь exact_func работает с массивом
    plt.plot(x_vals, y_exact, 'k-', linewidth=2, label='Сумма ряда (точное продолжение)')
    
    colors = ['r', 'g', 'b']
    for N, col in zip(N_list, colors):
        y_partial = partial_func(x_vals, N)
        plt.plot(x_vals, y_partial, col+'--', linewidth=1.5, label=f'Частичная сумма N={N}')
    
    if 'общего' not in title:
        x_orig = np.linspace(0, 4, 500)
        y_orig = [f_orig(xi) for xi in x_orig]
        plt.plot(x_orig, y_orig, 'm:', linewidth=2, label='Исходная функция на [0,4]')
    
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlabel('x')
    plt.ylabel('S(x)')
    plt.title(title)
    plt.legend(loc='best')
    plt.xlim(-8, 8)
    plt.ylim(ylim)
    plt.tight_layout()
    plt.savefig(save_name, dpi=150)
    plt.show()

x = np.linspace(-8, 8, 2000)
N_list = [3, 10, 30]

plot_fourier_series('Общий тригонометрический ряд Фурье (период T=4)',
                    x, S_general_exact, S_general_partial, N_list,
                    ylim=(-0.2, 1.2), save_name='general_fourier.png')

plot_fourier_series('Ряд Фурье по косинусам (чётное продолжение, период 8)',
                    x, S_cos_exact, S_cos_partial, N_list,
                    ylim=(-0.2, 1.2), save_name='cosine_fourier.png')

plot_fourier_series('Ряд Фурье по синусам (нечётное продолжение, период 8)',
                    x, S_sin_exact, S_sin_partial, N_list,
                    ylim=(-1.2, 1.2), save_name='sine_fourier.png')

plt.figure(figsize=(10, 4))
x_zoom = np.linspace(-1, 1, 1000)
y_exact_zoom = S_general_exact(x_zoom)
y_partial_zoom = S_general_partial(x_zoom, 30)
plt.plot(x_zoom, y_exact_zoom, 'k-', linewidth=2, label='Точная сумма')
plt.plot(x_zoom, y_partial_zoom, 'r--', linewidth=1.5, label='N=30')
plt.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
plt.title('Явление Гиббса: выбросы вблизи разрыва (общий ряд)')
plt.xlabel('x'); plt.ylabel('S(x)')
plt.legend(); plt.grid(True)
plt.savefig('gibbs_phenomenon.png', dpi=150)
plt.show()

print("Все графики сохранены.")
