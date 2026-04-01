#!/usr/bin/env python3
"""
Полный расчётный пайплайн: от решателя Рейнольдса до оценки надёжности.

Этап 1: Коэффициенты жёсткости/демпфирования K, C (8 штук)
Этап 2: Интегрирование орбиты ротора
Этап 3: Нагрузка на опору P(t)
Этап 4: Ресурс и надёжность по Лундбергу–Палмгрену

Сравнение: гладкий подшипник vs эллипсоидальная текстура.

Зависимости:
    pip install -e <путь_к_GPU_reynolds>   # https://github.com/Daos711/GPU_reynolds
    pip install numpy scipy matplotlib
"""

import os
import warnings
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from reynolds_solver import solve_reynolds
from reynolds_solver.utils import create_H_with_ellipsoidal_depressions

# Совместимость numpy 1.x / 2.x
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

# ──────────────────────────────────────────────────────────────────────
# Конфигурация (все физические параметры)
# ──────────────────────────────────────────────────────────────────────

# --- Геометрия ---
R = 0.035          # Радиус подшипника, м
c = 0.00005        # Радиальный зазор, м
L = 0.056          # Длина подшипника, м

# --- Режим работы ---
n_rpm = 2980                              # Скорость вращения, об/мин
omega_shaft = 2 * np.pi * n_rpm / 60      # Угловая скорость, рад/с
U = omega_shaft * (R - c)                 # Линейная скорость на валу, м/с
eta = 0.01105                             # Динамическая вязкость, Па·с

# --- Текстура ---
h_p = 0.00001      # Глубина углубления, м
H_p = h_p / c      # Безразмерная глубина
a_tex = 0.00241    # Полуось по Z, м
b_tex = 0.002214   # Полуось по φ, м
A_tex = 2 * a_tex / L   # Безразмерная полуось по Z
B_tex = b_tex / R        # Безразмерная полуось по φ

# --- Расположение углублений ---
N_phi_tex = 8       # Количество углублений по φ  (НЕ путать с num_phi_points!)
N_Z_tex = 11        # Количество углублений по Z   (НЕ путать с num_Z_points!)
phi_start_deg = 90
phi_end_deg = 270

# --- Масштабы ---
pressure_scale = (6 * eta * U * R) / (c ** 2)
load_scale = pressure_scale * (R * L) / 2

psi = c / R
K_scale = eta * omega_shaft * L / psi ** 3    # Н/м
C_scale = eta * L / psi ** 3                  # Н·с/м

# --- Сетка ---
num_phi_points = 500
num_Z_points = 500

phi_1D = np.linspace(0, 2 * np.pi, num_phi_points, endpoint=False)
Z = np.linspace(-1, 1, num_Z_points)
d_phi = 2 * np.pi / num_phi_points
d_Z = Z[1] - Z[0]

Phi_mesh, Z_mesh = np.meshgrid(phi_1D, Z)
cos_phi_mesh = np.cos(Phi_mesh)
sin_phi_mesh = np.sin(Phi_mesh)

# --- Параметр SOR (НЕ путать с omega_shaft!) ---
omega_sor = 1.5

# --- Координаты центров углублений ---
phi_c_arr = np.linspace(np.radians(phi_start_deg), np.radians(phi_end_deg), N_phi_tex)
Z_c_arr = np.linspace(-1 + A_tex, 1 - A_tex, N_Z_tex)
phi_c_grid, Z_c_grid = np.meshgrid(phi_c_arr, Z_c_arr)
phi_c_flat = phi_c_grid.ravel()
Z_c_flat = Z_c_grid.ravel()

# --- Параметры Лундберга–Палмгрена ---
C_bearing = 50000.0   # Динамическая грузоподъёмность, Н
p_lp = 3              # 3 — шариковые
beta_weibull = 1.5    # Параметр формы Вейбулла

# --- Параметры орбиты ---
m_rotor = 1.0         # масса ротора, кг
F0 = 1.0e4            # амплитуда внешней силы, Н
Omega = omega_shaft    # возмущающая частота

# --- Подшипник качения (ПК) — из каталога ---
K_pk_xx = 1e8       # Жёсткость ПК по x, Н/м
K_pk_yy = 1e8       # Жёсткость ПК по y, Н/м
K_pk_xy = 0.0       # Перекрёстная жёсткость ПК
K_pk_yx = 0.0
C_pk_xx = 500.0     # Демпфирование ПК по x, Н·с/м
C_pk_yy = 500.0     # Демпфирование ПК по y, Н·с/м
C_pk_xy = 0.0
C_pk_yx = 0.0

# --- Параметры возмущений ---
de = 5e-4             # для жёсткости
dv_diag = 5e-4        # для Cxx, Cyy (диагональные — стабильны)
dv_cross = 1e-2       # для Cxy, Cyx (перекрёстные — нужен больший шаг)

# --- Диапазон эксцентриситетов ---
epsilon_values = np.linspace(0.2, 0.8, 10)
epsilon0_operating = 0.6   # рабочая точка

# --- Директория для графиков ---
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
# Solver adapter
# ──────────────────────────────────────────────────────────────────────

def solver_adapter(ex, ey, exdot_star=0.0, eydot_star=0.0,
                   with_depressions=False):
    """
    Формирует зазор H(ex, ey), решает Рейнольдса, возвращает силы.

    КРИТИЧЕСКИ ВАЖНО: перестановка xprime <-> eydot!
      В решателе: F_dyn = xprime * sin(φ) + yprime * cos(φ)
      Зазор:      H = 1 + ex * cos(φ) + ey * sin(φ)
      Поэтому:
        ėx (при cos) -> yprime в решателе
        ėy (при sin) -> xprime в решателе

    Parameters
    ----------
    ex, ey : float
        Компоненты эксцентриситета.
    exdot_star, eydot_star : float
        Безразмерные скорости (для коэффициентов демпфирования).
    with_depressions : bool
        Добавлять текстуру.

    Returns
    -------
    Fx_dim, Fy_dim : float
        Размерные силы [Н].
    diag : dict
        Диагностика (n_iter, delta, P_max).
    """
    # Базовый зазор
    H0 = 1.0 + ex * cos_phi_mesh + ey * sin_phi_mesh

    # Текстура
    if with_depressions:
        H = create_H_with_ellipsoidal_depressions(
            H0, H_p, Phi_mesh, Z_mesh, phi_c_flat, Z_c_flat, A_tex, B_tex
        )
    else:
        H = H0.copy()

    # ПЕРЕСТАНОВКА: exdot -> yprime, eydot -> xprime
    P, delta, n_iter = solve_reynolds(
        H, d_phi, d_Z, R, L,
        omega=omega_sor,
        xprime=eydot_star,
        yprime=exdot_star
    )

    # Интегрирование давления -> силы (безразмерные)
    Fx_nd = _trapezoid(_trapezoid(P * cos_phi_mesh, phi_1D, axis=1), Z)
    Fy_nd = _trapezoid(_trapezoid(P * sin_phi_mesh, phi_1D, axis=1), Z)

    # Размерные силы
    Fx_dim = Fx_nd * load_scale
    Fy_dim = Fy_nd * load_scale

    diag = {
        "n_iter": n_iter,
        "delta": delta,
        "P_max": float(np.max(P)),
    }

    return Fx_dim, Fy_dim, diag


# ──────────────────────────────────────────────────────────────────────
# Compute K, C (8 коэффициентов)
# ──────────────────────────────────────────────────────────────────────

def compute_KC(epsilon, with_depressions=False, de_val=None,
               dv_diag_val=None, dv_cross_val=None):
    """
    Вычислить 8 коэффициентов жёсткости и демпфирования.

    Конвенция знаков (восстанавливающие):
        K_ij = -dF_i / dq_j
        C_ij = -dF_i / dq̇_j

    Используются раздельные шаги dv для диагональных (Cxx, Cyy)
    и перекрёстных (Cxy, Cyx) коэффициентов демпфирования.
    """
    if de_val is None:
        de_val = de
    if dv_diag_val is None:
        dv_diag_val = dv_diag
    if dv_cross_val is None:
        dv_cross_val = dv_cross

    ex0 = epsilon
    ey0 = 0.0

    diags = []

    # ─── Жёсткость: возмущение ex ± de ───
    Fx_p, Fy_p, d1 = solver_adapter(ex0 + de_val, ey0, with_depressions=with_depressions)
    Fx_m, Fy_m, d2 = solver_adapter(ex0 - de_val, ey0, with_depressions=with_depressions)
    diags.extend([d1, d2])

    Kxx_star = -(Fx_p - Fx_m) / (2 * de_val)
    Kyx_star = -(Fy_p - Fy_m) / (2 * de_val)

    # ─── Жёсткость: возмущение ey ± de ───
    Fx_p, Fy_p, d3 = solver_adapter(ex0, ey0 + de_val, with_depressions=with_depressions)
    Fx_m, Fy_m, d4 = solver_adapter(ex0, ey0 - de_val, with_depressions=with_depressions)
    diags.extend([d3, d4])

    Kxy_star = -(Fx_p - Fx_m) / (2 * de_val)
    Kyy_star = -(Fy_p - Fy_m) / (2 * de_val)

    # ─── Демпфирование ėx: Cxx (диаг, dv_diag), Cyx (перекр, dv_cross) ───
    # Cxx: прямой отклик Fx на ėx
    Fx_p, Fy_p, d5 = solver_adapter(ex0, ey0, exdot_star=+dv_diag_val, with_depressions=with_depressions)
    Fx_m, Fy_m, d6 = solver_adapter(ex0, ey0, exdot_star=-dv_diag_val, with_depressions=with_depressions)
    diags.extend([d5, d6])
    Cxx_star = -(Fx_p - Fx_m) / (2 * dv_diag_val)

    # Cyx: перекрёстный отклик Fy на ėx
    Fx_p2, Fy_p2, d5c = solver_adapter(ex0, ey0, exdot_star=+dv_cross_val, with_depressions=with_depressions)
    Fx_m2, Fy_m2, d6c = solver_adapter(ex0, ey0, exdot_star=-dv_cross_val, with_depressions=with_depressions)
    diags.extend([d5c, d6c])
    Cyx_star = -(Fy_p2 - Fy_m2) / (2 * dv_cross_val)
    print(f"    Cyx debug: Fy(+dv)={Fy_p2:.6e}, Fy(-dv)={Fy_m2:.6e}, "
          f"diff={Fy_p2 - Fy_m2:.6e}, delta+={d5c['delta']:.2e}, delta-={d6c['delta']:.2e}")

    # ─── Демпфирование ėy: Cyy (диаг, dv_diag), Cxy (перекр, dv_cross) ───
    # Cyy: прямой отклик Fy на ėy
    Fx_p, Fy_p, d7 = solver_adapter(ex0, ey0, eydot_star=+dv_diag_val, with_depressions=with_depressions)
    Fx_m, Fy_m, d8 = solver_adapter(ex0, ey0, eydot_star=-dv_diag_val, with_depressions=with_depressions)
    diags.extend([d7, d8])
    Cyy_star = -(Fy_p - Fy_m) / (2 * dv_diag_val)

    # Cxy: перекрёстный отклик Fx на ėy
    Fx_p2, Fy_p2, d7c = solver_adapter(ex0, ey0, eydot_star=+dv_cross_val, with_depressions=with_depressions)
    Fx_m2, Fy_m2, d8c = solver_adapter(ex0, ey0, eydot_star=-dv_cross_val, with_depressions=with_depressions)
    diags.extend([d7c, d8c])
    Cxy_star = -(Fx_p2 - Fx_m2) / (2 * dv_cross_val)
    print(f"    Cxy debug: Fx(+dv)={Fx_p2:.6e}, Fx(-dv)={Fx_m2:.6e}, "
          f"diff={Fx_p2 - Fx_m2:.6e}, delta+={d7c['delta']:.2e}, delta-={d8c['delta']:.2e}")

    # ─── Перевод в размерные ───
    # K: сила / de -> сила / (de * c) = K_dim [Н/м]
    Kxx_dim = Kxx_star / c
    Kxy_dim = Kxy_star / c
    Kyx_dim = Kyx_star / c
    Kyy_dim = Kyy_star / c

    # C: сила / dv -> сила / (dv * c * omega_shaft) = C_dim [Н·с/м]
    Cxx_dim = Cxx_star / (c * omega_shaft)
    Cxy_dim = Cxy_star / (c * omega_shaft)
    Cyx_dim = Cyx_star / (c * omega_shaft)
    Cyy_dim = Cyy_star / (c * omega_shaft)

    # ─── Безразмерные (делённые на масштаб) ───
    Kxx_nd = Kxx_dim / K_scale
    Kxy_nd = Kxy_dim / K_scale
    Kyx_nd = Kyx_dim / K_scale
    Kyy_nd = Kyy_dim / K_scale

    Cxx_nd = Cxx_dim / C_scale
    Cxy_nd = Cxy_dim / C_scale
    Cyx_nd = Cyx_dim / C_scale
    Cyy_nd = Cyy_dim / C_scale

    coef_nd = {
        "Kxx": Kxx_nd, "Kxy": Kxy_nd, "Kyx": Kyx_nd, "Kyy": Kyy_nd,
        "Cxx": Cxx_nd, "Cxy": Cxy_nd, "Cyx": Cyx_nd, "Cyy": Cyy_nd,
    }
    coef_dim = {
        "Kxx": Kxx_dim, "Kxy": Kxy_dim, "Kyx": Kyx_dim, "Kyy": Kyy_dim,
        "Cxx": Cxx_dim, "Cxy": Cxy_dim, "Cyx": Cyx_dim, "Cyy": Cyy_dim,
    }

    return coef_nd, coef_dim, diags


# ──────────────────────────────────────────────────────────────────────
# Sanity checks
# ──────────────────────────────────────────────────────────────────────

def sanity_check_KC(coef_nd, label=""):
    """Проверки физичности K, C."""
    ok = True
    for name in ["Kxx", "Kyy"]:
        if coef_nd[name] <= 0:
            warnings.warn(f"[{label}] {name} = {coef_nd[name]:.4e} <= 0  -- ПРОВЕРИТЬ КОНВЕНЦИИ!")
            ok = False
    for name in ["Cxx", "Cyy"]:
        if coef_nd[name] <= 0:
            warnings.warn(f"[{label}] {name} = {coef_nd[name]:.4e} <= 0  -- ПРОВЕРИТЬ КОНВЕНЦИИ!")
            ok = False

    # Проверка собственных значений симметричной части C
    C_mat = np.array([[coef_nd["Cxx"], coef_nd["Cxy"]],
                       [coef_nd["Cyx"], coef_nd["Cyy"]]])
    C_sym = 0.5 * (C_mat + C_mat.T)
    eigs = np.linalg.eigvalsh(C_sym)
    if np.any(eigs <= 0):
        warnings.warn(f"[{label}] Симм. часть C имеет неположит. собств. знач.: {eigs}")
        ok = False

    return ok


# ──────────────────────────────────────────────────────────────────────
# Проверка стабильности δ
# ──────────────────────────────────────────────────────────────────────

def check_delta_stability(epsilon, with_depressions=False):
    """Проверить, что коэффициенты устойчивы к изменению δ в 2 раза."""
    coef1, _, _ = compute_KC(epsilon, with_depressions=with_depressions,
                             de_val=de, dv_diag_val=dv_diag, dv_cross_val=dv_cross)
    coef2, _, _ = compute_KC(epsilon, with_depressions=with_depressions,
                             de_val=de / 2, dv_diag_val=dv_diag / 2, dv_cross_val=dv_cross / 2)

    print(f"\n{'='*60}")
    print(f"Проверка стабильности δ (ε={epsilon}, text={'да' if with_depressions else 'нет'})")
    print(f"{'Коэфф.':<8} {'δ':>12} {'δ/2':>12} {'Δ%':>8}")
    print(f"{'-'*44}")

    max_diff = 0.0
    for key in coef1:
        v1 = coef1[key]
        v2 = coef2[key]
        if abs(v1) > 1e-12:
            diff_pct = abs(v2 - v1) / abs(v1) * 100
        else:
            diff_pct = 0.0
        max_diff = max(max_diff, diff_pct)
        print(f"{key:<8} {v1:>12.4f} {v2:>12.4f} {diff_pct:>7.1f}%")

    status = "OK" if max_diff < 5 else "НУЖНА КОРРЕКЦИЯ δ"
    print(f"\nМакс. отклонение: {max_diff:.1f}% — {status}")
    return max_diff < 5


# ──────────────────────────────────────────────────────────────────────
# Параметры устойчивости
# ──────────────────────────────────────────────────────────────────────

def compute_stability_params(Kxx, Kxy, Kyx, Kyy, Cxx, Cxy, Cyx, Cyy):
    """Эквивалентная жёсткость, порог устойчивости, частота прецессии."""
    denom = Cxx + Cyy
    K_eq = (Kxx * Cyy + Kyy * Cxx - Kxy * Cyx - Kyx * Cxy) / denom
    gamma_st2 = ((K_eq - Kxx) * (K_eq - Kyy) - Kxy * Kyx) / (Cxx * Cyy - Cxy * Cyx)
    omega_st = K_eq / gamma_st2 if abs(gamma_st2) > 1e-30 else np.inf
    return K_eq, gamma_st2, omega_st


# ──────────────────────────────────────────────────────────────────────
# Этап 2: Интегрирование орбиты ротора
# ──────────────────────────────────────────────────────────────────────

def integrate_orbit(coef_nd_ps, n_periods=40, pts_per_period=200):
    """
    Двухопорная модель ротора:
        m·q̈ = F_ext − (K_ps + K_pk)·q − (C_ps + C_pk)·q̇

    coef_nd_ps — безразмерные K,C подшипника скольжения (зависят от текстуры).
    K_pk, C_pk — размерные параметры подшипника качения (из конфига).
    """
    m_nd = m_rotor * Omega ** 2 / K_scale
    F_nd = F0 / (K_scale * c)

    # Обезразмеривание ПК
    Kpk_xx_nd = K_pk_xx / K_scale
    Kpk_xy_nd = K_pk_xy / K_scale
    Kpk_yx_nd = K_pk_yx / K_scale
    Kpk_yy_nd = K_pk_yy / K_scale
    Cpk_xx_nd = C_pk_xx / C_scale
    Cpk_xy_nd = C_pk_xy / C_scale
    Cpk_yx_nd = C_pk_yx / C_scale
    Cpk_yy_nd = C_pk_yy / C_scale

    # Суммарные коэффициенты (ПС + ПК)
    Kxx_t = coef_nd_ps["Kxx"] + Kpk_xx_nd
    Kxy_t = coef_nd_ps["Kxy"] + Kpk_xy_nd
    Kyx_t = coef_nd_ps["Kyx"] + Kpk_yx_nd
    Kyy_t = coef_nd_ps["Kyy"] + Kpk_yy_nd
    Cxx_t = coef_nd_ps["Cxx"] + Cpk_xx_nd
    Cxy_t = coef_nd_ps["Cxy"] + Cpk_xy_nd
    Cyx_t = coef_nd_ps["Cyx"] + Cpk_yx_nd
    Cyy_t = coef_nd_ps["Cyy"] + Cpk_yy_nd

    def rotor_ode(t_star, state):
        x, y, vx, vy = state
        Fx_ext = F_nd * np.cos(t_star)
        Fy_ext = 0.0
        ax = (Fx_ext - Kxx_t * x - Kxy_t * y - Cxx_t * vx - Cxy_t * vy) / m_nd
        ay = (Fy_ext - Kyx_t * x - Kyy_t * y - Cyx_t * vx - Cyy_t * vy) / m_nd
        return [vx, vy, ax, ay]

    t_max = n_periods * 2 * np.pi
    t_eval = np.linspace(0, t_max, n_periods * pts_per_period)

    sol = solve_ivp(
        rotor_ode, (0, t_max), [0.0, 0.0, 0.0, 0.0],
        method="BDF", rtol=1e-8, atol=1e-10, t_eval=t_eval
    )

    return sol


# ──────────────────────────────────────────────────────────────────────
# Собственные значения системы (проверка устойчивости)
# ──────────────────────────────────────────────────────────────────────

def compute_eigenvalues(coef_nd_ps):
    """
    Собственные значения двухопорной линейной системы.
    Суммарные K,C = K_ps + K_pk, C_ps + C_pk.
    """
    m_nd = m_rotor * Omega ** 2 / K_scale

    Kxx_t = coef_nd_ps["Kxx"] + K_pk_xx / K_scale
    Kxy_t = coef_nd_ps["Kxy"] + K_pk_xy / K_scale
    Kyx_t = coef_nd_ps["Kyx"] + K_pk_yx / K_scale
    Kyy_t = coef_nd_ps["Kyy"] + K_pk_yy / K_scale
    Cxx_t = coef_nd_ps["Cxx"] + C_pk_xx / C_scale
    Cxy_t = coef_nd_ps["Cxy"] + C_pk_xy / C_scale
    Cyx_t = coef_nd_ps["Cyx"] + C_pk_yx / C_scale
    Cyy_t = coef_nd_ps["Cyy"] + C_pk_yy / C_scale

    A_sys = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [-Kxx_t / m_nd, -Kxy_t / m_nd, -Cxx_t / m_nd, -Cxy_t / m_nd],
        [-Kyx_t / m_nd, -Kyy_t / m_nd, -Cyx_t / m_nd, -Cyy_t / m_nd],
    ])

    eigs = np.linalg.eigvals(A_sys)
    return eigs


# ──────────────────────────────────────────────────────────────────────
# Этап 3: Нагрузка на опору P(t)
# ──────────────────────────────────────────────────────────────────────

def compute_bearing_load_pk(sol):
    """
    Нагрузка на подшипник КАЧЕНИЯ (ПК):
        F_pk(t) = K_pk · q(t) + C_pk · q̇(t)

    Используются фиксированные K_pk, C_pk из конфига (не зависят от текстуры).
    Текстура влияет на q(t) через изменение K_ps, C_ps в уравнении движения.
    """
    x_dim = sol.y[0] * c
    y_dim = sol.y[1] * c
    xdot_dim = sol.y[2] * c * Omega
    ydot_dim = sol.y[3] * c * Omega

    Fx_pk = (K_pk_xx * x_dim + K_pk_xy * y_dim
             + C_pk_xx * xdot_dim + C_pk_xy * ydot_dim)
    Fy_pk = (K_pk_yx * x_dim + K_pk_yy * y_dim
             + C_pk_yx * xdot_dim + C_pk_yy * ydot_dim)

    P_pk = np.sqrt(Fx_pk ** 2 + Fy_pk ** 2)
    return P_pk, Fx_pk, Fy_pk


# ──────────────────────────────────────────────────────────────────────
# Этап 4: Лундберг–Палмгрен
# ──────────────────────────────────────────────────────────────────────

def compute_Peq(P_bearing, t_nd, transient_fraction=0.2):
    """Эквивалентная нагрузка P_eq (отбросив переходный процесс)."""
    t_cut = transient_fraction * t_nd[-1]
    mask = t_nd > t_cut
    P_steady = P_bearing[mask]
    P_eq = (np.mean(P_steady ** p_lp)) ** (1.0 / p_lp)
    return P_eq


def compute_L10(P_eq):
    """Ресурс L10 в оборотах и часах."""
    L10_rev = (C_bearing / P_eq) ** p_lp * 1e6
    L10h = L10_rev / (60 * n_rpm)
    return L10_rev, L10h


def weibull_reliability(L10h, t_hours=None):
    """Кривая надёжности R(t) по Вейбуллу."""
    eta_wb = L10h / (-np.log(0.9)) ** (1.0 / beta_weibull)
    if t_hours is None:
        t_hours = np.linspace(0, 5 * L10h, 1000)
    R = np.exp(-(t_hours / eta_wb) ** beta_weibull)
    return t_hours, R


# ──────────────────────────────────────────────────────────────────────
# Построение графиков
# ──────────────────────────────────────────────────────────────────────

def plot_KC_vs_epsilon(eps_arr, KC_smooth, KC_textured, save=True):
    """8 графиков K_ij(ε) и C_ij(ε) — гладкий vs текстура."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    names_K = ["Kxx", "Kxy", "Kyx", "Kyy"]
    names_C = ["Cxx", "Cxy", "Cyx", "Cyy"]

    for idx, name in enumerate(names_K):
        ax = axes[0, idx]
        ax.plot(eps_arr, [kc[name] for kc in KC_smooth], "b-o", label="Гладкий", markersize=4)
        ax.plot(eps_arr, [kc[name] for kc in KC_textured], "r-s", label="Текстура", markersize=4)
        ax.set_xlabel("ε")
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for idx, name in enumerate(names_C):
        ax = axes[1, idx]
        ax.plot(eps_arr, [kc[name] for kc in KC_smooth], "b-o", label="Гладкий", markersize=4)
        ax.plot(eps_arr, [kc[name] for kc in KC_textured], "r-s", label="Текстура", markersize=4)
        ax.set_xlabel("ε")
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Коэффициенты жёсткости и демпфирования vs ε", fontsize=14)
    fig.tight_layout()
    if save:
        fig.savefig(f"{PLOT_DIR}/01_KC_vs_epsilon.png", dpi=150)
    plt.close(fig)


def plot_stability_vs_epsilon(eps_arr, stab_smooth, stab_textured, save=True):
    """K_eq, γ²_st, ω_st vs ε."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    labels = ["K_eq", "γ²_st", "ω_st"]

    for idx, lbl in enumerate(labels):
        ax = axes[idx]
        ax.plot(eps_arr, [s[idx] for s in stab_smooth], "b-o", label="Гладкий", markersize=4)
        ax.plot(eps_arr, [s[idx] for s in stab_textured], "r-s", label="Текстура", markersize=4)
        ax.set_xlabel("ε")
        ax.set_ylabel(lbl)
        ax.set_title(lbl)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Параметры устойчивости vs ε", fontsize=14)
    fig.tight_layout()
    if save:
        fig.savefig(f"{PLOT_DIR}/02_stability_vs_epsilon.png", dpi=150)
    plt.close(fig)


def plot_orbit(sol_smooth, sol_textured, save=True):
    """Орбита ротора: полная и последние 2 периода."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Полная орбита
    ax = axes[0]
    ax.plot(sol_smooth.y[0], sol_smooth.y[1], "b-", alpha=0.5, label="Гладкий")
    ax.plot(sol_textured.y[0], sol_textured.y[1], "r-", alpha=0.5, label="Текстура")
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.set_title("Полная орбита")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Последние 2 периода
    ax = axes[1]
    T = 2 * np.pi
    t_cut = sol_smooth.t[-1] - 2 * T
    mask_s = sol_smooth.t >= t_cut
    mask_t = sol_textured.t >= t_cut
    ax.plot(sol_smooth.y[0][mask_s], sol_smooth.y[1][mask_s], "b-", label="Гладкий")
    ax.plot(sol_textured.y[0][mask_t], sol_textured.y[1][mask_t], "r-", label="Текстура")
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.set_title("Последние 2 периода (установившийся режим)")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save:
        fig.savefig(f"{PLOT_DIR}/03_orbit.png", dpi=150)
    plt.close(fig)


def plot_Pt(t_nd_s, Pb_s, t_nd_t, Pb_t, save=True):
    """P(t) — нагрузка на опору."""
    fig, ax = plt.subplots(figsize=(12, 5))
    # Показываем последние ~5 периодов
    T = 2 * np.pi
    t_cut = max(t_nd_s[-1], t_nd_t[-1]) - 5 * T
    mask_s = t_nd_s >= t_cut
    mask_t = t_nd_t >= t_cut
    ax.plot(t_nd_s[mask_s] / T, Pb_s[mask_s], "b-", label="Гладкий")
    ax.plot(t_nd_t[mask_t] / T, Pb_t[mask_t], "r-", label="Текстура")
    ax.set_xlabel("t / T")
    ax.set_ylabel("P(t), Н")
    ax.set_title("Нагрузка на опору (последние 5 периодов)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save:
        fig.savefig(f"{PLOT_DIR}/04_Pt.png", dpi=150)
    plt.close(fig)


def plot_Peq_vs_epsilon(eps_arr, Peq_s, Peq_t, save=True):
    """P_eq vs ε."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eps_arr, Peq_s, "b-o", label="Гладкий", markersize=5)
    ax.plot(eps_arr, Peq_t, "r-s", label="Текстура", markersize=5)
    ax.set_xlabel("ε")
    ax.set_ylabel("P_eq, Н")
    ax.set_title("Эквивалентная нагрузка vs ε")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save:
        fig.savefig(f"{PLOT_DIR}/05_Peq_vs_epsilon.png", dpi=150)
    plt.close(fig)


def plot_L10h_vs_epsilon(eps_arr, L10h_s, L10h_t, save=True):
    """L10h vs ε."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(eps_arr, L10h_s, "b-o", label="Гладкий", markersize=5)
    ax.semilogy(eps_arr, L10h_t, "r-s", label="Текстура", markersize=5)
    ax.set_xlabel("ε")
    ax.set_ylabel("L₁₀, часов")
    ax.set_title("Ресурс L₁₀ vs ε")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    if save:
        fig.savefig(f"{PLOT_DIR}/06_L10h_vs_epsilon.png", dpi=150)
    plt.close(fig)


def plot_weibull(L10h_smooth, L10h_textured, save=True):
    """Кривые Вейбулла R(t) при ε₀."""
    t_max_h = 5 * max(L10h_smooth, L10h_textured)
    t_hours = np.linspace(0, t_max_h, 1000)
    _, R_s = weibull_reliability(L10h_smooth, t_hours)
    _, R_t = weibull_reliability(L10h_textured, t_hours)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_hours, R_s, "b-", label="Гладкий")
    ax.plot(t_hours, R_t, "r-", label="Текстура")
    ax.axhline(0.9, color="gray", ls="--", alpha=0.5, label="R = 0.9 (L₁₀)")
    ax.set_xlabel("t, часов")
    ax.set_ylabel("R(t)")
    ax.set_title(f"Надёжность по Вейбуллу (ε₀ = {epsilon0_operating})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save:
        fig.savefig(f"{PLOT_DIR}/07_weibull.png", dpi=150)
    plt.close(fig)


def plot_ratio_vs_epsilon(eps_arr, ratio_arr, save=True):
    """Отношение ресурсов L10_textured / L10_smooth vs ε."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eps_arr, ratio_arr, "g-o", markersize=5)
    ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("ε")
    ax.set_ylabel("L₁₀(текст.) / L₁₀(гладк.)")
    ax.set_title("Отношение ресурсов: текстура / гладкий")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save:
        fig.savefig(f"{PLOT_DIR}/08_ratio_vs_epsilon.png", dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("ПАЙПЛАЙН: от решателя Рейнольдса до оценки надёжности")
    print("=" * 70)

    # ──────────────────────────────────────────────────────────────────
    # Этап 1: K, C для диапазона ε
    # ──────────────────────────────────────────────────────────────────
    print("\n>>> ЭТАП 1: Вычисление коэффициентов K, C")

    # Проверка стабильности δ при рабочем ε
    print(f"\n--- Проверка стабильности δ при ε = {epsilon0_operating} ---")
    check_delta_stability(epsilon0_operating, with_depressions=False)
    check_delta_stability(epsilon0_operating, with_depressions=True)

    # Sweep по ε
    KC_smooth_nd = []
    KC_textured_nd = []
    stab_smooth = []
    stab_textured = []

    for i, eps in enumerate(epsilon_values):
        print(f"\n  ε = {eps:.2f} ({i + 1}/{len(epsilon_values)})")

        # Гладкий
        coef_nd_s, _, diags_s = compute_KC(eps, with_depressions=False)
        ok_s = sanity_check_KC(coef_nd_s, label=f"smooth ε={eps:.2f}")
        KC_smooth_nd.append(coef_nd_s)

        sp_s = compute_stability_params(**{k: coef_nd_s[k] for k in coef_nd_s})
        stab_smooth.append(sp_s)

        print(f"    Гладкий: Kxx={coef_nd_s['Kxx']:.3f} Kyy={coef_nd_s['Kyy']:.3f} "
              f"Cxx={coef_nd_s['Cxx']:.3f} Cyy={coef_nd_s['Cyy']:.3f} "
              f"{'OK' if ok_s else 'WARN'}")

        # Текстурированный
        coef_nd_t, _, diags_t = compute_KC(eps, with_depressions=True)
        ok_t = sanity_check_KC(coef_nd_t, label=f"textured ε={eps:.2f}")
        KC_textured_nd.append(coef_nd_t)

        sp_t = compute_stability_params(**{k: coef_nd_t[k] for k in coef_nd_t})
        stab_textured.append(sp_t)

        print(f"    Текстур: Kxx={coef_nd_t['Kxx']:.3f} Kyy={coef_nd_t['Kyy']:.3f} "
              f"Cxx={coef_nd_t['Cxx']:.3f} Cyy={coef_nd_t['Cyy']:.3f} "
              f"{'OK' if ok_t else 'WARN'}")

    # Графики Этапа 1
    plot_KC_vs_epsilon(epsilon_values, KC_smooth_nd, KC_textured_nd)
    plot_stability_vs_epsilon(epsilon_values, stab_smooth, stab_textured)
    print("\nГрафики K/C и устойчивости сохранены.")

    # ──────────────────────────────────────────────────────────────────
    # Этапы 2-4: для рабочей точки ε₀
    # ──────────────────────────────────────────────────────────────────
    idx0 = np.argmin(np.abs(epsilon_values - epsilon0_operating))
    coef_nd_s0 = KC_smooth_nd[idx0]
    coef_nd_t0 = KC_textured_nd[idx0]

    # ─── Верификация масштабов ───
    print(f"\n>>> ДИАГНОСТИКА МАСШТАБОВ (ε₀ = {epsilon0_operating})")
    print(f"  K_scale = {K_scale:.3e} Н/м")
    print(f"  C_scale = {C_scale:.3e} Н·с/м")
    Cxx_dim_check = coef_nd_s0["Cxx"] * C_scale
    print(f"  Cxx_nd(гладк.) = {coef_nd_s0['Cxx']:.4f}")
    print(f"  Cxx_dim = Cxx_nd × C_scale = {Cxx_dim_check:.3e} Н·с/м")
    print(f"  K_pk = {K_pk_xx:.3e} Н/м,  C_pk = {C_pk_xx:.3e} Н·с/м")
    Kpk_nd = K_pk_xx / K_scale
    Cpk_nd = C_pk_xx / C_scale
    print(f"  Kpk_nd = {Kpk_nd:.3e},  Kps_nd = {coef_nd_s0['Kxx']:.3e}")
    print(f"  Cpk_nd = {Cpk_nd:.3e},  Cps_nd = {coef_nd_s0['Cxx']:.3e}")
    print(f"  Kpk/Kps = {Kpk_nd / coef_nd_s0['Kxx']:.3e}")
    print(f"  Cpk/Cps = {Cpk_nd / coef_nd_s0['Cxx']:.3e}")

    # Собственные значения (устойчивость)
    print(f"\n>>> Собственные значения при ε₀ = {epsilon0_operating}")
    eigs_s = compute_eigenvalues(coef_nd_s0)
    eigs_t = compute_eigenvalues(coef_nd_t0)
    print(f"  Гладкий:  {eigs_s}")
    print(f"  Re(λ) < 0: {np.all(np.real(eigs_s) < 0)}")
    print(f"  Текстура: {eigs_t}")
    print(f"  Re(λ) < 0: {np.all(np.real(eigs_t) < 0)}")

    # Этап 2: Орбита (двухопорная модель: ПС + ПК)
    print(f"\n>>> ЭТАП 2: Интегрирование орбиты (ε₀ = {epsilon0_operating})")
    sol_s = integrate_orbit(coef_nd_s0)
    sol_t = integrate_orbit(coef_nd_t0)
    print(f"  Гладкий:  t_max = {sol_s.t[-1]:.1f}, точек = {len(sol_s.t)}")
    print(f"  Текстура: t_max = {sol_t.t[-1]:.1f}, точек = {len(sol_t.t)}")
    plot_orbit(sol_s, sol_t)

    # Этап 3: Нагрузка на подшипник КАЧЕНИЯ (ПК)
    print(f"\n>>> ЭТАП 3: Нагрузка на ПК (ε₀ = {epsilon0_operating})")

    # Диагностика амплитуд орбиты
    for label, sol in [("Гладкий", sol_s), ("Текстура", sol_t)]:
        x_dim = sol.y[0] * c
        y_dim = sol.y[1] * c
        xdot_dim = sol.y[2] * c * Omega
        ydot_dim = sol.y[3] * c * Omega
        Fx_K = K_pk_xx * x_dim
        Fx_C = C_pk_xx * xdot_dim
        print(f"  {label}: max|x|={np.max(np.abs(x_dim)):.2e} м, "
              f"max|y|={np.max(np.abs(y_dim)):.2e} м, "
              f"max|Fpk_K|={np.max(np.abs(Fx_K)):.2e} Н, "
              f"max|Fpk_C|={np.max(np.abs(Fx_C)):.2e} Н")

    Pb_s, _, _ = compute_bearing_load_pk(sol_s)
    Pb_t, _, _ = compute_bearing_load_pk(sol_t)
    plot_Pt(sol_s.t, Pb_s, sol_t.t, Pb_t)

    P_max_s = np.max(Pb_s)
    P_min_s = np.min(Pb_s)
    P_mean_s = np.mean(Pb_s)
    P_max_t = np.max(Pb_t)
    P_min_t = np.min(Pb_t)
    P_mean_t = np.mean(Pb_t)
    Peq_s0 = compute_Peq(Pb_s, sol_s.t)
    Peq_t0 = compute_Peq(Pb_t, sol_t.t)

    P_var_s = (P_max_s - P_min_s) / P_mean_s if P_mean_s > 0 else 0
    P_var_t = (P_max_t - P_min_t) / P_mean_t if P_mean_t > 0 else 0

    print(f"  Гладкий:  P_min={P_min_s:.1f}, P_max={P_max_s:.1f}, P_mean={P_mean_s:.1f}, "
          f"P_eq={Peq_s0:.1f} Н, P_var={P_var_s:.4f}")
    print(f"  Текстура: P_min={P_min_t:.1f}, P_max={P_max_t:.1f}, P_mean={P_mean_t:.1f}, "
          f"P_eq={Peq_t0:.1f} Н, P_var={P_var_t:.4f}")
    if P_var_s < 0.01 and P_var_t < 0.01:
        warnings.warn("P_var < 0.01 — P(t) почти константа, LP-блок может не различить случаи!")

    # Этап 4: L10, R(t) при ε₀
    print(f"\n>>> ЭТАП 4: Ресурс и надёжность (ε₀ = {epsilon0_operating})")
    L10rev_s, L10h_s0 = compute_L10(Peq_s0)
    L10rev_t, L10h_t0 = compute_L10(Peq_t0)
    ratio0 = L10h_t0 / L10h_s0 if L10h_s0 > 0 else np.inf

    print(f"  Гладкий:  L10 = {L10rev_s:.0f} об. = {L10h_s0:.1f} ч")
    print(f"  Текстура: L10 = {L10rev_t:.0f} об. = {L10h_t0:.1f} ч")
    print(f"  Отношение L10(текст.)/L10(гладк.) = {ratio0:.3f}")

    plot_weibull(L10h_s0, L10h_t0)

    # ──────────────────────────────────────────────────────────────────
    # Sweep P_eq, L10h по ε (этапы 3-4 для всех ε)
    # ──────────────────────────────────────────────────────────────────
    print("\n>>> Sweep P_eq, L10h по ε (с интегрированием орбиты для каждого ε)")

    Peq_smooth_arr = []
    Peq_textured_arr = []
    L10h_smooth_arr = []
    L10h_textured_arr = []
    ratio_arr = []

    for i, eps in enumerate(epsilon_values):
        print(f"  ε = {eps:.2f} ({i + 1}/{len(epsilon_values)})")

        coef_nd_s_i = KC_smooth_nd[i]
        coef_nd_t_i = KC_textured_nd[i]

        # Орбита (двухопорная)
        sol_si = integrate_orbit(coef_nd_s_i)
        sol_ti = integrate_orbit(coef_nd_t_i)

        # Нагрузка на ПК
        Pb_si, _, _ = compute_bearing_load_pk(sol_si)
        Pb_ti, _, _ = compute_bearing_load_pk(sol_ti)

        # P_eq
        Peq_si = compute_Peq(Pb_si, sol_si.t)
        Peq_ti = compute_Peq(Pb_ti, sol_ti.t)
        Peq_smooth_arr.append(Peq_si)
        Peq_textured_arr.append(Peq_ti)

        # L10h
        _, L10h_si = compute_L10(Peq_si)
        _, L10h_ti = compute_L10(Peq_ti)
        L10h_smooth_arr.append(L10h_si)
        L10h_textured_arr.append(L10h_ti)

        r = L10h_ti / L10h_si if L10h_si > 0 else np.inf
        ratio_arr.append(r)

        print(f"    P_eq: гл.={Peq_si:.1f} Н, текст.={Peq_ti:.1f} Н | "
              f"L10h: гл.={L10h_si:.1f}, текст.={L10h_ti:.1f}, ratio={r:.3f}")

    # Графики
    plot_Peq_vs_epsilon(epsilon_values, Peq_smooth_arr, Peq_textured_arr)
    plot_L10h_vs_epsilon(epsilon_values, L10h_smooth_arr, L10h_textured_arr)
    plot_ratio_vs_epsilon(epsilon_values, ratio_arr)

    # ──────────────────────────────────────────────────────────────────
    # Параметрический sweep по K_pk (sensitivity study)
    # ──────────────────────────────────────────────────────────────────
    print("\n>>> SENSITIVITY STUDY: влияние K_pk на результат (ε₀ = {:.2f})".format(
        epsilon0_operating))
    K_pk_variants = [1e8, 1e9, 1e10, 1e11]
    print(f"{'K_pk':>12} | {'Peq_smooth':>12} | {'Peq_text':>12} | "
          f"{'L10h_smooth':>14} | {'L10h_text':>14} | {'ratio':>8}")
    print("-" * 90)

    _sens_rows = []
    for K_pk_test in K_pk_variants:
        row = {"K_pk": K_pk_test}
        for case, coef_nd_ps in [("smooth", coef_nd_s0), ("textured", coef_nd_t0)]:
            m_nd = m_rotor * Omega ** 2 / K_scale
            F_nd = F0 / (K_scale * c)
            Kxx_t = coef_nd_ps["Kxx"] + K_pk_test / K_scale
            Kxy_t = coef_nd_ps["Kxy"]
            Kyx_t = coef_nd_ps["Kyx"]
            Kyy_t = coef_nd_ps["Kyy"] + K_pk_test / K_scale
            Cxx_t = coef_nd_ps["Cxx"] + C_pk_xx / C_scale
            Cxy_t = coef_nd_ps["Cxy"]
            Cyx_t = coef_nd_ps["Cyx"]
            Cyy_t = coef_nd_ps["Cyy"] + C_pk_yy / C_scale

            def _ode(t_star, state, Kxx=Kxx_t, Kxy=Kxy_t, Kyx=Kyx_t, Kyy=Kyy_t,
                     Cxx=Cxx_t, Cxy=Cxy_t, Cyx=Cyx_t, Cyy=Cyy_t,
                     m=m_nd, F=F_nd):
                x, y, vx, vy = state
                ax = (F * np.cos(t_star) - Kxx*x - Kxy*y - Cxx*vx - Cxy*vy) / m
                ay = (- Kyx*x - Kyy*y - Cyx*vx - Cyy*vy) / m
                return [vx, vy, ax, ay]

            t_max_s = 40 * 2 * np.pi
            t_eval_s = np.linspace(0, t_max_s, 8000)
            sol_tmp = solve_ivp(_ode, (0, t_max_s), [0, 0, 0, 0],
                                method="BDF", rtol=1e-8, atol=1e-10, t_eval=t_eval_s)
            x_d = sol_tmp.y[0] * c
            y_d = sol_tmp.y[1] * c
            xdot_d = sol_tmp.y[2] * c * Omega
            ydot_d = sol_tmp.y[3] * c * Omega
            Fx = K_pk_test * x_d + C_pk_xx * xdot_d
            Fy = K_pk_test * y_d + C_pk_yy * ydot_d
            P_tmp = np.sqrt(Fx**2 + Fy**2)
            Peq = compute_Peq(P_tmp, sol_tmp.t)
            _, L10h = compute_L10(Peq)
            row[f"Peq_{case}"] = Peq
            row[f"L10h_{case}"] = L10h
        row["ratio"] = row["L10h_textured"] / row["L10h_smooth"] if row["L10h_smooth"] > 0 else np.inf
        _sens_rows.append(row)
        print(f"{K_pk_test:>12.0e} | {row['Peq_smooth']:>12.2f} | {row['Peq_textured']:>12.2f} | "
              f"{row['L10h_smooth']:>14.2e} | {row['L10h_textured']:>14.2e} | "
              f"{row['ratio']:>8.3f}")

    print("\nЕсли ratio устойчив к вариации K_pk — относительный эффект текстуры надёжен.")

    # ──────────────────────────────────────────────────────────────────
    # Итоговая таблица
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("ИТОГОВАЯ ТАБЛИЦА")
    print("=" * 90)
    print(f"{'ε':>6} | {'P_eq_smooth':>12} | {'P_eq_text':>12} | "
          f"{'L10h_smooth':>12} | {'L10h_text':>12} | {'ratio':>8}")
    print("-" * 90)
    for i, eps in enumerate(epsilon_values):
        print(f"{eps:>6.2f} | {Peq_smooth_arr[i]:>12.1f} | {Peq_textured_arr[i]:>12.1f} | "
              f"{L10h_smooth_arr[i]:>12.1f} | {L10h_textured_arr[i]:>12.1f} | "
              f"{ratio_arr[i]:>8.3f}")
    print("=" * 90)

    # Общий вывод
    significance_tol = 0.01  # 1% порог значимости
    print("\n>>> ВЫВОДЫ:")
    improves = [r > 1 + significance_tol for r in ratio_arr]
    worsens = [r < 1 - significance_tol for r in ratio_arr]
    if all(improves):
        print("  Текстура УВЕЛИЧИВАЕТ ресурс во всём диапазоне ε.")
    elif all(worsens):
        print("  Текстура УМЕНЬШАЕТ ресурс во всём диапазоне ε.")
    else:
        print("  Эффект текстуры зависит от ε:")
        for i, eps in enumerate(epsilon_values):
            r = ratio_arr[i]
            if r < 1 - significance_tol:
                print(f"    ε={eps:.2f}: текстура УХУДШАЕТ (ratio={r:.3f})")
            elif r > 1 + significance_tol:
                print(f"    ε={eps:.2f}: текстура УЛУЧШАЕТ (ratio={r:.3f})")
            else:
                print(f"    ε={eps:.2f}: эффект незначим (ratio={r:.3f})")

    print(f"\nВсе графики сохранены в {PLOT_DIR}/")
    print("Пайплайн завершён.")


if __name__ == "__main__":
    main()
