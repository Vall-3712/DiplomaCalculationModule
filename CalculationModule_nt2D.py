import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

#Автомат выбора шага

P0 = 101325.0
temperature_changeable = 2000.0
temperature_normal = 298.15
k1 =  (6 * (10 ** 13) / (1000 ** 0.5)) * (2.7182 ** (- 164541.24 / 8.3144 / temperature_changeable))
R = 8.3144

X_O2 = 0.13099
X_H2 = 0.26198
X_N2 = 0.49278
X_H2O = 0.11425

M_H2 = 0.002015
M_N2 = 0.028013
M_O2 = 0.031999
M_H2O = 0.018015

OXYGEN_POLYNOMIAL_COEFFICIENTS = (
    2.649719416 * 10,
    1.481243415e4,
    1.131103113e-2,
    -4.044660710e-6,
    7.382809022e-10,
    -4.97174777e-14,
)   

HYDROGEN_POLYNOMIAL_COEFFICIENTS = (
    2.770420424 * 10,
    1.636218387e3,
    2.054652307e-3,
    1.123400140e-6,
    -3.334795123e-10,
    2.6071140007e-14,
)

WATER_POLYNOMIAL_COEFFICIENTS = (
    2.821655197 * 10,
    3.882050326e4,
    1.551421947e-2,
    -1.919153436e-6,
    -7.483133392e-11,
    2.275436566e-14,
)


DH_O2 = 0.0
DH_H2 = 0.0
DH_H2O = -241814.0

S0_O2 = 205.035
S0_H2 = 130.570
S0_H2O = 188.724

def M_sum(x_H2, m_H2, x_O2, m_O2, x_H2O, m_H2O, x_N2, m_N2):
    return x_H2 * m_H2 + x_O2 * m_O2 + x_H2O * m_H2O + x_N2 * m_N2

#1 Рассчитываем мольно-массовые доли вещества

def Moll_Mass(x_H2, x_O2, x_H2O, x_N2):
    gamma_H2 = x_H2 / M_sum(X_H2, M_H2, X_O2, M_O2, X_H2O, M_H2O, X_N2, M_N2)
    gamma_O2 = x_O2 / M_sum(X_H2, M_H2, X_O2, M_O2, X_H2O, M_H2O, X_N2, M_N2)
    gamma_H2O = x_H2O / M_sum(X_H2, M_H2, X_O2, M_O2, X_H2O, M_H2O, X_N2, M_N2)
    gamma_N2 = x_N2 / M_sum(X_H2, M_H2, X_O2, M_O2, X_H2O, M_H2O, X_N2, M_N2)
    return gamma_H2, gamma_O2, gamma_H2O, gamma_N2

#2 Расчёт плотности смеси 

def density(p0, temperature_current, gamma_H2, gamma_O2, gamma_H2O, gamma_N2):
    return p0 / (8.314 * temperature_current * (gamma_O2 + gamma_H2 + gamma_H2O + gamma_N2))

#3 Расчёт Wi правых частей уравнения химической смеси

def W_i(k1, k_opposite, H2_val, O2_val, H2O_val):
    Wi_dH2_dt = (-1.0) * (k1 * (H2_val ** 1.0) * (O2_val ** 0.5) - (k_opposite * H2O_val))
    Wi_dO2_dt = (-0.5) * (k1 * (H2_val ** 1.0) * (O2_val ** 0.5) - (k_opposite * H2O_val))
    Wi_dH2O_dt = (1) * (k1 * (H2_val ** 1.0) * (O2_val ** 0.5)- (k_opposite * H2O_val)) 
    return Wi_dH2_dt, Wi_dO2_dt, Wi_dH2O_dt 

#4 Расчёт термодинамики

def Cp_thermodynamic(T, F):
    T = min(T, 6000)
    return (
        F[0]
        + F[1] / (T ** 2) 
        + F[2] * T 
        + F[3] * (T ** 2) 
        + F[4] * (T ** 3) 
        + F[5] * (T ** 4)
    )
    
def Cp_int_thermodynamic(T, t0, F):
    T = min(T, 6000)
    return (
        F[0] * (T - t0) 
        - (F[1] / T) + (F[1] / t0) 
        + F[2] * (T ** 2 - t0 ** 2) / 2 
        + F[3] * (T ** 3 - t0 ** 3) / 3 
        + F[4] * (T ** 4 - t0 ** 4) / 4 
        + F[5] * (T ** 5 - t0 ** 5) / 5
    )

def Cp_int_T_thermodynamic(T, t0, F): 
    return (
        F[0] * (math.log(T) - math.log(t0)) 
        - 0.5 * F[1] * (1 / (T ** 2) - 1 / (t0 ** 2)) 
        + F[2] * (T - t0) 
        + 0.5 * F[3] * (T ** 2 - t0 ** 2) 
        + (1 / 3) * F[4] * (T ** 3 - t0 ** 3) 
        + (1 / 4) * F[5] * (T ** 4 - t0 ** 4)
    )
    
def H_thermodynamic(delta_H, T, t0, F):
    return delta_H + Cp_int_thermodynamic(T, t0, F)

def S_thermodynamic(s_T, T, t0, F):
    return s_T + Cp_int_T_thermodynamic(T, t0, F)

#5 Расчёт прочих частей уравнения 

def sum_Cp(val_O2, val_H2, val_H2O, h_O2, h_H2, h_H2O, ro):
    return (val_O2 * h_O2 + val_H2 * h_H2 + val_H2O * h_H2O) / ro

# 6 Интегрирование Рунге-Кутт

def solve_runge_kutta(k1, gamma_h2, gamma_o2, gamma_h2o, gamma_n2, h, N, output_file):
    
    # Инициализация массивов
    time = np.zeros(N+1)
    h2 = np.zeros(N+1)
    n2 = np.zeros(N+1)
    o2 = np.zeros(N+1)
    h2o = np.zeros(N+1)
    ro = np.zeros(N+1)
    gam_h2 = np.zeros(N+1)
    gam_o2 = np.zeros(N+1)
    gam_h2o = np.zeros(N+1)
    gam_n2 = np.zeros(N+1)
    temperature = np.zeros(N+1)

    # Начальные условия
    
    time[0] = 0.0
    temperature[0] = temperature_changeable
    sum_gamma = gamma_h2 + gamma_o2 + gamma_h2o + gamma_n2
    ro[0] = P0 / (8.314 * temperature[0] * sum_gamma)
    h2[0] = gamma_h2  * ro[0]
    o2[0] = gamma_o2  * ro[0]
    h2o[0] = gamma_h2o  * ro[0]
    n2[0] = gamma_n2  * ro[0]
    gam_h2[0] = gamma_h2
    gam_o2[0] = gamma_o2
    gam_h2o[0] = gamma_h2o
    gam_n2[0] = gamma_n2
    
    def dpGammaH2_dt(k1, k_opposite, h2_val, o2_val, h2o_val):
        return (-1.0 * (k1 * (h2_val**1.0) * (o2_val**0.5) - (k_opposite * h2o_val))) 
    
    def dpGammaO2_dt(k1, k_opposite, h2_val, o2_val, h2o_val):
        return (-0.5 * (k1 * (h2_val**1.0) * (o2_val**0.5) - (k_opposite * h2o_val))) 
    
    def dpGammaH2O_dt(k1, k_opposite, h2_val, o2_val, h2o_val):
        return (1 * (k1 *(h2_val**1.0) * (o2_val**0.5) - (k_opposite * h2o_val))) 
    
    def calculate_coefficients(k_opposite, h2, o2, h2o):
            return (
                h * dpGammaH2_dt(k1, k_opposite, h2, o2, h2o),
                h * dpGammaO2_dt(k1, k_opposite, h2, o2, h2o),
                h * dpGammaH2O_dt(k1, k_opposite, h2, o2, h2o),
            )
    
    output_file.write(
        f'{"i":<10} {"t":<10} {"temperature":<13} '
        f'{"H_sum":<13} {"H2":<13} {"O2":<13} '
        f'{"H2O":<13} {"H_H2":<13} {"H_O2":<13} '
        f'{"H_H2O":<13} {"Cp_H2":<13} {"Cp_O2":<13} '
        f'{"Cp_H2O":<13}  {"S_H2":<13} {"S_O2":<13} '
        f'{"S_H2OS":<13} {"GammaH2":<10} {"GammaO2":<10} '
        f'{"GammaH2O":<10} {"Wi_H2":<15} {"Wi_O2":<15} '
        f'{"Wi_H2O":<15} {"Si_H2":<13} {"Si_O2":<13} '
        f'{"Si_H2O":<13} {"G_H2":<13} {"G_O2":<13} '
        f'{"G_H2O":<13} {"k1":<15} {"k_opposite":<13} '
        f'{"Cp_sum":<13} {"W4":<13}\n'
    )

   # Основной цикл

    for i in tqdm(range(N)):
        current_time = time[i]
        current_temperature = temperature[i]
        
        current_h2 = h2[i]
        current_o2 = o2[i]
        current_h2o = h2o[i]
        current_n2 = n2[i]
        ro[i] = current_h2 * M_H2 + current_o2 * M_O2 + current_h2o * M_H2O + current_n2 * M_N2
    
        # Применяем правила регулировки шага
 
        if (i > 10000) and (h2o[i-1] != 0):
            if (current_h2o / h2o[i-1] > 1.03):
                h = h * 0.010
        if (i > 10000) and (h2o[i-1] != 0):
            if (current_h2o / h2o[i-1] < 1.001):
                h =  h * 1.0001
        if h > h_begin * 0.4:
                h = h_begin * 0.4
        
        H_H2 = H_thermodynamic(DH_H2, current_temperature, temperature_normal, HYDROGEN_POLYNOMIAL_COEFFICIENTS)
        H_O2 = H_thermodynamic(DH_O2, current_temperature, temperature_normal, OXYGEN_POLYNOMIAL_COEFFICIENTS)
        H_H2O = H_thermodynamic(DH_H2O, current_temperature, temperature_normal, WATER_POLYNOMIAL_COEFFICIENTS)
        
        CP_H2 = Cp_thermodynamic(current_temperature, HYDROGEN_POLYNOMIAL_COEFFICIENTS)
        CP_O2 = Cp_thermodynamic(current_temperature, OXYGEN_POLYNOMIAL_COEFFICIENTS)
        CP_H2O = Cp_thermodynamic(current_temperature, WATER_POLYNOMIAL_COEFFICIENTS)
        CP_N2 = 35.37
                
        current_ro = ro[i]
        cur_h2_gamma = current_h2 / current_ro
        cur_o2_gamma = current_o2 / current_ro
        cur_h2o_gamma = current_h2o / current_ro
        cur_n2_gamma = gamma_n2
        
        gam_h2[i] = cur_h2_gamma
        gam_o2[i] = cur_o2_gamma
        gam_h2o[i] = cur_h2o_gamma
        gam_n2[i] = gamma_n2
        
        H_sum = cur_h2_gamma * H_H2 + cur_o2_gamma * H_O2 + cur_h2o_gamma * H_H2O 
        Cp_sum = cur_h2_gamma * CP_H2 + cur_o2_gamma * CP_O2 + cur_h2o_gamma * CP_H2O + cur_n2_gamma * CP_N2
        
        S_H2 = S_thermodynamic(S0_H2, current_temperature,  temperature_normal, HYDROGEN_POLYNOMIAL_COEFFICIENTS)
        S_O2 = S_thermodynamic(S0_O2, current_temperature, temperature_normal, OXYGEN_POLYNOMIAL_COEFFICIENTS)
        S_H2O = S_thermodynamic(S0_H2O, current_temperature, temperature_normal, WATER_POLYNOMIAL_COEFFICIENTS)
        
        Si_H2 = (-R) * math.log((current_ro * R * current_temperature * cur_h2_gamma) / P0) + S_H2
        Si_O2 = (-R) * math.log((current_ro * R * current_temperature * cur_o2_gamma) / P0) + S_O2
        Si_H2O = (-R) * math.log((current_ro * R * current_temperature * (cur_h2o_gamma + 0.000000001)) / P0) + S_H2O
        
        G_H2 = H_H2 - (current_temperature * S_H2) 
        G_O2 = H_O2 - (current_temperature * S_O2) 
        G_H2O = H_H2O - (current_temperature * S_H2O) 
        
        k1 =  (6 * (10 ** 13) / (1000 ** 0.5)) * (math.exp(- 164541.24 / 8.3144 / current_temperature))
        kk = -0.5 * (math.log((R * current_temperature) / P0)) + ((G_H2O - G_H2 - G_O2 * 0.5) / (R  * current_temperature))
        k_opposite = k1 * math.exp(kk)
        
        Wi_H2, Wi_O2, Wi_H2O = W_i(k1, k_opposite, current_h2, current_o2, current_h2o)
        W4 = - (H_H2 * Wi_H2  + H_O2 * Wi_O2 + H_H2O * Wi_H2O)

        # Коэффициенты k1
        k1_h2, k1_o2, k1_h2o = calculate_coefficients(k_opposite, current_h2, current_o2, current_h2o)
        
        # Коэффициенты k2
        h2_temp = current_h2 + 0.5 * k1_h2
        o2_temp = current_o2 + 0.5 * k1_o2
        h2o_temp = current_h2o + 0.5 * k1_h2o
        k2_h2, k2_o2, k2_h2o = calculate_coefficients(k_opposite, h2_temp, o2_temp, h2o_temp)
        
        # Коэффициенты k3
        h2_temp = current_h2 + 0.5 * k2_h2
        o2_temp = current_o2 + 0.5 * k2_o2
        h2o_temp = current_h2o + 0.5 * k2_h2o
        k3_h2, k3_o2, k3_h2o = calculate_coefficients(k_opposite, h2_temp, o2_temp, h2o_temp)
        
        # Коэффициенты k4
        h2_temp = current_h2 + k3_h2
        o2_temp = current_o2 + k3_o2
        h2o_temp = current_h2o + k3_h2o
        k4_h2, k4_o2, k4_h2o = calculate_coefficients(k_opposite, h2_temp, o2_temp, h2o_temp)
        
        # Обновление значений
        h2[i+1] = current_h2 + (k1_h2 + 2 * k2_h2 + 2 * k3_h2 + k4_h2) / 6
        o2[i+1] = current_o2 + (k1_o2 + 2 * k2_o2 + 2 * k3_o2 + k4_o2) / 6
        h2o[i+1] = current_h2o + (k1_h2 + 2 * k2_h2o + 2 * k3_h2o + k4_h2o) / 6  
        n2[i+1] = current_n2
        time[i+1] = current_time + h
        
        h2_newton = h2[i+1] 
        o2_newton = o2[i+1] 
        h2o_newton = h2o[i+1] 
        n2_newton = n2[i+1] 
        
        ro_newton = h2_newton * M_H2 + o2_newton * M_O2 + h2o_newton * M_H2O + n2_newton * M_N2
        gamma_h2_newton = h2_newton / ro_newton
        gamma_o2_newton = o2_newton / ro_newton
        gamma_h2o_newton = h2o_newton / ro_newton
        gamma_n2_newton = gamma_n2
        
        e = 1
        TK = current_temperature 
        
        while e > 0.001:
            H_H2_N = H_thermodynamic(DH_H2, TK, temperature_normal, HYDROGEN_POLYNOMIAL_COEFFICIENTS)
            H_O2_N = H_thermodynamic(DH_O2, TK, temperature_normal, OXYGEN_POLYNOMIAL_COEFFICIENTS)
            H_H2O_N = H_thermodynamic(DH_H2O, TK, temperature_normal, WATER_POLYNOMIAL_COEFFICIENTS)
            
            FT = (gamma_h2_newton * H_H2_N + gamma_o2_newton * H_O2_N + gamma_h2o_newton * H_H2O_N + gamma_n2_newton * (32.403 * TK)) - (114991.1569 + gamma_n2_newton * 64806.0)
            
            CP_H2_N = Cp_thermodynamic(TK, HYDROGEN_POLYNOMIAL_COEFFICIENTS)
            CP_O2_N = Cp_thermodynamic(TK, OXYGEN_POLYNOMIAL_COEFFICIENTS)
            CP_H2O_N = Cp_thermodynamic(TK, WATER_POLYNOMIAL_COEFFICIENTS)
            CP_N2_N = CP_N2
            
            Cp_sum_N = gamma_h2_newton * CP_H2_N + gamma_o2_newton * CP_O2_N + gamma_h2o_newton * CP_H2O_N + gamma_n2_newton * CP_N2_N
            T1 = TK - FT / Cp_sum_N
            e = abs(T1 - TK)
            TK = T1
            
        temperature[i+1] = TK
        
        if i == (N - 1):
            gam_h2[i+1] = cur_h2_gamma
            gam_o2[i+1] = cur_o2_gamma
            gam_h2o[i+1] = cur_h2o_gamma
            gam_n2[i+1] = cur_n2_gamma

        
        if i % 1 == 0:
            output_file.write(
                (
                f'{i:<10} {current_time:<10.7f} {temperature[i]:<13.2f} '
                f'{H_sum:<13.4f} {current_h2:<13.10f} {current_o2:<13.10f} '
                f'{current_h2o:<13.10f} {H_H2:<13.4f} {H_O2:<13.4f} '
                f'{H_H2O:<13.4f} {CP_H2:<13.4f} {CP_O2:<13.4f} '
                f'{CP_H2O:<13.4f} {S_H2:<13.4f} {S_O2:<13.4f} '
                f'{S_H2O:<13.4f} {cur_h2_gamma:<10.5f} {cur_o2_gamma:<10.5f} '
                f'{cur_h2o_gamma:<10.5f} {Wi_H2:<15.4f} {Wi_O2:<15.4f} '
                f'{Wi_H2O:<15.4f} {Si_H2:<13.4f} {Si_O2:<13.4f} '
                f'{Si_H2O:<13.4f} {G_H2:<13.1f} {G_O2:<13.1f} '
                f'{G_H2O:<13.1f} {k1:<15.4f} {k_opposite:<13.20f} {Cp_sum:<13.4f} {W4}\n'
                )
            )
            print(f'{i:<10} {current_time:<10.7f} {current_h2:<13.10f} {current_o2:<13.10f} {current_h2o:<13.10f}')
   
    return time, gam_h2, gam_o2, gam_h2o, gam_n2, temperature
    
# Конец объявления функций, начало программы

h = 1e-10 * 0.5
h_begin = h
N = 400
output = open('out2000-H2O+N2.txt', 'w')

M = M_sum(X_H2, M_H2, X_O2, M_O2, X_H2O, M_H2O, X_N2, M_N2)
Gamma_H2, Gamma_O2, Gamma_H2O, Gamma_N2 = Moll_Mass(X_H2, X_O2, X_H2O, X_N2)
p = density(P0, temperature_changeable, Gamma_H2, Gamma_O2, Gamma_H2O, Gamma_N2)

t, H2, O2, H2O, N2, Temperature = solve_runge_kutta(k1, Gamma_H2, Gamma_O2, Gamma_H2O, Gamma_N2, h, N, output)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].plot(t, H2, label='$H_2$')
axs[0].plot(t, O2, label='$O_2$')
axs[0].plot(t, H2O, label='$H_2O$')
axs[0].plot(t, N2, label='$N_2$')
axs[0].set_xlabel('Время')
axs[0].set_ylabel('Значения')
axs[0].set_title('Решение системы дифференциальных уравнений')
axs[0].legend()
axs[0].grid(True)


axs[1].plot(t, Temperature, label='Температура °K')
axs[1].set_xlabel('Время')
axs[1].set_ylabel('Значения')
axs[1].set_title('Решение системы дифференциальных уравнений')
axs[1].legend()
axs[1].grid(True)
plt.show()
