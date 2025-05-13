import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parámetros del sistema
T0 = 90.0         # Temperatura inicial del cuerpo (°C)
Tamb = 25.0       # Temperatura ambiente (°C)
k = 0.07          # Constante de enfriamiento
t0 = 0            # Tiempo inicial (min)
tf = 30           # Tiempo final (min)
n = 30            # Número de pasos
h = (tf - t0) / n # Tamaño de paso (min)

# Definición de la EDO: dT/dt = -k(T - Tamb)
def f(t, T):
    return -k * (T - Tamb)

# Condición inicial
t_vals = [t0]
T_euler = [T0]  # T(0) = 90

# Método de Euler
t = t0
T = T0
for i in range(n):
    T = T + h * f(t, T)
    t = t + h
    t_vals.append(t)
    T_euler.append(T)

# Solución analítica: T(t) = Tamb + (T0 - Tamb) * e^(-k * t)
t_analitica = np.linspace(t0, tf, 200)
T_analitica = Tamb + (T0 - Tamb) * np.exp(-k * t_analitica)

# Mostrar resultados numéricos
print("  t (min) |   T_aproximada (Euler) (°C)")
print("-" * 38)
for t_val, T_val in zip(t_vals, T_euler):
    print(f"{t_val:8.2f} | {T_val:26.6f}")

# Guardar resultados en CSV
df = pd.DataFrame({
    "t (min)": t_vals,
    "T_aproximada (Euler)": T_euler
})
df.to_csv("resultados_enfriamiento.csv", index=False)

# Gráfica
plt.figure(figsize=(8, 5))
plt.plot(t_vals, T_euler, 'o-', label='Solución aproximada (Euler)', color='blue')
plt.plot(t_analitica, T_analitica, '-', label='Solución analítica', color='green')
plt.title('Enfriamiento de un cuerpo (Ley de Newton)')
plt.xlabel('Tiempo (min)')
plt.ylabel('Temperatura (°C)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()