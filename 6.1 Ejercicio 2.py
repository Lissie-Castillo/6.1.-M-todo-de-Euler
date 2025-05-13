import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parámetros del sistema
g = 9.81       # Aceleración debida a la gravedad (m/s^2)
m = 2.0        # Masa del objeto (kg)
k = 0.5        # Coeficiente de fricción lineal (kg/s)
t0 = 0         # Tiempo inicial (s)
tf = 10        # Tiempo final (s)
n = 50         # Número de pasos
h = (tf - t0) / n  # Tamaño de paso

# Definición de la EDO: dv/dt = g - (k/m)v
def f(t, v):
    return g - (k / m) * v

# Condición inicial
t_vals = [t0]
v_euler = [0]  # v(0) = 0

# Método de Euler
t = t0
v = 0
for i in range(n):
    v = v + h * f(t, v)
    t = t + h
    t_vals.append(t)
    v_euler.append(v)

# Solución analítica: v(t) = (mg/k)(1 - e^(-(k/m)t))
t_analitica = np.linspace(t0, tf, 200)
v_analitica = (m * g / k) * (1 - np.exp(-(k / m) * t_analitica))

# Mostrar resultados numéricos
print("  t (s)   |   v_aproximada (Euler)")
print("-" * 32)
for t_val, v_val in zip(t_vals, v_euler):
    print(f"{t_val:7.2f} | {v_val:21.6f}")

# Guardar resultados en CSV
df = pd.DataFrame({
    "t (s)": t_vals,
    "v_aproximada (Euler)": v_euler
})
df.to_csv("resultados_caida_libre.csv", index=False)

# Gráfica
plt.figure(figsize=(8, 5))
plt.plot(t_vals, v_euler, 'o-', label='Solución aproximada (Euler)', color='blue')
plt.plot(t_analitica, v_analitica, '-', label='Solución analítica', color='green')
plt.title('Caída libre con resistencia del aire')
plt.xlabel('Tiempo (s)')
plt.ylabel('Velocidad (m/s)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()