import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parámetros del circuito
R = 1000          # Ohmios
C = 0.001         # Faradios
V_fuente = 5      # Voltios
t0 = 0            # Tiempo inicial
tf = 5            # Tiempo final
n = 20            # Número de pasos
h = (tf - t0) / n # Tamaño de paso

# Definición de la EDO: dV/dt = (1/RC)(Vfuente - V)
def f(t, V):
    return (1 / (R * C)) * (V_fuente - V)

# Condición inicial
t_vals = [t0]
V_euler = [0]  # V(0) = 0

# Método de Euler
t = t0
V = 0
for i in range(n):
    V = V + h * f(t, V)
    t = t + h
    t_vals.append(t)
    V_euler.append(V)

# Solución analítica
t_analitica = np.linspace(t0, tf, 100)
V_analitica = V_fuente * (1 - np.exp(-t_analitica / (R * C)))

# Mostrar resultados numéricos
print("  t (s)   |   V_aproximada (Euler)")
print("-" * 32)
for t_val, V_val in zip(t_vals, V_euler):
    print(f"{t_val:7.2f} | {V_val:21.6f}")

# Guardar resultados en CSV
df = pd.DataFrame({
    "t (s)": t_vals,
    "V_aproximada (Euler)": V_euler
})
df.to_csv("resultados_carga_capacitor.csv", index=False)

# Gráfica
plt.figure(figsize=(8, 5))
plt.plot(t_vals, V_euler, 'o-', label='Solución aproximada (Euler)', color='blue')
plt.plot(t_analitica, V_analitica, '-', label='Solución analítica', color='green')
plt.title('Carga de un Capacitor en Circuito RC')
plt.xlabel('Tiempo (s)')
plt.ylabel('Voltaje (V)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()