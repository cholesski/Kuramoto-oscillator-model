import numpy as np
import matplotlib.pyplot as plt

def kuramoto(theta, K, omega):
  N = len(theta)
  dtheta_dt = np.zeros(N)
  for i in range (N):
    sum = 0
    for j in range (N):
      sum += np.sin(theta[j] - theta[i])
      dtheta_dt[i] += omega[i] + K/N * sum
  return dtheta_dt

def rk4(theta, K, omega, h, timesteps):
  history = [theta.copy()]

  for _ in range(timesteps):
    k1 = h * kuramoto(theta, K, omega)
    k2 = h * kuramoto(theta + 0.5 * k1, K, omega)
    k3 = h * kuramoto(theta + 0.5 * k2, K, omega)
    k4 = h * kuramoto(theta + k3, K, omega)

    theta = theta + (1.0/6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    history.append(theta.copy())

  return np.array(history)


N = 100 # broj oscilatora
timesteps = 1000 # vreme simulacije
h = 0.001 # korak
K = 0.8 # jacina uticaja medju oscilatorima

theta_initial = np.random.uniform(0, 2 * np.pi, N)
omega = np.random.uniform(0.5, 2, N)

theta_history = rk4(theta_initial, K, omega, h, timesteps)

plt.figure(figsize=(10,6))
for i in range(N):
  plt.plot(theta_history[:, i], label=f"Oscilator {i+1}")

plt.title(f"Kuramoto model - Sistem sa {N} oscilatora")
plt.xlabel("Vremenski korak")
plt.ylabel("Theta")
plt.legend()
plt.show()
