import re
import numpy as np
import matplotlib.pyplot as plt

# 1. Les données fournies
data = """
253
PCG iter: 155
PCG time: 1.89 ms.
CG iter: 619
CG time: 1.99 ms.


291
PCG iter: 158
PCG time: 3.05 ms.
CG iter: 651
CG time: 3.75 ms.


344
PCG iter: 157
PCG time: 4.17 ms.
CG iter: 727
CG time: 7.76 ms.


427
PCG iter: 171
PCG time: 3.75 ms.
CG iter: 811
CG time: 4.28 ms.


491
PCG iter: 180
PCG time: 6.76 ms.
CG iter: 878
CG time: 8.98 ms.


655
PCG iter: 188
PCG time: 12.05 ms.
CG iter: 1026
CG time: 8.32 ms.


1009
PCG iter: 264
PCG time: 17.40 ms.
CG iter: 1288
CG time: 15.06 ms.



1685
PCG iter: 330
PCG time: 32.56 ms.
CG iter: 1620
CG time: 31.52 ms.



3544
PCG iter: 480
PCG time: 109.04 ms.
CG iter: 2348
CG time: 91.22 ms.



13478
PCG iter: 949
PCG time: 1271.01 ms.
CG iter: 4286
CG time: 637.98 ms.



16503
PCG iter: 1057
PCG time: 1810.26 ms.
CG iter: 4916
CG time: 890.62 ms.
"""

# 2. Extraire les valeurs N, PCG time, CG time
# Utilisation d'une expression régulière pour trouver les blocs de données
pattern = re.compile(r"(\d+)\s+PCG iter:.*?PCG time:\s*([\d.]+)\s*ms\.\s+CG iter:.*?CG time:\s*([\d.]+)\s*ms\.", re.DOTALL)
matches = pattern.findall(data)

# Listes pour stocker les valeurs extraites
n_values = []
pcg_times = []
cg_times = []

# Remplir les listes en convertissant les chaines en nombres
for match in matches:
    n_values.append(int(match[0]))
    pcg_times.append(float(match[1]))
    cg_times.append(float(match[2]))

# Convertir les listes en arrays NumPy pour faciliter les calculs (comme log10)
n_values = np.array(n_values)
pcg_times = np.array(pcg_times)
cg_times = np.array(cg_times)

# 3. Calculer log10(N) pour l'axe des X
x_log_n = np.log10(n_values)

# 4. Créer le graphique
plt.figure(figsize=(10, 6)) # Optionnel: ajuste la taille de la figure

# Tracer les temps PCG en fonction de log10(N)
plt.plot(x_log_n, pcg_times, marker='o', linestyle='-', label='PCG Time (ms)')

# Tracer les temps CG en fonction de log10(N)
plt.plot(x_log_n, cg_times, marker='s', linestyle='--', label='CG Time (ms)')

# 5. Ajouter les étiquettes, le titre, la légende et une grille
plt.xlabel('log10(N)') # Etiquette de l'axe X
plt.ylabel('Time (ms)') # Etiquette de l'axe Y
plt.title('Temps PCG vs CG en fonction de log10(N)') # Titre du graphique
plt.legend() # Affiche la légende (basée sur les 'label' des plots)
plt.grid(True) # Ajoute une grille pour faciliter la lecture

# 6. Afficher le graphique à l'écran
plt.tight_layout() # Ajuste automatiquement les marges pour que tout soit visible
plt.show()