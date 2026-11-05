import numpy as np
from kernel import load
E = load('model.language_model.embed_tokens.weight')
g = 1.0 + load('model.language_model.norm.weight')
d = g * (E[248069] - E.mean(0)); d /= np.linalg.norm(d)
lift = E @ (g * d)
for t in [248069, 220, 12, 248044, 198, 248045, 248046, 248068, 248043, 248047]:
    print(t, round(float(lift[t]), 3))
print('cos(E_think, E_248044)', float(E[248069] @ E[248044] / np.linalg.norm(E[248069]) / np.linalg.norm(E[248044])))
