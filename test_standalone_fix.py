"""Quick test to verify the standalone line selection fix."""
import re

def _is_data_like(text):
    if re.search(r'\d', text):
        return True
    words = text.split()
    if len(words) >= 2:
        upper = sum(1 for w in words if len(w) > 1 and w[0].isupper())
        if upper >= 2:
            return True
    return False

orig_lines = [
    "Adresse de l'etablissement",
    "7 Boulevard des Allies 94600 Choisy-le-Roi",
    "Nom commercial",
    "***Redacted***",
    "Activite(s) exercee(s)",
    "Pratique de l'exercice veterinaire",
    "Date de commencement d'activite",
    "01/04/2008",
    "En attente de la production de la piece justifiant de la capacite",
    "Origine du fonds ou de l'activite",
    "Creation",
    "Mode d'exploitation",
    "Exploitation directe",
]
n = len(orig_lines)
print(f"Lines: {n} (odd={n%2!=0})")

# OLD scoring (without value quality)
best_k_old = 0
best_score_old = (float('inf'), float('inf'))
for k in range(n):
    remaining = [orig_lines[i] for i in range(n) if i != k]
    labels = [remaining[i] for i in range(0, len(remaining), 2)]
    data_count = sum(1 for l in labels if _is_data_like(l))
    max_ll = max(len(l) for l in labels)
    score = (data_count, max_ll)
    if score < best_score_old:
        best_score_old = score
        best_k_old = k
print(f'OLD standalone: index {best_k_old} = "{orig_lines[best_k_old]}"')

# NEW scoring (with value quality)
best_k_new = 0
best_score_new = (float('inf'), float('inf'), float('inf'))
for k in range(n):
    remaining = [orig_lines[i] for i in range(n) if i != k]
    labels = [remaining[i] for i in range(0, len(remaining), 2)]
    values = [remaining[i] for i in range(1, len(remaining), 2)]
    data_count = sum(1 for l in labels if _is_data_like(l))
    value_quality = -sum(1 for v in values if _is_data_like(v))
    max_ll = max(len(l) for l in labels)
    score = (data_count, value_quality, max_ll)
    if score < best_score_new:
        best_score_new = score
        best_k_new = k
print(f'NEW standalone: index {best_k_new} = "{orig_lines[best_k_new]}"')
print()
print('Expected: index 8 = "En attente de la production..."')
print(f"FIX WORKS: {best_k_new == 8}")
