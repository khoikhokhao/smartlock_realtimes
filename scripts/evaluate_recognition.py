import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support

# === 1. Load embeddings ===
ENC_FILE = "face_encodings.pkl"
with open(ENC_FILE, 'rb') as f:
    encs, names = pickle.load(f)

encs = np.array(encs)
names = np.array(names)
unique_names = list(set(names))
print(f"[INFO] Loaded {len(encs)} encodings for {len(unique_names)} person(s).")

# === 2. Build pairwise distances ===
pairs, labels, distances = [], [], []
for i in range(len(encs)):
    for j in range(i + 1, len(encs)):
        dist = np.linalg.norm(encs[i] - encs[j])
        same = 1 if names[i] == names[j] else 0
        distances.append(dist)
        labels.append(same)
distances = np.array(distances)
labels = np.array(labels)

# === 3. ROC & AUC ===
fpr, tpr, thresholds = roc_curve(labels, -distances)  # invert distances (smaller=better)
roc_auc = auc(fpr, tpr)

# EER (Equal Error Rate)
fnr = 1 - tpr
eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]

print(f"\n[RESULTS]")
print(f"AUC (ROC): {roc_auc:.4f}")
print(f"EER: {eer:.4f} at threshold {eer_threshold:.4f}")

# === 4. Precision / Recall / F1 tại ngưỡng 0.28 ===
THRESH = 0.28
preds = (distances < THRESH).astype(int)
prec, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
print(f"Threshold={THRESH:.2f} -> Precision={prec:.3f}  Recall={recall:.3f}  F1={f1:.3f}")

# === 5. Plot ROC curve ===
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC (AUC={roc_auc:.3f})')
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Face Recognition')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
