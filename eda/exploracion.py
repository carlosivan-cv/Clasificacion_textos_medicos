import os
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from nltk.corpus import stopwords
import nltk

# Descargar stopwords (solo la primera vez)
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Ruta base a tus datos
base_path = r"../data"

data = []
for split in ["train", "test"]:
    for label in ["pls", "non_pls"]:
        folder = os.path.join(base_path, split, label)
        for file in os.listdir(folder):
            if file.endswith((".txt", ".md")):
                filepath = os.path.join(folder, file)
                with open(filepath, encoding="utf-8") as f:
                    text = f.read()
                    data.append({
                        "file": file,
                        "split": split,   # train o test
                        "label": label,   # pls o non_pls
                        "text": text,
                        "raw_length": len(text.split())  # longitud sin procesar (en palabras)
                    })

# Guardar en un DataFrame
df = pd.DataFrame(data)
print("Tamaño total:", df.shape)
print(df.head())

# ======================================================
# 1. Conteo de archivos por combinación
# ======================================================
counts = df.groupby(["split", "label"]).size().reset_index(name="count")
print("\nCantidad de documentos por combinación:\n", counts)

plt.figure(figsize=(6,4))
sns.barplot(data=counts, x="split", y="count", hue="label")
plt.title("Número de documentos por categoría")
plt.ylabel("Número de archivos")
plt.show()

# ======================================================
# 2. Longitud promedio del texto (sin procesar)
# ======================================================
avg_length = df.groupby(["split", "label"])["raw_length"].mean().reset_index()
print("\nLongitud promedio (palabras) por combinación:\n", avg_length)

plt.figure(figsize=(6,4))
sns.barplot(data=avg_length, x="split", y="raw_length", hue="label")
plt.title("Longitud promedio de documentos")
plt.ylabel("Promedio de palabras")
plt.show()

# ======================================================
# 3. Top 5 palabras por combinación (sin stopwords)
# ======================================================
def clean_tokens(text):
    tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return tokens

df["tokens"] = df["text"].apply(clean_tokens)

print("\nTop 5 palabras más frecuentes por combinación:")
for (split, label), subset in df.groupby(["split", "label"]):
    all_words = [word for tokens in subset["tokens"] for word in tokens]
    top_words = Counter(all_words).most_common(5)
    print(f"{split}-{label}: {top_words}")
