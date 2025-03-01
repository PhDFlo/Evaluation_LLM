import torch, torchvision
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
import matplotlib.pyplot as plt
import psutil
import gc

# Liste des modèles à évaluer
models = [
    # ("albert-large-v2", 18),  # 18M parameters
    # ("albert-xlarge-v2", 60),  # 60M parameters
    # ("distilbert-base-uncased", 67),  # 67M parameters
    # ("bert-base-uncased", 110),  # 110M parameters
    # ("roberta-base", 125),  # 125M parameters
    # ("bert-large-uncased", 335),  # 335M parameters
    # ("roberta-large", 355),  # 355M parameters
    # ("gpt2-large", 774),  # 774M parameters
    # ("gpt2-xl", 1500), # 1.5B parameters
    # ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 1500), # 1.5B parameters
    # ("EleutherAI/gpt-neo-2.7B", 2700) # 2.7B parameters
    ("TheBloke/Llama-2-13B-chat-GPTQ", 13000),  # 13B parameters
]

# Texte d'exemple
text = "The quick brown fox jumps over the lazy dog."

# Initialiser les listes pour les résultats
num_parameters_list = []
tokens_per_second_list = []
memory_usage_list = []

for model_id, num_params in models:
    # Charger le tokenizer et le modèle
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)

    # Afficher le nombre de poids du modèle
    num_parameters = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Number of parameters in the model ({model_id}): {num_parameters}M")

    # Tokenizer le texte
    inputs = tokenizer(text, return_tensors="pt")

    # Mesurer le temps de traitement et la consommation mémoire
    start_time = time.time()
    process = psutil.Process()

    # Prendre plusieurs mesures de la mémoire avant l'inférence
    start_memory_samples = [process.memory_info().rss / 1e6 for _ in range(5)]
    start_memory = sum(start_memory_samples) / len(start_memory_samples)

    outputs = model(**inputs)

    # Prendre plusieurs mesures de la mémoire après l'inférence
    end_memory_samples = [process.memory_info().rss / 1e6 for _ in range(5)]
    end_memory = sum(end_memory_samples) / len(end_memory_samples)

    end_time = time.time()

    # Calculer le nombre de tokens par seconde
    num_tokens = len(inputs["input_ids"][0])
    time_taken = end_time - start_time
    tokens_per_second = num_tokens / time_taken
    memory_usage = end_memory - start_memory

    print(f"Number of tokens: {num_tokens}")
    print(f"Time taken: {time_taken:.4f} seconds")
    print(f"Tokens per second: {tokens_per_second:.2f}")
    print(f"Memory usage: {memory_usage:.2f} MB")

    # Ajouter les résultats aux listes
    num_parameters_list.append(num_parameters)
    tokens_per_second_list.append(tokens_per_second)
    memory_usage_list.append(memory_usage)

    # Libérer la mémoire
    del model
    del tokenizer
    torch.cuda.empty_cache()
    torch.cuda.synchronize()  # Forcer le vidage du cache CUDA
    gc.collect()  # Forcer la collecte des objets inutilisés

# Tracer le graphique
fig, ax1 = plt.subplots(figsize=(10, 6))

color = "tab:blue"
ax1.set_xlabel("Number of Parameters (in millions)")
ax1.set_ylabel("Tokens per Second", color=color)
ax1.plot(num_parameters_list, tokens_per_second_list, marker="o", color=color)
ax1.tick_params(axis="y", labelcolor=color)

ax2 = ax1.twinx()  # Instancier un second axe qui partage le même axe x
color = "tab:red"
ax2.set_ylabel("Memory Usage (MB)", color=color)
ax2.plot(num_parameters_list, memory_usage_list, marker="x", color=color)
ax2.tick_params(axis="y", labelcolor=color)

fig.tight_layout()  # Pour éviter que les labels se chevauchent
plt.title("Tokens per Second and Memory Usage vs Number of Parameters")
plt.grid(True)
plt.show()

# Vider la RAM à la fin du script
gc.collect()
