import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
import matplotlib.pyplot as plt

# Liste des modèles à évaluer
models = [    
    ("albert-large-v2", 18), # 18M parameters
    ("albert-xlarge-v2", 60), # 60M parameters
    ("distilbert-base-uncased", 67), # 67M parameters
    ("bert-base-uncased", 110), # 110M parameters
    ("roberta-base", 125), # 125M parameters
    ("bert-large-uncased", 335), # 335M parameters
    ("roberta-large", 355), # 355M parameters
    ("gpt2-large", 774), # 774M parameters
    ("gpt2-xl", 1500), # 1.5B parameters
    #("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 1500), # 1.5B parameters
    #("EleutherAI/gpt-neo-2.7B", 2700) # 2.7B parameters
]

# Texte d'exemple
text = "The quick brown fox jumps over the lazy dog."

# Initialiser les listes pour les résultats
num_parameters_list = []
tokens_per_second_list = []

for model_id, num_params in models:
    # Charger le tokenizer et le modèle
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)

    # Afficher le nombre de poids du modèle
    num_parameters = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Number of parameters in the model ({model_id}): {num_parameters}M")

    # Tokenizer le texte
    inputs = tokenizer(text, return_tensors="pt")

    # Mesurer le temps de traitement
    start_time = time.time()
    outputs = model(**inputs)
    end_time = time.time()

    # Calculer le nombre de tokens par seconde
    num_tokens = len(inputs["input_ids"][0])
    time_taken = end_time - start_time
    tokens_per_second = num_tokens / time_taken

    print(f"Number of tokens: {num_tokens}")
    print(f"Time taken: {time_taken:.4f} seconds")
    print(f"Tokens per second: {tokens_per_second:.2f}")

    # Ajouter les résultats aux listes
    num_parameters_list.append(num_parameters)
    tokens_per_second_list.append(tokens_per_second)

    # Libérer la mémoire
    del model
    del tokenizer
#   torch.cuda.empty_cache()

# Tracer le graphique
plt.figure(figsize=(10, 6))
plt.plot(num_parameters_list, tokens_per_second_list, marker='o')
plt.xlabel('Number of Parameters (in millions)')
plt.ylabel('Tokens per Second')
plt.title('Tokens per Second vs Number of Parameters')
plt.grid(True)
plt.show()