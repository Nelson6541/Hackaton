from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "distilgpt2"

print("Chargement du modèle...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "Explain cultural diversity in society."

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,   
    max_new_tokens=50
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nRéponse du modèle :\n")
print(response)