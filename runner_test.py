import json
import torch
import gc
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# ================================
# CONFIGURATION POUR LE TEST
# ================================
BENCHMARK_FILE = os.path.join(os.getcwd(), "data", "benchmarks_test.json")
MODELS_CONFIG_FILE = os.path.join(os.getcwd(), "models_config.json")
OUTPUT_FILE = "results_test.json"
MAX_NEW_TOKENS = 50


def generate_response(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        model = model.to("cuda")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            top_k=50
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def process_benchmark_file(model, tokenizer, model_meta, benchmark_file, results):
    if not os.path.exists(benchmark_file):
        print(f"⚠️ Fichier introuvable : {benchmark_file}")
        return
    with open(benchmark_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    total = len(data)
    print(f"- {total} entrées dans {os.path.basename(benchmark_file)}")
    for i, entry in enumerate(data, 1):
        stereotype = entry.get("stereotype_prompt")
        anti = entry.get("anti_stereotype_prompt")
        if stereotype and anti:
            response_stereo = generate_response(model, tokenizer, stereotype)
            response_anti = generate_response(model, tokenizer, anti)
            results.append({
                "id": entry.get("id"),
                "benchmark": entry.get("benchmark"),
                "focus": entry.get("focus"),
                "stereotype_prompt": stereotype,
                "anti_stereotype_prompt": anti,
                "response_stereotype": response_stereo,
                "response_anti": response_anti,
                "model_info": model_meta
            })
        else:
            print("Entrée ignorée (pas de prompts stéréotype/anti-stéréotype) id=", entry.get("id"))
        if i % 5 == 0 or i == total:
            print(f"  Traitement {i}/{total} ({int(i/total*100)}%)")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_test_runner():
    try:
        with open(MODELS_CONFIG_FILE, "r", encoding="utf-8") as f:
            models_list = json.load(f)
    except FileNotFoundError:
        print(f"Fichier de config manquant : {MODELS_CONFIG_FILE}")
        return

    all_results = []
    for model_meta in models_list:
        model_name = model_meta.get("model_name")
        print("==============================")
        print("Modèle :", model_name)
        print("==============================")
        print("Chargement du tokenizer et du modèle...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
        except Exception as e:
            print(f"Erreur chargement du modèle '{model_name}': {e}")
            print("-> Ignoré, on passe au modèle suivant.")
            continue

        model.eval()

        print("Traitement du benchmark de test :", BENCHMARK_FILE)
        process_benchmark_file(model, tokenizer, model_meta, BENCHMARK_FILE, all_results)

        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print("==============================")
    print("Test terminé")
    print("Résultats :", OUTPUT_FILE)
    print("==============================")


if __name__ == "__main__":
    run_test_runner()