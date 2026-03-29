import json
import torch
import gc
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

# ================================
# CONFIGURATION
# ================================
BASE_DIR = os.getcwd()
MAX_NEW_TOKENS = 50
MODELS_CONFIG_FILE = os.path.join(BASE_DIR, "models_config.json")

# ================================
# GÉNÉRATION
# ================================
def generate_response(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ================================
# TRAITEMENT BENCHMARK
# ================================
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
        prompt = entry.get("prompt")

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

        elif prompt:
            response = generate_response(model, tokenizer, prompt)
            results.append({
                "id": entry.get("id"),
                "benchmark": entry.get("benchmark"),
                "focus": entry.get("focus"),
                "prompt": prompt,
                "response": response,
                "expected_behavior": entry.get("expected_behavior"),
                "model_info": model_meta
            })

        else:
            print("Entrée ignorée id=", entry.get("id"))

        # affichage progression
        if i % 20 == 0 or i == total:
            print(f"  {i}/{total} ({int(i/total*100)}%)")

        # nettoyage mémoire
        if i % 20 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ================================
# MAIN
# ================================
def run_runner(benchmark_choice):

    # lecture JSON robuste
    try:
        with open(MODELS_CONFIG_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
            if isinstance(raw, dict) and "models" in raw:
                models_list = raw["models"]
            elif isinstance(raw, list):
                models_list = raw
            else:
                print("❌ Format JSON non reconnu dans models_config.json")
                return
    except Exception as e:
        print("Erreur lecture models_config.json :", e)
        return

    print(f"✅ {len(models_list)} modèles chargés")

    # benchmarks
    pairs_file = os.path.join(BASE_DIR, "benchmarks_pairs.json")
    fused_file = os.path.join(BASE_DIR, "benchmarks_fused.json")

    if benchmark_choice == "pairs":
        benchmarks = [pairs_file]
        output_path = "results_runner_pairs.json"

    elif benchmark_choice == "fused":
        benchmarks = [fused_file]
        output_path = "results_runner_fused.json"

    else:
        benchmarks = [pairs_file, fused_file]
        output_path = "results_runner_all.json"

    all_results = []

    for model_meta in models_list:

        # skip modèles API pour éviter les crashs
        if model_meta.get("api") is True:
            print(f"⏭️ Skip API model : {model_meta.get('model_name')}")
            continue

        model_name = model_meta.get("model_name")
        print("\n==============================")
        print("Modèle :", model_name)
        print("==============================")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)

            if torch.cuda.is_available():
                model.to("cuda")

        except Exception as e:
            print(f"Erreur modèle {model_name} :", e)
            continue

        model.eval()

        for bench in benchmarks:
            print("Traitement :", bench)
            process_benchmark_file(model, tokenizer, model_meta, bench, all_results)

        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # sauvegarde
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n==============================")
    print("FINI ✅")
    print("Résultats :", output_path)
    print("==============================")


# ================================
# ENTRY POINT
# ================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        choices=["pairs", "fused", "all"],
        default="all"
    )
    args = parser.parse_args()
    run_runner(args.benchmark)
