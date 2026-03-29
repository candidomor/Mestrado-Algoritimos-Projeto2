import os
import csv
import time
import random
import sys
import gc
import platform
import psutil
import tracemalloc
import statistics
import multiprocessing
import matplotlib.pyplot as plt

from typing import List, Dict, Any

# Ajuste do limite de recursão elevado para acomodar a estrutura linear de BST pior caso 
sys.setrecursionlimit(200000)

from trees import BST, AVLTree, CommodityRecord

def setup_ultimate_isolation():
    """
    Escala a prioridade do OS e trava a execução em um núcleo físico exato (Core 0),
    anulando o cache-miss penalty de context-switching imposto pelo Windows scheduler.
    """
    try:
        p = psutil.Process(os.getpid())
        
        # 1. Prioridade alta do OS
        if platform.system() == 'Windows':
            p.nice(psutil.HIGH_PRIORITY_CLASS)
        else:
            p.nice(-10) 
            
        # 2. Affinity Lock - Trancando no Nucleo Principal 
        # (psutil pega threads lógicas, então isolamos no 0)
        p.cpu_affinity([0])
        
        print("[OS] Ultimate Isolation Ativado:")
        print("  -> Prioridade elevada para HIGH.")
        print("  -> Afinidade travada estritamente no CPU Core 0.")
        
    except Exception as e:
        print(f"[OS Warnings] Não foi possível ativar o ultra isolamento de CPU: {e}")


def print_v4_hardware_info() -> str:
    def get_size(bytes, suffix="B"):
        factor = 1024
        for unit in ["", "K", "M", "G", "T", "P"]:
            if bytes < factor:
                return f"{bytes:.2f}{unit}{suffix}"
            bytes /= factor

    svmem = psutil.virtual_memory()
    
    info = (
        f"----- HARDWARE PROFILING (V4 ULTRA-ISOLATED) -----\n"
        f"OS System: {platform.system()} {platform.release()} ({platform.version()})\n"
        f"Processor: {platform.processor()}\n"
        f"Physical Cores: {psutil.cpu_count(logical=False)}\n"
        f"Total Logical Cores: {psutil.cpu_count(logical=True)}\n"
        f"Total RAM: {get_size(svmem.total)}\n"
        f"Available RAM: {get_size(svmem.available)}\n"
        f"Execution Environment: HIGH PRIORITY | CPU AFFINITY CORE #0 | GC FREEZE ON \n"
        f"Python Version: {platform.python_version()}\n"
        f"--------------------------------------------------\n"
    )
    print(info)
    return info


def parse_csv(filepath: str) -> List[CommodityRecord]:
    records = []
    with open(filepath, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row['commodity_code']}_{row['date']}"
            records.append(CommodityRecord(key, row))
    return records


def run_warmup(records_sample: List[CommodityRecord]):
    """
    Aquecimento de JIT e L1 cache sem aferição de tempo.
    """
    print(">> Executando Fase de Warm-up V4...")
    bst = BST()
    avl = AVLTree()
    for rec in records_sample:
        bst.insert(rec)
        avl.insert(rec)
    del bst
    del avl
    gc.collect()
    print(">> Warm-up concluído.")


def evaluate_batch_baremetal(tree_instance, data_chunk: List[CommodityRecord]) -> float:
    """
    Mede O TEMPO da forma mais próxima possível do hardware (Bare-Metal) no Python.
    - Evita Context Switches (CPU Affinity prévia)
    - Desliga interrupções de varredura do Coletor de Lixo
    - Usa o Timer mais preciso do sistema para variação na nanosfera
    """
    # Force Garbage Collection out of the way before the freeze
    gc.collect()
    
    # ------------------ SEÇÃO CRÍTICA ------------------ #
    gc.disable() # Congela temporariamente o varredor de memoria nativo do CPython
    start = time.perf_counter()
    
    for rec in data_chunk:
        tree_instance.insert(rec)
        
    end = time.perf_counter()
    gc.enable()  # Devolve o controle para o Sistema Operacional
    # ------------------ FIM SEÇÃO CRÍTICA -------------- #
    
    return end - start


def asymptotic_benchmark_v4(scenario_name: str, records: List[CommodityRecord], chunk_sizes: List[int], repetitions: int = 3):
    """
    Roda testes isolados O(N) com a medição de alta-escala do V4.
    """
    print(f"\n[V4 Ultra-Isolado Assintótica] -> Cenário: {scenario_name}")
    results = []

    for n_size in chunk_sizes:
        data_chunk = records[:n_size]
        bst_times = []
        avl_times = []

        print(f"  Avaliando lote de precisão L1 com N = {n_size} ...")

        for _ in range(repetitions):
            bst = BST()
            avl = AVLTree()
            
            # Bare-metal metrics
            bst_t = evaluate_batch_baremetal(bst, data_chunk)
            avl_t = evaluate_batch_baremetal(avl, data_chunk)

            bst_times.append(bst_t)
            avl_times.append(avl_t)
            
            del bst
            del avl
            gc.collect()

        bst_mean = statistics.mean(bst_times)
        avl_mean = statistics.mean(avl_times)
        
        results.append({
            "N": n_size,
            "bst_mean": bst_mean,
            "avl_mean": avl_mean,
            "bst_stdev": statistics.stdev(bst_times) if repetitions > 1 else 0,
            "avl_stdev": statistics.stdev(avl_times) if repetitions > 1 else 0
        })

    return {"scenario": scenario_name, "data": results}


def generate_v4_plots(all_curves: List[Dict], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    for scenario_dict in all_curves:
        scenario = scenario_dict["scenario"]
        metrics = scenario_dict["data"]
        
        N_vals = [m["N"] for m in metrics]
        bst_means = [m["bst_mean"] for m in metrics]
        avl_means = [m["avl_mean"] for m in metrics]

        plt.figure(figsize=(10, 6))
        # BST Curve Plotting
        plt.plot(N_vals, bst_means, marker='o', label='BST (GC Freezed/Isolated)', color='darkred', linewidth=2)
        
        # AVL Curve Plotting
        plt.plot(N_vals, avl_means, marker='s', label='AVL (GC Freezed/Isolated)', color='darkgreen', linewidth=2)

        plt.title(f"V4: Análise Assintótica O(N) com Ultra Isolamento Bare-Metal - {scenario}")
        plt.xlabel("Quantidade de Elementos Chave (N)")
        plt.ylabel("Latência Lógica Pura Acumulada (Segundos)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        safe_name = scenario.replace(" ", "_").replace("(", "").replace(")", "").lower()
        plt.savefig(os.path.join(output_dir, f'v4_baremetal_{safe_name}.png'))
        plt.close()


def generate_v4_text_report(hw_info: str, all_curves: List[Dict], output_path: str):
    log = []
    log.append(hw_info)
    
    log.append("\n==========================================================")
    log.append("  ANÁLISE BARE-METAL V4 (Exclusão do Context Switching OS)")
    log.append("==========================================================")
    log.append(" NOTA TÉCNICA: Os tempos abaixo representam a latência de ")
    log.append(" processamento orgânico real do algoritmo despido do peso ")
    log.append(" do Garbage Collector CPython e interferência do Windows.")
    log.append("==========================================================")

    for scenario_dict in all_curves:
        log.append(f"\n>>>> CENÁRIO: {scenario_dict['scenario']} <<<<")
        log.append(f"  | Vol (N)  |  BST V4 (s)     |  AVL V4 (s)     |")
        log.append(f"  |----------|-----------------|-----------------|")
        for m in scenario_dict["data"]:
            log.append(f"  | {m['N']:<8} |  {m['bst_mean']:<13.4f}  |  {m['avl_mean']:<13.4f}  |")

    full_text = "\n".join(log)
    print(full_text)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)


if __name__ == '__main__':
    base_dir = r"C:\Users\Cândido Moreira\Dropbox\Mestrado Profissional\Matérias\Obrigatórias\Algoritmos e Programação\Projeto 02"
    csv_file = os.path.join(base_dir, "wb_commodity_price_intelligence_1960_2026.csv")
    
    output_dir_v4 = os.path.join(base_dir, "relatorios", "v4")
    os.makedirs(output_dir_v4, exist_ok=True)
    os.makedirs(os.path.join(output_dir_v4, "imagens"), exist_ok=True)

    # 1. Ultimate Isolation e Hardware Report
    setup_ultimate_isolation()
    hw_info = print_v4_hardware_info()

    # 2. Extract Data
    records = parse_csv(csv_file)
    
    # Gerar os 3 Datasets
    ordered_records = records.copy()
    
    reversed_records = records.copy()
    reversed_records.reverse()
    
    shuffled_records = records.copy()
    random.seed(99)
    random.shuffle(shuffled_records)

    # 3. Warm-up
    run_warmup(shuffled_records[:1000])

    # 4. Benchmark V4
    chunk_scales = [10000, 20000, 30000, 40000, len(records)]
    reps = 3 

    curve_results = []
    curve_results.append(asymptotic_benchmark_v4("Ordered (Crescente)", ordered_records, chunk_scales, reps))
    curve_results.append(asymptotic_benchmark_v4("Reversed (Decrescente)", reversed_records, chunk_scales, reps))
    curve_results.append(asymptotic_benchmark_v4("Shuffled (Aleatório)", shuffled_records, chunk_scales, reps))

    # 5. Compilação
    generate_v4_plots(curve_results, os.path.join(output_dir_v4, "imagens"))
    generate_v4_text_report(hw_info, curve_results, os.path.join(output_dir_v4, "resultados_v4.txt"))

    print("\nBenchmark V4 Bare-Metal Concluído! Relatórios rigorosos gerados.")
