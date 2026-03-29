import os
import csv
import time
import random
import sys
import gc
import platform
import psutil
import multiprocessing
import matplotlib.pyplot as plt

from typing import List, Dict, Any

# Ajuste do limite de recursão elevado para pior caso
sys.setrecursionlimit(200000)

from trees import BST, AVLTree, CommodityRecord


def setup_high_priority():
    try:
        p = psutil.Process(os.getpid())
        if platform.system() == 'Windows':
            p.nice(psutil.HIGH_PRIORITY_CLASS)
        else:
            p.nice(-10) 
    except Exception as e:
        pass


def print_v5_hardware_info() -> str:
    def get_size(bytes, suffix="B"):
        factor = 1024
        for unit in ["", "K", "M", "G", "T", "P"]:
            if bytes < factor:
                return f"{bytes:.2f}{unit}{suffix}"
            bytes /= factor

    svmem = psutil.virtual_memory()    
    info = (
        f"----- HARDWARE PROFILING (V5 OVERHEAD TEST) -----\n"
        f"OS System: {platform.system()} {platform.release()} ({platform.version()})\n"
        f"Processor: {platform.processor()}\n"
        f"Physical Cores: {psutil.cpu_count(logical=False)}\n"
        f"Total Logical Cores: {psutil.cpu_count(logical=True)}\n"
        f"Total RAM: {get_size(svmem.total)}\n"
        f"Execution Priority: Elevada (High Priority)\n"
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

# -------------------------------------------------------------------------
# MULTIPROCESSING WORKERS
# -------------------------------------------------------------------------

def _worker_process_insertion(records_chunk: List[CommodityRecord], tree_type: str) -> float:
    """Função alvo do Pool que será rodada paralela e isoladamente em N núcleos"""
    tree = BST() if tree_type == 'BST' else AVLTree()
    
    # Force Garbage Collection out of the way before the freeze for maximum isolated metric
    gc.collect()
    gc.disable()
    
    start = time.perf_counter()
    for rec in records_chunk:
        tree.insert(rec)
    end = time.perf_counter()
    
    gc.enable()
    return end - start

# -------------------------------------------------------------------------
# BENCHMARK ENGINE
# -------------------------------------------------------------------------

def single_thread_eval(tree_class, data: List[CommodityRecord]) -> float:
    tree = tree_class()
    gc.collect()
    gc.disable()
    
    start = time.perf_counter()
    for rec in data:
        tree.insert(rec)
    end = time.perf_counter()
    
    gc.enable()
    return end - start


def multi_process_eval(tree_type_str: str, data: List[CommodityRecord], cores: int = 4) -> float:
    """Divide a carga nos Cores informados e tira o overhead total"""
    n = len(data)
    chunk_size = n // cores
    chunks = []
    
    for i in range(cores):
        start_idx = i * chunk_size
        # O último chunk leva o resto
        end_idx = n if i == cores - 1 else (i + 1) * chunk_size 
        chunks.append(data[start_idx:end_idx])

    start_wall_clock = time.perf_counter()
    
    with multiprocessing.Pool(processes=cores) as pool:
        results = []
        for c in chunks:
            results.append(pool.apply_async(_worker_process_insertion, (c, tree_type_str)))
        
        # O tempo isolado de cada thread ignora o overhead de IPC (Inter-process comm).
        # Ao forçarmos o r.get() trancamos o Wall-Clock provando o tráfego brutal de submissao dos dados 
        # (Pickling arguments) do Pydict
        [r.get() for r in results]

    end_wall_clock = time.perf_counter()
    
    return end_wall_clock - start_wall_clock


def benchmark_v5_overhead(scenario_name: str, records: List[CommodityRecord], chunk_sizes: List[int]):
    """
    Roda os testes escalares Single Thread x Multi Process e reporta o crescimento do IPC/Overhead.
    """
    print(f"\n[V5 Multiprocess Overhead] -> Cenário: {scenario_name}")
    results = []

    for n_size in chunk_sizes:
        data_chunk = records[:n_size]
        
        print(f"  Avaliando Escabilidade MPI com N = {n_size} ...")

        # -----------------------------
        # 1. Avaliação Single-Thread
        # -----------------------------
        bst_st = single_thread_eval(BST, data_chunk)
        avl_st = single_thread_eval(AVLTree, data_chunk)
        
        # -----------------------------
        # 2. Avaliação Multi-Process (4 Cores)
        # -----------------------------
        bst_mp = multi_process_eval("BST", data_chunk, cores=4)
        avl_mp = multi_process_eval("AVL", data_chunk, cores=4)
        
        results.append({
            "N": n_size,
            "bst_st": bst_st,
            "avl_st": avl_st,
            "bst_mp": bst_mp,
            "avl_mp": avl_mp
        })

    return {"scenario": scenario_name, "data": results}

# -------------------------------------------------------------------------
# GENERATION
# -------------------------------------------------------------------------

def generate_v5_plots(all_curves: List[Dict], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    for scenario_dict in all_curves:
        scenario = scenario_dict["scenario"]
        metrics = scenario_dict["data"]
        
        N_vals = [m["N"] for m in metrics]
        
        bst_st = [m["bst_st"] for m in metrics]
        bst_mp = [m["bst_mp"] for m in metrics]
        
        avl_st = [m["avl_st"] for m in metrics]
        avl_mp = [m["avl_mp"] for m in metrics]

        # ---- PLOT 1: BST Single vs MP ----
        plt.figure(figsize=(10, 6))
        plt.plot(N_vals, bst_st, marker='o', label='BST (Single-Thread 1 Core)', color='blue', linewidth=2)
        plt.plot(N_vals, bst_mp, marker='s', label='BST (Multi-Process 4 Cores)', color='red', linestyle='--', linewidth=2)
        
        plt.title(f"V5 Multiprocessing Overhead - Tempo de Montagem BST ({scenario})")
        plt.xlabel("Quantidade de Elementos Chave Repartidos (N)")
        plt.ylabel("Tempo Ptotal (Segundos)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper left')
        plt.tight_layout()
        safe_name = scenario.replace(" ", "_").replace("(", "").replace(")", "").lower()
        plt.savefig(os.path.join(output_dir, f'v5_overhead_bst_{safe_name}.png'))
        plt.close()

        # ---- PLOT 2: AVL Single vs MP ----
        plt.figure(figsize=(10, 6))
        plt.plot(N_vals, avl_st, marker='o', label='AVL (Single-Thread 1 Core)', color='green', linewidth=2)
        plt.plot(N_vals, avl_mp, marker='s', label='AVL (Multi-Process 4 Cores)', color='orange', linestyle='--', linewidth=2)
        
        plt.title(f"V5 Multiprocessing Isolado - IPC Penalty na AVL ({scenario})")
        plt.xlabel("Quantidade de Elementos Chave (N)")
        plt.ylabel("Tempo Total / IPC Delay (Segundos)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'v5_overhead_avl_{safe_name}.png'))
        plt.close()


def generate_v5_text_report(hw_info: str, all_curves: List[Dict], output_path: str):
    log = []
    log.append(hw_info)
    
    log.append("\n==========================================================")
    log.append("  O PARADOXO DO PARALELISMO MATRICIAL (Pickle IPC Delay)")
    log.append("==========================================================")

    for scenario_dict in all_curves:
        log.append(f"\n>>>> CENÁRIO: {scenario_dict['scenario']} <<<<")
        log.append(f"  | Vol (N)  | BST ST(s) | BST MP 4-Cores | Atraso BST MP % | AVL ST(s) | AVL MP 4-Cores | Atraso AVL MP % |")
        log.append(f"  |----------|-----------|----------------|-----------------|-----------|----------------|-----------------|")
        for m in scenario_dict["data"]:
            
            # Evitar divisão nula matemática se a AVL ST for infinitamente rápida
            bst_ratio = ((m['bst_mp'] - m['bst_st']) / m['bst_st'] * 100) if m['bst_st'] > 0 else 0
            avl_ratio = ((m['avl_mp'] - m['avl_st']) / m['avl_st'] * 100) if m['avl_st'] > 0 else 0
            
            log.append(f"  | {m['N']:<8} | {m['bst_st']:<9.4f} | {m['bst_mp']:<14.4f} | +{bst_ratio:<14.1f} | {m['avl_st']:<9.4f} | {m['avl_mp']:<14.4f} | +{avl_ratio:<14.1f} |")

    full_text = "\n".join(log)
    print(full_text)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)


if __name__ == '__main__':
    multiprocessing.freeze_support() 
    
    base_dir = r"C:\Users\Cândido Moreira\Dropbox\Mestrado Profissional\Matérias\Obrigatórias\Algoritmos e Programação\Projeto 02"
    csv_file = os.path.join(base_dir, "wb_commodity_price_intelligence_1960_2026.csv")
    
    output_dir_v5 = os.path.join(base_dir, "relatorios", "v5")
    os.makedirs(output_dir_v5, exist_ok=True)
    os.makedirs(os.path.join(output_dir_v5, "imagens"), exist_ok=True)

    # 1. Setup
    setup_high_priority()
    hw_info = print_v5_hardware_info()

    # 2. Extract Data
    records = parse_csv(csv_file)
    
    ordered_records = records.copy()
    reversed_records = records.copy()
    reversed_records.reverse()
    shuffled_records = records.copy()
    random.seed(99)
    random.shuffle(shuffled_records)

    # 3. Benchmark V5 Pools
    chunk_scales = [10000, 20000, 30000, 40000, len(records)]

    curve_results = []
    curve_results.append(benchmark_v5_overhead("Ordered (Crescente)", ordered_records, chunk_scales))
    curve_results.append(benchmark_v5_overhead("Reversed (Decrescente)", reversed_records, chunk_scales))
    curve_results.append(benchmark_v5_overhead("Shuffled (Aleatório)", shuffled_records, chunk_scales))

    # 4. Compilação
    generate_v5_plots(curve_results, os.path.join(output_dir_v5, "imagens"))
    generate_v5_text_report(hw_info, curve_results, os.path.join(output_dir_v5, "resultados_v5.txt"))

    print("\nBenchmark V5 de IPC Paralelismo Concluído! Relatórios rigorosos gerados.")
