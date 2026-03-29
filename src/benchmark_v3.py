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

# Ajuste do limite de recursão elevado para acomodar a estrutura linear de BST pior caso em 49k itens
sys.setrecursionlimit(200000)

from trees import BST, AVLTree, CommodityRecord

def setup_high_priority():
    """
    Tenta elevar a prioridade do processo Python no nível do Sistema Operacional (Windows).
    Isso minimiza o 'Thread Starvation' causado por outros programas rodando no fundo,
    garantindo que os ticks do processador sejam dedicados ao benchmark.
    """
    try:
        p = psutil.Process(os.getpid())
        if platform.system() == 'Windows':
            # Eleva para HIGH_PRIORITY_CLASS (REALTIME é perigoso e pode travar mouses/teclados)
            p.nice(psutil.HIGH_PRIORITY_CLASS)
            print("[OS] Prioridade do Processo elevada para HIGH_PRIORITY via psutil.")
        else:
            # Unix-like (-20 é a maior prioridade viável, requer sudo normalmente)
            p.nice(-10) 
            print("[OS] Prioridade do Processo ajustada via psutil/nice.")
    except Exception as e:
        print(f"[OS Warnings] Não foi possível elevar a prioridade do processo: {e}")


def print_hardware_info() -> str:
    def get_size(bytes, suffix="B"):
        factor = 1024
        for unit in ["", "K", "M", "G", "T", "P"]:
            if bytes < factor:
                return f"{bytes:.2f}{unit}{suffix}"
            bytes /= factor

    svmem = psutil.virtual_memory()
    
    info = (
        f"----- HARDWARE PROFILING (V3) -----\n"
        f"OS System: {platform.system()} {platform.release()} ({platform.version()})\n"
        f"Processor: {platform.processor()}\n"
        f"Physical Cores: {psutil.cpu_count(logical=False)}\n"
        f"Total Logical Cores: {psutil.cpu_count(logical=True)}\n"
        f"Total RAM: {get_size(svmem.total)}\n"
        f"Available RAM: {get_size(svmem.available)}\n"
        f"Execution Priority: Elevada (High Priority)\n"
        f"Python Version: {platform.python_version()}\n"
        f"-----------------------------------\n"
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
    Roda um miniteste silencioso para forçar o Sistema Operacional e
    a RAM a alocarem as páginas necessárias e aquecerem o Cache L1/L2 do Processador.
    """
    print(">> Executando Fase de Warm-up (Aquecimento de Cache/JIT)...")
    bst = BST()
    avl = AVLTree()
    for rec in records_sample:
        bst.insert(rec)
        avl.insert(rec)
    del bst
    del avl
    gc.collect()
    print(">> Warm-up concluído.")


def evaluate_batch(tree_instance, data_chunk: List[CommodityRecord]) -> float:
    """Mede APENAS O TEMPO com precisão de nanosegundos (perf_counter)."""
    gc.collect()
    # Usa perf_counter (CPU Timer monotonico real), em vez de time() (Relogio fragil do Windows/NTP)
    start = time.perf_counter()
    for rec in data_chunk:
        tree_instance.insert(rec)
    end = time.perf_counter()
    return end - start


def asymptotic_benchmark(scenario_name: str, records: List[CommodityRecord], chunk_sizes: List[int], repetitions: int = 3):
    """
    Executa testes parciais crescendo N para analisar a curva O(N).
    Retorna uma lista de tuplas com os tempos médios: [(N, bst_time, avl_time), ...]
    """
    print(f"\n[Curva Assintótica] -> Cenário: {scenario_name}")
    results = []

    for n_size in chunk_sizes:
        data_chunk = records[:n_size]
        bst_times = []
        avl_times = []

        print(f"  Avaliando lote com N = {n_size} ...")

        for _ in range(repetitions):
            bst = BST()
            avl = AVLTree()
            
            bst_t = evaluate_batch(bst, data_chunk)
            avl_t = evaluate_batch(avl, data_chunk)

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


def _worker_process_insertion(records_chunk: List[CommodityRecord], tree_type: str) -> float:
    """Função alvo para rodar isolada em um Core Lógico nativo via Multiprocess"""
    tree = BST() if tree_type == 'BST' else AVLTree()
    start = time.perf_counter()
    for rec in records_chunk:
        tree.insert(rec)
    end = time.perf_counter()
    return end - start


def parallel_vs_single_benchmark(records: List[CommodityRecord]):
    """
    Compara o ato de Inserir N records em 1 Única Árvore (Single-Thread)
    com o ato de Inserir os mesmos N records, mas divididos em 4 Árvores Rodando em Paralelo.
    """
    print("\n[Threading & CPU Overhead] -> Analisando Carga Single-Thread vs Multi-Process Pool")
    
    # Vamos pegar uma fatia menor (Ex: Aleatória 20k) para esse teste para não travar os processos
    random.seed(42)
    sample = random.sample(records, 20000)

    # 1. TESTE SINGLE THREAD
    t_start = time.perf_counter()
    bst = BST()
    for rec in sample:
        bst.insert(rec)
    t_end = time.perf_counter()
    s_thread_time = t_end - t_start
    del bst
    gc.collect()

    # 2. TESTE MULTIPROCESSED (Divide carga de 20k itens por 4 nucleos lógicos = 5k itens por Core)
    chunk1 = sample[0:5000]
    chunk2 = sample[5000:10000]
    chunk3 = sample[10000:15000]
    chunk4 = sample[15000:20000]

    t_start = time.perf_counter()
    
    # Process Pool Executor bypassa o GIL instanciando ambientes Python limpos separados
    with multiprocessing.Pool(processes=4) as pool:
        # Pede para o pool de 4 Cores Executar a func. O apply_async nao bloqueia.
        results = [
            pool.apply_async(_worker_process_insertion, (chunk1, 'BST')),
            pool.apply_async(_worker_process_insertion, (chunk2, 'BST')),
            pool.apply_async(_worker_process_insertion, (chunk3, 'BST')),
            pool.apply_async(_worker_process_insertion, (chunk4, 'BST'))
        ]
        # Espera as amarras e captura resultados
        times = [r.get() for r in results]

    t_end = time.perf_counter()
    m_thread_total_time = t_end - t_start # Wall-Clock Total real
    
    print(f"  -> Single-Thread Wall-Clock (20k Inserções Sequenciais na BST): {s_thread_time:.4f}s")
    print(f"  -> Multi-Process Wall-Clock (4 Processos rodando 5k simultâneos): {m_thread_total_time:.4f}s")
    
    return {
        "single_thread": s_thread_time,
        "multi_process": m_thread_total_time,
        "process_count": 4,
        "total_records": 20000
    }


def generate_v3_plots(all_curves: List[Dict], output_dir: str):
    """
    Plota as curvas assintoticas matadoras provando o aumento de tempo em O(N).
    Para 3 cenarios (Ordered, Reversed, Shuffled), mostramos BST e AVL.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for scenario_dict in all_curves:
        scenario = scenario_dict["scenario"]
        metrics = scenario_dict["data"]
        
        N_vals = [m["N"] for m in metrics]
        bst_means = [m["bst_mean"] for m in metrics]
        avl_means = [m["avl_mean"] for m in metrics]

        plt.figure(figsize=(10, 6))
        # BST Curve Plotting
        plt.plot(N_vals, bst_means, marker='o', label='BST', color='blue', linewidth=2)
        
        # AVL Curve Plotting
        plt.plot(N_vals, avl_means, marker='s', label='AVL', color='green', linewidth=2)

        plt.title(f"Análise Curva Assintótica de Custo ($T(N)$) - {scenario}")
        plt.xlabel("Quantidade de Elementos Chave (N)")
        plt.ylabel("Latência Computacional Acumulada (Segundos)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        # Format filename to be safe
        safe_name = scenario.replace(" ", "_").replace("(", "").replace(")", "").lower()
        plt.savefig(os.path.join(output_dir, f'v3_assintotica_{safe_name}.png'))
        plt.close()


def generate_v3_text_report(hw_info: str, all_curves: List[Dict], m_thread_res: Dict, output_path: str):
    log = []
    log.append(hw_info)
    
    log.append("\n==========================================================")
    log.append("  ANÁLISE DE ESCALABILIDADE ASSINTÓTICA (Custo vs. Volume)")
    log.append("==========================================================")

    for scenario_dict in all_curves:
        log.append(f"\n>>>> CENÁRIO: {scenario_dict['scenario']} <<<<")
        log.append(f"  | Vol (N)  |  BST Médio (s)  |  AVL Médio (s)  |")
        log.append(f"  |----------|-----------------|-----------------|")
        for m in scenario_dict["data"]:
            log.append(f"  | {m['N']:<8} |  {m['bst_mean']:<13.4f}  |  {m['avl_mean']:<13.4f}  |")

    log.append("\n==========================================================")
    log.append("  ISOLAMENTO PROCESSUAL (Multiprocessing VS Single-Thread)")
    log.append("==========================================================")
    log.append(f" Teste: Dividir sub-árvores vs Construir Árvore Única ({m_thread_res['total_records']} itens Aleatórios)")
    log.append(f" -> Wall-Clock Single-Thread: {m_thread_res['single_thread']:.4f}s")
    log.append(f" -> Wall-Clock Multi-Process ({m_thread_res['process_count']} Nucléos Isolados): {m_thread_res['multi_process']:.4f}s")
    
    gain = m_thread_res['single_thread'] - m_thread_res['multi_process']
    percent_gain = (gain / m_thread_res['single_thread']) * 100
    log.append(f" O Multi-Processing economizou no tempo global real: {percent_gain:.1f}%")

    full_text = "\n".join(log)
    print(full_text)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)


if __name__ == '__main__':
    # Este if __name__ == '__main__' e MUST-HAVE para python multiprocessing funcionar nativamente no Windows sem loops zumbis!
    multiprocessing.freeze_support() 
    
    base_dir = r"C:\Users\Cândido Moreira\Dropbox\Mestrado Profissional\Matérias\Obrigatórias\Algoritmos e Programação\Projeto 02"
    csv_file = os.path.join(base_dir, "wb_commodity_price_intelligence_1960_2026.csv")
    
    # Criar pasta para nova geracao de outputs V3
    output_dir_v3 = os.path.join(base_dir, "relatorios", "v3")
    os.makedirs(output_dir_v3, exist_ok=True)
    os.makedirs(os.path.join(output_dir_v3, "imagens"), exist_ok=True)

    # Iniciar Proteções de Processo SO
    setup_high_priority()
    hw_info = print_hardware_info()

    # Leitura Base
    records = parse_csv(csv_file)
    
    # Gerar os 3 Datasets
    ordered_records = records.copy()
    
    reversed_records = records.copy()
    reversed_records.reverse()
    
    shuffled_records = records.copy()
    random.seed(99)
    random.shuffle(shuffled_records)

    # 1. Warm-up
    run_warmup(shuffled_records[:1000])

    # 2. Benchmark Assintotico em Partes
    chunk_scales = [10000, 20000, 30000, 40000, len(records)]
    # Usando apenas 3 repetições pra não demorar mais de 30-40 minutos (a BST degradada a 49k é O(N^2) pesado)
    reps = 3 

    curve_results = []
    curve_results.append(asymptotic_benchmark("Ordered (Crescente)", ordered_records, chunk_scales, reps))
    curve_results.append(asymptotic_benchmark("Reversed (Decrescente)", reversed_records, chunk_scales, reps))
    curve_results.append(asymptotic_benchmark("Shuffled (Aleatório)", shuffled_records, chunk_scales, reps))

    # 3. Benchmark de Carga GIL VS Multiprocessing
    mp_results = parallel_vs_single_benchmark(records)

    # 4. Compilação
    generate_v3_plots(curve_results, os.path.join(output_dir_v3, "imagens"))
    generate_v3_text_report(hw_info, curve_results, mp_results, os.path.join(output_dir_v3, "resultados_v3.txt"))

    print("\nBenchmark V3 O(N) Concluído! Relatórios rigorosos encontram-se na pasta relatorios/v3")
