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
import matplotlib.pyplot as plt

# Usar tipagem para melhorar a legibilidade e manutenibilidade
from typing import List, Dict, Any

# Ajuste do limite de recursão elevado para acomodar a estrutura linear de BST pior caso 
sys.setrecursionlimit(100000)

from trees import BST, AVLTree, CommodityRecord


def print_hardware_info() -> str:
    """
    Coleta os metadados valiosos das configurações do computador/ambiente.
    Útil para justificar estatisticamente as limitações físicas do benchmark.
    """
    # Converter bytes para gigabytes
    def get_size(bytes, suffix="B"):
        factor = 1024
        for unit in ["", "K", "M", "G", "T", "P"]:
            if bytes < factor:
                return f"{bytes:.2f}{unit}{suffix}"
            bytes /= factor

    svmem = psutil.virtual_memory()
    
    info = (
        f"----- HARDWARE PROFILING -----\n"
        f"OS System: {platform.system()} {platform.release()} ({platform.version()})\n"
        f"Processor: {platform.processor()}\n"
        f"Physical Cores: {psutil.cpu_count(logical=False)}\n"
        f"Total Logical Cores: {psutil.cpu_count(logical=True)}\n"
        f"Total RAM: {get_size(svmem.total)}\n"
        f"Available RAM: {get_size(svmem.available)}\n"
        f"Python Version: {platform.python_version()}\n"
        f"-------------------------------\n"
    )
    print(info)
    return info


def parse_csv(filepath: str) -> List[CommodityRecord]:
    """Lê o arquivo CSV ordenado e cria os objetos iniciais de Commodities"""
    records = []
    with open(filepath, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row['commodity_code']}_{row['date']}"
            records.append(CommodityRecord(key, row))
    return records


def run_single_test(tree_instance, records: List[CommodityRecord], sample_keys: List[str]) -> Dict[str, Any]:
    """
    Executa a ingestão e busca em uma única instância de árvore gerada, e 
    reporta isoladamente as estatísticas base.
    """
    # --- Rastreio de Limite de Memória (Rigor) ---
    gc.collect() # Invocar Coletor de Lixo do Python p/ zerar vestígios do teste passado
    tracemalloc.start() 

    # Inserção
    start_time = time.time()
    for rec in records:
        tree_instance.insert(rec)
    insert_time = time.time() - start_time
    
    # Extração de Memória Ocupada e parada do tracker
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024) # Conversão pra MB

    # Altura
    if isinstance(tree_instance, BST):
        height = tree_instance.get_height_iterative()
    else:
        height = tree_instance.get_height(tree_instance.root)

    # Busca (A busca não aloca grandes nós persistentes, não mediremos RAM aqui).
    start_time = time.time()
    for key in sample_keys:
        tree_instance.search(key)
    search_time = time.time() - start_time

    return {
        "insert_time": insert_time,
        "search_time": search_time,
        "peak_ram_mb": peak_mb,
        "height": height
    }


def benchmark_scenario(scenario_name: str, records: List[CommodityRecord], target_keys: List[str], repetitions=5):
    """
    Roda um laboratório repetitivo do cenário atual (Ordered, Shuffled ou Reversed)
    calculando Média e Desvio Padrão para Inserção, Busca e RAM.
    """
    print(f"\n>>>> Executando Cenário: {scenario_name} ({repetitions} repetições) <<<<")
    
    bst_metrics = {"insert": [], "search": [], "ram": [], "heights": []}
    avl_metrics = {"insert": [], "search": [], "ram": [], "heights": []}

    for i in range(repetitions):
        print(f"  -> Rodada {i+1}/{repetitions}")
        
        # Rigor absoluto: Criamos novas árvores do completo zero a cada iteração
        bst = BST()
        avl = AVLTree()

        # Medidas da BST
        bst_res = run_single_test(bst, records, target_keys)
        bst_metrics["insert"].append(bst_res["insert_time"])
        bst_metrics["search"].append(bst_res["search_time"])
        bst_metrics["ram"].append(bst_res["peak_ram_mb"])
        bst_metrics["heights"].append(bst_res["height"])

        # Medidas da AVL
        avl_res = run_single_test(avl, records, target_keys)
        avl_metrics["insert"].append(avl_res["insert_time"])
        avl_metrics["search"].append(avl_res["search_time"])
        avl_metrics["ram"].append(avl_res["peak_ram_mb"])
        avl_metrics["heights"].append(avl_res["height"])
        
        # Liberação massiva de memória explícita - Limos referências pesadas
        del bst
        del avl
        gc.collect() 

    # Agregação Média e Desvio Padrão
    def calc_stats(metric_list):
        mean = statistics.mean(metric_list)
        stdev = statistics.stdev(metric_list) if len(metric_list) > 1 else 0
        return mean, stdev

    # Altura na AVL e BST variam de forma diferente
    # AVL será constante baseada na qtd. BST sofrerá dependendo da ordem.
    bst_h_mean, _ = calc_stats(bst_metrics["heights"])
    avl_h_mean, _ = calc_stats(avl_metrics["heights"])

    bst_ins_mean, bst_ins_std = calc_stats(bst_metrics["insert"])
    avl_ins_mean, avl_ins_std = calc_stats(avl_metrics["insert"])

    bst_sch_mean, bst_sch_std = calc_stats(bst_metrics["search"])
    avl_sch_mean, avl_sch_std = calc_stats(avl_metrics["search"])

    bst_ram_mean, bst_ram_std = calc_stats(bst_metrics["ram"])
    avl_ram_mean, avl_ram_std = calc_stats(avl_metrics["ram"])

    return {
        "scenario": scenario_name,
        "n_records": len(records),
        "n_searches": len(target_keys),
        "bst": {
            "h_mean": bst_h_mean,
            "ins": (bst_ins_mean, bst_ins_std),
            "search": (bst_sch_mean, bst_sch_std),
            "ram": (bst_ram_mean, bst_ram_std),
        },
        "avl": {
            "h_mean": avl_h_mean,
            "ins": (avl_ins_mean, avl_ins_std),
            "search": (avl_sch_mean, avl_sch_std),
            "ram": (avl_ram_mean, avl_ram_std),
        }
    }


def full_benchmark_suite(records: List[CommodityRecord], repetitions: int = 5):
    """Prepara os três datasets necessários e encaminha pro laço."""
    
    # Gerando os alvos de busca - Iguais em todos cenarios
    random.seed(42)
    sample_records = random.sample(records, min(10000, len(records)))
    sample_keys = [r.key for r in sample_records]
    sample_keys.extend([f"NON_EXISTENT_KEY_{i}" for i in range(1000)])
    random.shuffle(sample_keys)

    # Criando clones dos datasets (gastam memoria RAM duplicando as referencias para as structs)
    print("Preparando Dataframes Múltiplos...")
    
    ordered_records = records.copy()
    
    reversed_records = records.copy()
    reversed_records.reverse()
    
    shuffled_records = records.copy()
    random.seed(99) # garantir que o embaralhamento é sempre o mesmo se rodar novamente
    random.shuffle(shuffled_records)

    # Array de relatórios brutos
    final_stats = []

    # Executa as 3 ordens
    final_stats.append(benchmark_scenario("Ordered (Crescente)", ordered_records, sample_keys, repetitions))
    final_stats.append(benchmark_scenario("Shuffled (Aleatório)", shuffled_records, sample_keys, repetitions))
    final_stats.append(benchmark_scenario("Reversed (Decrescente)", reversed_records, sample_keys, repetitions))

    return final_stats


def plot_v2_results(stats: List[Dict], output_dir: str):
    """
    Geração gráfica robusta com matplotlib que compara Inserção, Busca e Memória.
    Comporta "Error Bars" representando o desvio padrão de segurança da estatistica.
    """
    os.makedirs(output_dir, exist_ok=True)
    scenarios = [s["scenario"] for s in stats]
    x_positions = range(len(scenarios))
    
    # Extrair Métricas pra formato Chart Compatível
    def extract(tree_type, m_key):
        means = [s[tree_type][m_key][0] for s in stats]
        stds  = [s[tree_type][m_key][1] for s in stats]
        return means, stds

    # INSERÇÃO ---------------------------------------------------------------------
    bst_ins_means, bst_ins_stds = extract("bst", "ins")
    avl_ins_means, avl_ins_stds = extract("avl", "ins")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([x - 0.2 for x in x_positions], bst_ins_means, 0.4, yerr=bst_ins_stds, label='BST', color='blue', capsize=5)
    ax.bar([x + 0.2 for x in x_positions], avl_ins_means, 0.4, yerr=avl_ins_stds, label='AVL', color='green', capsize=5)
    
    ax.set_ylabel('Tempo de Inserção (segundos)')
    ax.set_title('Média de Tempo de Inserção por Cenário (+ Desvio Padrão)')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(scenarios)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'v2_tempo_insercao.png'))
    plt.close()

    # BUSCA ------------------------------------------------------------------------
    bst_sch_means, bst_sch_stds = extract("bst", "search")
    avl_sch_means, avl_sch_stds = extract("avl", "search")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([x - 0.2 for x in x_positions], bst_sch_means, 0.4, yerr=bst_sch_stds, label='BST', color='blue', capsize=5)
    ax.bar([x + 0.2 for x in x_positions], avl_sch_means, 0.4, yerr=avl_sch_stds, label='AVL', color='green', capsize=5)
    
    # Observação: em Ordered ou Reversed, a BST pode chegar a 3-5 Segundos, 
    # engolindo visualmente a barrinha da AVL no Linear Scale. Isso é perfeitamente correto.
    ax.set_ylabel('Tempo de Busca (segundos)') 
    ax.set_title(f"Média de Tempo de Busca por Cenário (+ Desvio Padrão)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(scenarios)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'v2_tempo_busca.png'))
    plt.close()

    # PICO DE MEMÓRIA (RAM) -------------------------------------------------------
    bst_ram_means, bst_ram_stds = extract("bst", "ram")
    avl_ram_means, avl_ram_stds = extract("avl", "ram")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([x - 0.2 for x in x_positions], bst_ram_means, 0.4, yerr=bst_ram_stds, label='BST', color='maroon', capsize=5)
    ax.bar([x + 0.2 for x in x_positions], avl_ram_means, 0.4, yerr=avl_ram_stds, label='AVL', color='darkorange', capsize=5)
    
    ax.set_ylabel('Consumo de RAM no Pico (MB)')
    ax.set_title('Pico Numérico de Memória RAM Ocupada')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(scenarios)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'v2_memoria_ram.png'))
    plt.close()

    # ALTURA (Discreta) -----------------------------------------------------------
    # Nao precisamos de STD DEV pra altura, em arvore ordenada será sempre igual na BST (16621) 
    bst_hs = [s["bst"]["h_mean"] for s in stats]
    avl_hs = [s["avl"]["h_mean"] for s in stats]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([x - 0.2 for x in x_positions], bst_hs, 0.4, label='BST Heights', color='gray')
    ax.bar([x + 0.2 for x in x_positions], avl_hs, 0.4, label='AVL Heights', color='black')
    
    # Customizando texto das alturas
    for idx, (b, a) in enumerate(zip(bst_hs, avl_hs)):
        ax.text(idx - 0.2, b + 1, f"{int(b)}", ha='center', va='bottom', fontsize=9)
        ax.text(idx + 0.2, a + 1, f"{int(a)}", ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Altura Média da Estrutura')
    ax.set_title('Formatos Degenerados (Pior Caso) vs. Saudáveis')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(scenarios)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'v2_altura_arvores.png'))
    plt.close()


def generate_textual_report(hardware_info: str, stats: List[Dict], output_path: str):
    log = []
    log.append(hardware_info)
    
    for s in stats:
        log.append(f"====== CENÁRIO: {s['scenario']} ======")
        log.append(f" - Quantidade de Chaves Inseridas: {s['n_records']}")
        log.append(f" - Quantidade de Buscas em Lote:   {s['n_searches']}")
        
        # BST
        log.append(f"  [BST]")
        log.append(f"    - Altura Alcançada: {s['bst']['h_mean']:.0f} nós")
        log.append(f"    - Tempo de Inserção: {s['bst']['ins'][0]:.4f}s (±{s['bst']['ins'][1]:.4f}s spread)")
        log.append(f"    - Tempo de Busca:    {s['bst']['search'][0]:.4f}s (±{s['bst']['search'][1]:.4f}s spread)")
        log.append(f"    - Pico de Memória:   {s['bst']['ram'][0]:.2f}MB (±{s['bst']['ram'][1]:.2f}MB spread)")

        # AVL
        log.append(f"  [AVL]")
        log.append(f"    - Altura Alcançada: {s['avl']['h_mean']:.0f} nós")
        log.append(f"    - Tempo de Inserção: {s['avl']['ins'][0]:.4f}s (±{s['avl']['ins'][1]:.4f}s spread)")
        log.append(f"    - Tempo de Busca:    {s['avl']['search'][0]:.4f}s (±{s['avl']['search'][1]:.4f}s spread)")
        log.append(f"    - Pico de Memória:   {s['avl']['ram'][0]:.2f}MB (±{s['avl']['ram'][1]:.2f}MB spread)")
        log.append("")

    full_text = "\n".join(log)
    print(full_text)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)


if __name__ == '__main__':
    base_dir = r"C:\Users\Cândido Moreira\Dropbox\Mestrado Profissional\Matérias\Obrigatórias\Algoritmos e Programação\Projeto 02"
    csv_file = os.path.join(base_dir, "wb_commodity_price_intelligence_1960_2026.csv")
    
    # Criar pasta para nova geracao de outputs V2
    output_dir_v2 = os.path.join(base_dir, "relatorios", "v2")
    os.makedirs(output_dir_v2, exist_ok=True)
    os.makedirs(os.path.join(output_dir_v2, "imagens"), exist_ok=True)

    # 1. Hardware Report
    hw_info = print_hardware_info()

    # 2. Extract Data
    print(f"Lendo base CSV brutos ({csv_file})...")
    records = parse_csv(csv_file)
    
    # 3. Disparo da Bateria de Benchmarks Complexos com 5 Repetições de cada Caso
    stats = full_benchmark_suite(records, repetitions=5)

    # 4. Geração Externa
    print("\nSalvando Gráficos V2 com matplotlib...")
    plot_v2_results(stats, os.path.join(output_dir_v2, "imagens"))

    print("Escrevendo Relatório Compilado...")
    textual_report_path = os.path.join(output_dir_v2, "resultados_v2.txt")
    generate_textual_report(hw_info, stats, textual_report_path)

    print(f"\nOperação V2 Múltipla Finalizada com Sucesso! Relatórios em {output_dir_v2}")