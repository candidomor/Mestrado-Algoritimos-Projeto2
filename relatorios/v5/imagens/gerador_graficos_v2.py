import os
import re
import matplotlib.pyplot as plt
import numpy as np

# Configurações de Estética para Relatório de Mestrado
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

def parse_results(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    scenarios = {}
    # Regex para capturar o nome do cenário e a tabela correspondente
    pattern = r'>>>> CENÁRIO: (.*?) <<<<\n\s*\| Vol \(N\).*?\|\n\s*\|-.*?\|\n(.*?)(?=\n\n|\n>|====|$)'
    matches = re.finditer(pattern, content, re.DOTALL)

    for match in matches:
        scenario_name = match.group(1).strip()
        table_data = match.group(2).strip().split('\n')
        
        data = []
        for line in table_data:
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) >= 7:
                data.append({
                    'N': int(parts[0]),
                    'bst_st': float(parts[1]),
                    'bst_mp': float(parts[2]),
                    'avl_st': float(parts[4]),
                    'avl_mp': float(parts[5])
                })
        scenarios[scenario_name] = data
    
    return scenarios

def calculate_heights(N, scenario):
    """
    Calcula alturas teóricas/estimadas para os cenários.
    AVL: ~1.44 * log2(N)
    BST Ordered/Reversed: N (Degenerada em Lista)
    BST Shuffled: ~2 * ln(N) (Média em dados aleatórios)
    """
    avl_h = 1.44 * np.log2(N)
    if 'Ordered' in scenario or 'Reversed' in scenario:
        bst_h = N
    else:
        bst_h = 2 * np.log(N)
    return bst_h, avl_h

def generate_plots(scenarios, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Comparativo de Performance ST (BST vs AVL)
    plt.figure(figsize=(12, 7))
    styles = {'Ordered (Crescente)': ('-', 'o'), 'Reversed (Decrescente)': ('--', 's'), 'Shuffled (Aleatório)': (':', '^')}
    
    for name, data in scenarios.items():
        ls, mk = styles.get(name, ('-', 'o'))
        N = [d['N'] for d in data]
        bst = [d['bst_st'] for d in data]
        avl = [d['avl_st'] for d in data]
        
        plt.plot(N, bst, label=f'BST - {name}', color='red', linestyle=ls, marker=mk, alpha=0.7)
        plt.plot(N, avl, label=f'AVL - {name}', color='green', linestyle=ls, marker=mk, alpha=0.9)

    plt.title('Comparativo de Latência de Inserção: BST vs AVL (Single-Thread)')
    plt.xlabel('Volume de Dados (N)')
    plt.ylabel('Tempo de Execução (Segundos)')
    plt.yscale('log') # Escala Log para ver a diferença abissal
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'v2_comparativo_st_bst_vs_avl.png'))
    plt.close()

    # 2. Análise de Speedup (Eficiência do Paralelismo)
    plt.figure(figsize=(10, 6))
    for name, data in scenarios.items():
        N = [d['N'] for d in data]
        # Speedup = ST / MP. Se > 1, há ganho. Se < 1, há perda (Overhead).
        speedup_bst = [d['bst_st'] / d['bst_mp'] for d in data]
        speedup_avl = [d['avl_st'] / d['avl_mp'] for d in data]
        
        plt.plot(N, speedup_bst, label=f'Speedup BST ({name})', marker='o', alpha=0.8)
        plt.plot(N, speedup_avl, label=f'Speedup AVL ({name})', marker='s', alpha=0.8, linestyle='--')

    plt.axhline(y=1, color='black', linestyle='-', linewidth=1.5, alpha=0.5, label='Limite de Eficiência (ST=MP)')
    plt.fill_between([min(N), max(N)], 0, 1, color='red', alpha=0.1, label='Região de Overhead (Perda)')
    plt.fill_between([min(N), max(N)], 1, 5, color='green', alpha=0.1, label='Região de Ganho (Escalabilidade)')
    
    plt.title('Análise de Speedup: Eficiência do Multiprocessing (4 Cores)')
    plt.xlabel('Volume de Dados (N)')
    plt.ylabel('Fator de Aceleração (ST / MP)')
    plt.legend(loc='upper left', fontsize='x-small', ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'v2_analise_speedup_paralelo.png'))
    plt.close()

    # 3. Complexidade Estrutural (Altura Estimada das Árvores)
    plt.figure(figsize=(10, 6))
    for name, data in scenarios.items():
        N = [d['N'] for d in data]
        heights = [calculate_heights(d['N'], name) for d in data]
        bst_h = [h[0] for h in heights]
        avl_h = [h[1] for h in heights]
        
        plt.plot(N, bst_h, label=f'Altura BST ({name})', marker='o')
        plt.plot(N, avl_h, label=f'Altura AVL ({name})', marker='x', linestyle='--')

    plt.title('Evolução da Altura da Árvore (Complexidade Estrutural)')
    plt.xlabel('Número de Nós (N)')
    plt.ylabel('Altura / Profundidade da Árvore')
    plt.yscale('log')
    plt.legend(loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'v2_complexidade_altura_estimada.png'))
    plt.close()

    # 4. Custo Unitário de Inserção (Escalabilidade)
    plt.figure(figsize=(10, 6))
    for name, data in scenarios.items():
        N = [d['N'] for d in data]
        # Tempo por nó em microsegundos
        cost_bst = [(d['bst_st'] / d['N']) * 1e6 for d in data]
        cost_avl = [(d['avl_st'] / d['N']) * 1e6 for d in data]
        
        plt.plot(N, cost_bst, label=f'Custo BST ({name})', marker='o')
        plt.plot(N, cost_avl, label=f'Custo AVL ({name})', marker='s')

    plt.title('Custo Unitário de Inserção (Eficiência Algorítmica)')
    plt.xlabel('Volume de Dados (N)')
    plt.ylabel('Tempo por Nó (µs)')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'v2_custo_unitario_insercao.png'))
    plt.close()

if __name__ == '__main__':
    # Caminhos relativos
    input_file = os.path.join('..', 'resultados_v5.txt')
    output_path = 'graficos_v2'
    
    print(f"Iniciando processamento de {input_file}...")
    try:
        scenarios_data = parse_results(input_file)
        if not scenarios_data:
            print("Erro: Nenhum dado encontrado no arquivo de resultados.")
        else:
            print(f"Dados extraídos para {len(scenarios_data)} cenários.")
            generate_plots(scenarios_data, output_path)
            print(f"Gráficos gerados com sucesso em {output_path}/")
    except Exception as e:
        print(f"Ocorreu um erro durante a geração dos gráficos: {e}")
