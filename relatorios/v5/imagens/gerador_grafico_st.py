import os
import re
import matplotlib.pyplot as plt

# Configurações de Estética para Relatório de Mestrado
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
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

def generate_unified_st_plot(scenarios, output_filepath):
    plt.figure(figsize=(12, 7))
    
    colors = {
        'Ordered (Crescente)': 'blue',
        'Reversed (Decrescente)': 'green',
        'Shuffled (Aleatório)': 'red'
    }

    for name, data in scenarios.items():
        color = colors.get(name, 'black')
        N = [d['N'] for d in data]
        
        bst_st = [d['bst_st'] for d in data]
        avl_st = [d['avl_st'] for d in data]

        # Linha contínua para BST, tracejada para AVL
        plt.plot(N, bst_st, color=color, linestyle='-', marker='o', 
                 label=f'BST - {name}')
        plt.plot(N, avl_st, color=color, linestyle='--', marker='s', 
                 label=f'AVL - {name}')

    plt.title("V5 Desempenho Single-Thread (ST) - Teste Consolidado (BST e AVL)", fontsize=16)
    plt.xlabel("Volume de Dados (N)", fontsize=12)
    plt.ylabel("Tempo Total (Segundos) - Escala Log", fontsize=12)
    plt.yscale('log')
    
    # Configurando legenda na parte externa do gráfico
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_filepath)
    plt.close()

if __name__ == '__main__':
    # Caminhos relativos
    input_file = os.path.join('..', 'resultados_v5.txt')
    output_file = 'v5_overhead_st_cenarios_unificado.png'
    
    print(f"Iniciando processamento de {input_file} para ST...")
    try:
        scenarios_data = parse_results(input_file)
        if not scenarios_data:
            print("Erro: Nenhum dado encontrado no arquivo de resultados.")
        else:
            generate_unified_st_plot(scenarios_data, output_file)
            print(f"Gráfico ST unificado gerado em: {output_file}")
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
