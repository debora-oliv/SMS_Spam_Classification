import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import platform
import psutil
from prettytable import PrettyTable


def print_env_info():
    """Imprime as informações do hardware e sistema."""
    print("="*40)
    print("      AMBIENTE DE EXECUÇÃO")
    print("="*40)
    print(f"Sistema Operacional: {platform.system()}")
    print(f"Processador: {platform.processor()}")
    print(f"Memória RAM Total: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    print(f"Versão do Python: {sys.version.split()[0]}")
    print(f"Versão do NumPy: {np.__version__}\n")


def run_model_by_dataset(model, n_experiments, x_tr, y_tr, x_te, y_te, iters=50, lr=0.01):
    """Função que roda os experimentos N vezes e extrai as métricas."""
    times = []
    train_accs = []
    test_accs = []
    model_results = None

    for i in range(n_experiments):
        print(f"          Experimento {i+1}/{n_experiments}")
        # Treinamento
        d = model(x_tr, y_tr, x_te, y_te, num_iterations=iters, learning_rate=lr, print_cost=False)

        # Salvar métricas
        times.append(d['total_optimization_time'])
        train_accs.append(d['train_acc'])
        test_accs.append(d['test_acc'])
        print("-" * 35)

        model_results = d

    return times, train_accs, test_accs, model_results['costs'][-1]


def run_all_experiments(v_model, nonv_model, X_train, Y_train, X_test, Y_test):
    """Orquestra a comparação principal (Speedup, Tabelas e Gráficos de Barra)."""
    num_experiments = 5 
    fixed_learning_rate = 0.01
    fixed_num_iterations = 50 # Define quantas épocas o algoritmo percorrerá por experimento

    print(f"\nRodando {num_experiments} experimentos (VETORIZADA)...")
    v_times, v_train_acc, v_test_acc, v_cost = run_model_by_dataset(
        v_model, num_experiments, X_train, Y_train, X_test, Y_test, iters=fixed_num_iterations, lr=fixed_learning_rate
    )

    print(f"\nRodando {num_experiments} experimentos (NÃO-VETORIZADA)...")
    nonv_times, nonv_train_acc, nonv_test_acc, nonv_cost = run_model_by_dataset(
        nonv_model, num_experiments, X_train, Y_train, X_test, Y_test, iters=fixed_num_iterations, lr=fixed_learning_rate
    )

    # --- RESULTADOS ---
    mean_v, std_v = np.mean(v_times), np.std(v_times)
    mean_nonv, std_nonv = np.mean(nonv_times), np.std(nonv_times)
    speedup = mean_nonv / mean_v
    speedups_individuais = [nonv_times[i] / v_times[i] for i in range(len(v_times))]

    print("\n" +"="* 35)
    print("      Resumo dos Experimentos      ")
    print("=" * 35)

    print("\nIMPLEMENTAÇÃO VETORIZADA:")
    print("-" * 35)
    print(f"Tempo Médio de Otimização: {mean_v:.4f} segundos")
    print(f"Tempo Desvio Padrão: {std_v:.4f} segundos")
    print(f"Acurácia Média de Treino: {np.mean(v_train_acc):.2f} %")
    print(f"Acurácia Média de Teste: {np.mean(v_test_acc):.2f} %")

    print("\nIMPLEMENTAÇÃO NÃO-VETORIZADA:")
    print("-" * 35)
    print(f"Tempo Médio de Otimização: {mean_nonv:.4f} segundos")
    print(f"Tempo Desvio Padrão: {std_nonv:.4f} segundos")
    print(f"Acurácia Média de Treino: {np.mean(nonv_train_acc):.2f} %")
    print(f"Acurácia Média de Teste: {np.mean(nonv_test_acc):.2f} %")

    print("\nDADOS SPEEDUP:")
    print("-" * 35)
    print(f"Speedup Médio: {speedup:.2f}x")
    print(f"Desvio padrão dos Speedups: ± {np.std(speedups_individuais):.2f}x")

    # --- TABELA COMPARATIVA ---
    tabela = PrettyTable()
    tabela.field_names = ["Implementação", "Tempo Médio (s)", "Desvio Padrão (±)", "Speedup"]
    tabela.add_row(["Não-Vetorizada (Loops)", f"{mean_nonv:.4f}", f"{std_nonv:.4f}", "1.00x (Base)"])
    tabela.add_row(["Vetorizada (NumPy)", f"{mean_v:.4f}", f"{std_v:.4f}", f"{speedup:.2f}x"])
    tabela.align["Implementação"] = "l"
    tabela.padding_width = 1

    print("\n" + "="*70)
    print("Tabela - Análise Comparativa de Desempenho")
    print(tabela)

    # --- GRÁFICOS DE BARRAS ---
    print("\nGerando Gráfico de Barras Comparativo...")
    experimentos = [f'Exp {i+1}' for i in range(num_experiments)]
    x = np.arange(len(experimentos))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, nonv_times, width, label='Não-Vetorizada', color='coral')
    plt.bar(x + width/2, v_times, width, label='Vetorizada', color='skyblue')

    plt.ylabel('Tempo de Execução (segundos)')
    plt.title('Comparação de Performance: 5 Experimentos', weight='bold')
    plt.xticks(x, experimentos)
    plt.legend()

    plt.annotate(f'Speedup Médio: {speedup:.1f}x',
                 xy=(0.5, 0.2), xycoords='axes fraction',
                 ha='center', fontsize=12, fontweight='bold', color='darkred',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="darkred", lw=2))

    for i in range(len(v_times)):
        plt.text(x[i], nonv_times[i] + (max(nonv_times) * 0.01), 
                 f'{speedups_individuais[i]:.1f}x', 
                 ha='center', va='bottom', fontsize=10, fontweight='bold', color='darkred')

    plt.tight_layout()
    plt.savefig('grafico_barras_comparativo.png')
    plt.close()


def run_speedup_vs_samples(v_model, nonv_model, X_train_processed, y_train_processed, X_test_processed, y_test_processed):
    """Executa a medição de tempo em função do tamanho da amostra (m) e gera o gráfico de linha."""
    tamanhos_m = [100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4457]
    speedup_outcomes = []
    num_experiments_spdup = 2

    print(f"\n\n=======================================================")
    print(f"MEDIR TEMPO DE EXECUÇÃO EM FUNÇÃO DO TAMANHO DA AMOSTRA")
    print(f"=======================================================\n")

    for m in tamanhos_m:
        X_fatiado = X_train_processed[:, :m]
        Y_fatiado = y_train_processed[:, :m]
        
        print(f"Tamanho: {m} amostras")
        
        # Teste Iterativa
        t0 = time.time()
        run_model_by_dataset(nonv_model, num_experiments_spdup, X_fatiado, Y_fatiado, X_test_processed, y_test_processed)
        tempo_nonv = time.time() - t0

        # Teste Vetorizada
        t1 = time.time()
        run_model_by_dataset(v_model, num_experiments_spdup, X_fatiado, Y_fatiado, X_test_processed, y_test_processed)
        tempo_v = time.time() - t1

        speedup_outcomes.append(tempo_nonv / tempo_v)
        print(f"   -> Speedup registrado: {tempo_nonv / tempo_v:.2f}x\n")

    # --- GRÁFICO DE LINHA ---
    plt.figure(figsize=(12, 6))
    plt.plot(tamanhos_m, speedup_outcomes, marker='o', linestyle='-', color='darkgreen', linewidth=2)
    plt.title('Evolução do Speedup vs. Tamanho das Amostras (m)', fontsize=14, weight='bold')
    plt.xlabel('Número de Amostras (m)', fontsize=12)
    plt.ylabel('Speedup (x vezes mais rápido)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)

    for i, txt in enumerate(speedup_outcomes):
        plt.annotate(f"{txt:.1f}x", (tamanhos_m[i], speedup_outcomes[i]), 
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('evolucao_speedup_amostras.png')
    plt.close()
    print("Gráficos salvos com sucesso na raiz do projeto!")
