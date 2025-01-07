import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import locale
import gc
import time

# Registrar o início do script
start_time = datetime.now()
print(f"\nScript iniciado em: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

# ===== INÍCIO DO CÓDIGO =====

# Configurar o locale para o formato numérico brasileiro
locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')

# Funções para formatar números
def formatar_decimais(numero):
    numero_formatado = locale.format_string('%.2f', numero, grouping=True)
    return numero_formatado

def formatar_inteiro(numero):
    numero_formatado = locale.format_string('%.0f', int(numero), grouping=True)
    return numero_formatado

# Parâmetros para a simulação de Monte Carlo
num_simulations = 1000000  # Número elevado para teste de desempenho
num_days = 252  # Dias úteis no ano

# Definir o caminho do arquivo CSV
input_path = r"C:\Users\david\OneDrive\Python VsCode\ML NVIDIA STOCK\\"
input_file = "NVDA_1999-01-01_2024-12-04.csv"
input_final = input_path + input_file


# ===== CARREGAR DADOS =====
# Verificar se o arquivo CSV existe
if not os.path.exists(input_final):
    print(f"Arquivo CSV não encontrado no caminho: {input_final}")
    exit()

try:
    # Carregar os dados do CSV
    data = pd.read_csv(input_final)
except Exception as e:
    print(f"Erro ao carregar o arquivo CSV: {e}")
    exit()


# ===== PREPARAÇÃO DOS DADOS =====

# Validar colunas do arquivo
required_columns = ['Date', 'Close']
if not all(column in data.columns for column in required_columns):
    print(f"Erro: O arquivo não contém as colunas necessárias: {required_columns}")
    exit()

# Remover valores nulos e converter a coluna 'Date'
data = data.dropna(subset=['Close'])
data['Date'] = pd.to_datetime(data['Date'])

# Calcular retornos logarítmicos diários
data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))

# Extrair a média e o desvio padrão dos retornos
mu = data['Log_Returns'].mean()
sigma = data['Log_Returns'].std()

# Verificar o último preço de fechamento
last_price = data['Close'].iloc[-1]
if last_price <= 0:
    print("Erro: O último preço de fechamento é inválido.")
    exit()


# ===== SIMULAÇÃO DE MONTE CARLO (Sem Vetorização) =====

# Medir o tempo para simulação
start_simulation = datetime.now()

# Gerar choques aleatórios para todas as simulações
time_steps = num_days
random_shocks = np.random.normal(0, 1, size=(time_steps, num_simulations))
simulated_prices = last_price * np.exp(
    np.cumsum((mu - 0.5 * sigma ** 2) + sigma * random_shocks, axis=0)
)

end_simulation = datetime.now()

## Código obsoleto demora absurdamente mais quando feito com loop ao inves de vetor numpy
# # Medir o tempo para simulação
# start_simulation = time.perf_counter()

# simulated_prices = np.zeros((num_days, num_simulations))
# simulated_prices[0, :] = last_price

# for sim in range(num_simulations):  # Loop por simulação
#     for day in range(1, num_days):  # Loop por dia
#         random_shock = np.random.normal(0, 1)  # Gerar choque aleatório
#         simulated_prices[day, sim] = simulated_prices[day - 1, sim] * np.exp(
#             (mu - 0.5 * sigma ** 2) + sigma * random_shock
#         )

# end_simulation = time.perf_counter()


# ===== RESUMO DOS RESULTADOS =====
final_prices = simulated_prices[-1, :]
cenario_pessimista = np.percentile(final_prices, 5)
cenario_otimista = np.percentile(final_prices, 95)
estimativa_preco_media = np.mean(final_prices)
estimativa_preco_mediana = np.median(final_prices)
percentual_crescimento_pessimista = ((cenario_pessimista / last_price) - 1) * 100
percentual_crescimento_otimista = ((cenario_otimista / last_price) - 1) * 100

# Percentual de crescimento médio
percentual_crescimento = ((final_prices / last_price) - 1) * 100
crescimento_medio = np.mean(percentual_crescimento)
conf_interval = 1.96 * np.std(percentual_crescimento) / np.sqrt(num_simulations)
margem_erro_crescimento_media = conf_interval * 2


# Calcular a diferença entre a data máxima e a data mínima
data_inicio = data['Date'].min()
data_fim = data['Date'].max()
diferenca_tempo = data_fim - data_inicio

# Converter o intervalo de tempo em anos
intervalo_anos = diferenca_tempo.days / 365.25

# Exibir resultados
print(f"\nAção: NVDC34")
print(f"Empresa: NVIDIA")

print(f"\nData de início da análise: {data_inicio.strftime('%Y-%m-%d')}")
print(f"Data de término da análise: {data_fim.strftime('%Y-%m-%d')}")
print(f"Intervalo de dias: {formatar_inteiro(diferenca_tempo.days)} dias")
print(f"Intervalo de anos: {intervalo_anos:.1f} anos")

print(f"\nValor atual: ${formatar_decimais(last_price)} (em {data['Date'].iloc[-1].strftime('%Y-%m-%d')})")
print(f"Número de dias úteis no ano: {formatar_inteiro(num_days)}")
print(f"Número de Simulações Monte Carlo: {formatar_inteiro(num_simulations)}")

print(f"\nCenário pessimista (5º Percentil): {formatar_decimais(percentual_crescimento_pessimista)}% (${formatar_decimais(cenario_pessimista)})")
print(f"Cenário otimista (95º Percentil): {formatar_decimais(percentual_crescimento_otimista)}% (${formatar_decimais(cenario_otimista)})")
print(f"\nCenário previsto (média): {formatar_decimais(crescimento_medio)}% (${formatar_decimais(estimativa_preco_media)})")
print(f"Margem de Erro (média): {formatar_decimais(margem_erro_crescimento_media)}%")
print(f"Intervalo de Confiança: 95% (centrado na média)")


# Plotar os caminhos simulados
plt.figure(figsize=(12, 6))
plt.plot(simulated_prices[:, :100], alpha=0.1, color='blue')  # Apenas 100 simulações para visualização
plt.axhline(y=cenario_pessimista, color='red', linestyle='--', label='Pessimista (5%)')
plt.axhline(y=cenario_otimista, color='green', linestyle='--', label='Otimista (95%)')
plt.axhline(y=estimativa_preco_media, color='black', linestyle='-', label='Média')
plt.legend()
plt.title('Simulação de Monte Carlo dos Preços da Ação da NVIDIA (1 Ano)')
plt.xlabel('Dias')
plt.ylabel('Preço (USD)')
plt.grid()



# ========== FIM DO CÓDIGO ==========

# Coletar lixo e limpar variáveis
gc.collect()
random_shocks = None
simulated_prices = None

# Registrar o término do script
end_time = datetime.now()
duration = end_time - start_time
formatted_duration = str(duration).split(".")[0]
print(f"\nFinalizado em: {end_time.strftime('%Y-%m-%d %H:%M:%S')} (Duração: {formatted_duration})\n")

plt.show() #exibir o gráfico após contabilizar tempo
