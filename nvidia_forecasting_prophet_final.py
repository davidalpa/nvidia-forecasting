import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# Carregar os dados
input_path = r"C:\Users\david\OneDrive\Python VsCode\ML NVIDIA STOCK\\"
input_file = "NVDA_1999-01-01_2024-12-04.csv"
input_final = input_path + input_file

data = pd.read_csv(input_final)
data['Date'] = pd.to_datetime(data['Date'])
data.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

# Filtrar apenas os dados de 2024
data_2024 = data[(data['ds'] >= '2024-01-01') & (data['ds'] <= '2024-12-31')]

# Criar o modelo Prophet e ajustar aos dados de 2024
model = Prophet(daily_seasonality=True, yearly_seasonality=True)
model.fit(data_2024)

# Criar dataframe para previsões (2025)
future = model.make_future_dataframe(periods=365, freq='D')  # Prever até o final de 2025
forecast = model.predict(future)

# Ajustar as previsões para refletir o comportamento realista
last_real_date = data_2024['ds'].iloc[-1]
last_real_value = data_2024['y'].iloc[-1]
daily_std = data_2024['y'].diff().std()  # Desvio padrão das diferenças diárias
amplitude_scale = 0.7  # Fator de ajuste para a oscilação

# Garantir continuidade da previsão e adicionar oscilações realistas
forecast['yhat_with_structure'] = forecast['yhat']
forecast.loc[forecast['ds'] > last_real_date, 'yhat_with_structure'] = (
    forecast.loc[forecast['ds'] > last_real_date, 'yhat']
    + last_real_value
    - forecast.loc[forecast['ds'] == last_real_date, 'yhat'].values[0]
)

# Adicionar oscilações suavizadas
oscillation_noise = np.random.normal(0, amplitude_scale * daily_std, len(forecast[forecast['ds'] > last_real_date]))
forecast.loc[forecast['ds'] > last_real_date, 'yhat_with_structure'] += oscillation_noise

# Garantir que os valores não sejam negativos
forecast['yhat_with_structure'] = forecast['yhat_with_structure'].clip(lower=0)

# Plotar os resultados
plt.figure(figsize=(15, 6))

# Dados reais (2024)
plt.plot(data_2024['ds'], data_2024['y'], color='green', label='Valores Reais - 2024')

# Previsões ajustadas (2025)
plt.plot(
    forecast.loc[forecast['ds'] > last_real_date, 'ds'],
    forecast.loc[forecast['ds'] > last_real_date, 'yhat_with_structure'],
    color='blue',
    label='Previsões com Oscilações - 2025'
)

# Intervalo de confiança
plt.fill_between(
    forecast.loc[forecast['ds'] > last_real_date, 'ds'],
    forecast.loc[forecast['ds'] > last_real_date, 'yhat_with_structure'] - amplitude_scale * daily_std,
    forecast.loc[forecast['ds'] > last_real_date, 'yhat_with_structure'] + amplitude_scale * daily_std,
    color='blue',
    alpha=0.2,
    label='Intervalo de Confiança'
)

# Linha vertical indicando início da previsão
plt.axvline(x=last_real_date, color='red', linestyle='--', label='Início da Previsão (2025)')

# Configurações do gráfico
plt.title('Previsão Detalhada de Preços das Ações da NVIDIA (2024-2025)')
plt.xlabel('Data')
plt.ylabel('Preço de Fechamento em Dólares')
plt.legend()
plt.grid()
plt.show()

# Calcular o valor final da ação prevista e o percentual de crescimento
final_forecast_value = forecast.loc[forecast['ds'] == forecast['ds'].max(), 'yhat_with_structure'].values[0]
growth_percentage = ((final_forecast_value - last_real_value) / last_real_value) * 100

# Cenário otimista e pessimista
optimistic_value = final_forecast_value + (amplitude_scale * daily_std)
pessimistic_value = final_forecast_value - (amplitude_scale * daily_std)
optimistic_growth = ((optimistic_value - last_real_value) / last_real_value) * 100
pessimistic_growth = ((pessimistic_value - last_real_value) / last_real_value) * 100

# Exibir os resultados
print(f"Valor final previsto da ação (2025): ${final_forecast_value:.2f}")
print(f"Crescimento percentual em relação ao último valor de 2024: {growth_percentage:.2f}%")
print(f"Cenário otimista - Valor: ${optimistic_value:.2f}, Crescimento: {optimistic_growth:.2f}%")
print(f"Cenário pessimista - Valor: ${pessimistic_value:.2f}, Crescimento: {pessimistic_growth:.2f}%")
