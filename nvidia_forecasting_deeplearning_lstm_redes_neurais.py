import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model

# ===== CARREGAR E PREPARAR OS DADOS =====
input_path = r"C:\\Users\\david\\OneDrive\\Python VsCode\\ML NVIDIA STOCK\\"
input_file = "NVDA_1999-01-01_2024-12-04.csv"
data = pd.read_csv(input_path + input_file)

data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Filtrar apenas o ano de 2024
data_2024 = data.loc['2024-01-01':'2024-12-31']

# Normalizar os preços de fechamento
scaler = MinMaxScaler(feature_range=(0, 1))
data_2024['Close_scaled'] = scaler.fit_transform(data_2024[['Close']])

# Criar janelas de tempo
def create_sequences(data, window_size):
    sequences = []
    for i in range(len(data) - window_size):
        sequences.append(data[i:i + window_size])
    return np.array(sequences)

window_size = 30
sequences = create_sequences(data_2024['Close_scaled'].values, window_size)

# Dividir em treino e teste
test_size = int(0.2 * len(sequences))
X_train = sequences[:-test_size]
X_test = sequences[-test_size:]

# Ajustar o formato dos dados
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# ===== CARREGAR O MODELO TREINADO =====
model = load_model(input_path + 'lstm_nvda_model.h5')

# ===== PREVER OS VALORES =====
future_steps = 365  # Previsão para 2025
last_sequence = X_test[-1]  # Última sequência conhecida
predictions = []

for _ in range(future_steps):
    pred = model.predict(last_sequence.reshape(1, window_size, 1), verbose=0)
    predictions.append(pred[0, 0])
    last_sequence = np.append(last_sequence[1:], pred, axis=0)

# Reverter a normalização
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Ajustar previsões com base em valores absolutos históricos
avg_daily_change = data_2024['Close'].diff().mean()  # Mudança média diária em valores absolutos
daily_std_dev = data_2024['Close'].diff().std()  # Desvio padrão diário
adjusted_predictions = []
current_price = data_2024['Close'].iloc[-1]

for _ in range(future_steps):
    random_change = np.random.normal(avg_daily_change, daily_std_dev)  # Mudança aleatória diária
    current_price += random_change
    current_price = max(0, current_price)  # Garantir que o preço não seja negativo
    adjusted_predictions.append(current_price)

adjusted_predictions = np.array(adjusted_predictions)

# Adicionar as previsões ajustadas ao DataFrame
dates_future = pd.date_range(start=data_2024.index[-1] + pd.Timedelta(days=1), periods=future_steps)
predicted_df = pd.DataFrame({'Date': dates_future, 'Predicted': adjusted_predictions})
predicted_df.set_index('Date', inplace=True)

# Cenário otimista e pessimista
optimistic_predictions = adjusted_predictions + daily_std_dev  # Cenário otimista baseado no desvio padrão
pessimistic_predictions = adjusted_predictions - daily_std_dev  # Cenário pessimista baseado no desvio padrão

# ===== PLOTAR O GRÁFICO =====
plt.figure(figsize=(14, 7))

# Valores reais (2024)
plt.plot(data_2024['Close'], label='Valores Reais - 2024', color='green')

# Previsões (2025)
plt.plot(predicted_df.index, predicted_df['Predicted'], label='Previsões - 2025', color='blue')

# Intervalo otimista e pessimista
plt.fill_between(
    predicted_df.index,
    pessimistic_predictions,
    optimistic_predictions,
    color='blue',
    alpha=0.2,
    label='Intervalo de Confiança'
)

# Linha indicando início da previsão
plt.axvline(x=data_2024.index[-1], color='red', linestyle='--', label='Início da Previsão (2025)')

# Configurações do gráfico
plt.title('Previsão Detalhada de Preços das Ações da NVIDIA (2024-2025)')
plt.xlabel('Data')
plt.ylabel('Preço de Fechamento em Dólares')
plt.legend()
plt.grid()
plt.show()

# ===== CALCULAR CRESCIMENTO PERCENTUAL =====
last_real_value = data_2024['Close'].iloc[-1]
final_forecast_value = adjusted_predictions[-1]
growth_percentage = ((final_forecast_value - last_real_value) / last_real_value) * 100
optimistic_growth = ((optimistic_predictions[-1] - last_real_value) / last_real_value) * 100
pessimistic_growth = ((pessimistic_predictions[-1] - last_real_value) / last_real_value) * 100

# Exibir resultados
print(f"Valor final previsto para o final de 2025: ${final_forecast_value:.2f}")
print(f"Crescimento percentual previsto: {growth_percentage:.2f}%")
print(f"Cenário otimista: Crescimento de {optimistic_growth:.2f}%")
print(f"Cenário pessimista: Crescimento de {pessimistic_growth:.2f}%")
