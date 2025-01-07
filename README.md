# Análise preditiva das Ações da Nvidia

Neste projeto, fiz um estudo de **Data Science** usando modelos de **Machine Learning** para prever as ações da **Nvidia** para o ano de **2025**.



## Desafio

Tendo em mente que as ações são extremamente voláteis e difíceis de prever, foram utilizados alguns modelos de **Machine Learning** e **Deep Learning**, como:

- **PyTorch**
- **TensorFlow**
- **LSTM**
- **NeuralProphet**
- **Prophet**
- **Análise de Monte Carlo**



## Resultados

O resultado mais interessante foi com o modelo **Prophet** e a análise Monte Carlo, que atingiu uma previsão que aparenta ser bem realista, os modelos de deeplearning necessitaram de mais dados e não desempenharam tão bem.



## Análise Estatística de Monte Carlo

![Análise de Monte Carlo](foto_nvidia_monte_carlo_final.png)

Já na **análise de Monte Carlo**, simulei 1 milhão de cenários e considerei os cenários pessimistas e otimistas para chegar a uma média interessante.

- **Cenário otimista**: Crescimento de **206%**
- **Cenário pessimista**: Queda de **-57%**
- **Média**: Crescimento de **37%**



## Análise de Machine Learning (Prophet)

![Descrição da Imagem](foto_nvidia_forecasting_prophet_final.png)

Já a análise de **Machine Learning** com o modelo **Prophet** prevê um aumento de **72,8%** nas ações até o último dia de 2025.



## Análise de Deep Learning (LSTM - Long Short-Term Memory)

![Descrição da Imagem](foto_nvidia_forecasting_deeplearning_lstm_redes_neurais.png)

Já a análise de **Deep Learning** com o modelo **LSTM** prevê um aumento de **56,5%** nas ações até o último dia de 2025.




## Conclusão

Na minha opinião o melhor modelo foi o LSTM, por ser capaz de traduzir um cenário mais realista possível.

E aí? Será que alguma dessas previsões vai se confirmar? Vamos acompanhar e ver como o mercado da Nvidia se comporta em 2025!
