# Implementação de um Perceptron

Este repositório contém uma implementação de um **Perceptron**, um dos algoritmos mais básicos e fundamentais em aprendizado de máquina supervisionado. O Perceptron é um classificador linear que aprende a separar dados de diferentes classes por meio de um hiperplano.

## Principais Funcionalidades
- **Treinamento Supervisionado**: O algoritmo ajusta seus pesos com base no erro cometido durante o treinamento.
- **Customização de Hiperparâmetros**: Possibilidade de ajustar a taxa de aprendizado (learning rate) e o número de iterações para melhor desempenho.
- **Visualização do Processo de Aprendizado**: Gráficos que mostram a evolução do hiperplano de separação (opcional, caso implementado).

## O que é o Perceptron?
O Perceptron é um modelo simples de rede neural criado por **Frank Rosenblatt** em 1958. Ele utiliza uma combinação linear dos pesos e entradas para realizar previsões binárias. A cada erro cometido, os pesos são atualizados usando a regra de aprendizado do Perceptron:

**Regra de Atualização:**
\[ w_{t+1} = w_t + \eta (y - \hat{y}) x \]

Onde:
- \( w_t \): pesos atuais
- \( \eta \): taxa de aprendizado
- \( y \): rótulo verdadeiro
- \( \hat{y} \): previsão do modelo
- \( x \): entrada correspondente

## Aplicações
Embora simples, o Perceptron é útil para:
- Problemas de classificação linearmente separáveis
- Introdução ao funcionamento de redes neurais
- Estudo e ensino de algoritmos de aprendizado de máquina

## Compilar
- gcc perceptron.c -lm -o output && ./output
- Mais detalhes sobre o programa encontra-se em LEIA-ME.txt
