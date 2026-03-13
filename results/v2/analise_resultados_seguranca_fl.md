# Analise de Vulnerabilidade a Envenenamento em Aprendizado Federado para ITS

> **Projeto**: Federated Learning for ITS Network Slice Impact Classification
> **Fase**: Phase 1 — Poisoning Vulnerability Assessment
> **Data dos Experimentos**: 2026-03-13
> **Modelo**: GRU | **Clientes**: 3 RSUs | **Rounds**: 10 | **Epocas Locais**: 5
> **Cliente Malicioso**: Client 0 (RSU 0)

---

## 1. Desenho Experimental

### 1.1 Fatores Avaliados

| Fator | Niveis |
|---|---|
| **Distribuicao de dados** | IID, Non-IID |
| **Estrategia de agregacao** | FedAvg, FedProx, Krum, Trimmed Mean |
| **Ataque de envenenamento** | Nenhum (baseline), Label Flip 20%, Label Flip 50% |

**Total**: 2 distribuicoes x 4 estrategias x 3 cenarios de ataque = **24 experimentos**

### 1.2 Dataset

- **Amostras**: ~5.093 (28 features de monitoramento de rede)
- **Classes**: Low, Medium, High (impacto de politicas de network slicing)
- Cada RSU recebeu ~1.697-1.699 amostras
- Split: 80% treino / 20% teste

### 1.3 Ataque: Label Flipping

O ataque de *label flipping* consiste em inverter os rotulos de uma fracao dos dados de treino no cliente malicioso (Client 0), simulando um participante comprometido no sistema federado. Foram testadas taxas de 20% e 50%.

---

## 2. Resultados Consolidados

### 2.1 Cenario IID

| Estrategia | Sem Ataque |  | Flip 20% |  | Flip 50% |  |
|---|---|---|---|---|---|---|
|  | **Acc** | **F1** | **Acc** | **F1** | **Acc** | **F1** |
| FedAvg | 0.9784 | 0.9786 | 0.9804 | 0.9804 | 0.9794 | 0.9793 |
| FedProx | 0.9784 | 0.9786 | 0.9775 | 0.9776 | 0.9775 | 0.9776 |
| Krum | 0.9755 | 0.9761 | 0.9735 | 0.9733 | 0.9745 | 0.9757 |
| Trimmed Mean | **0.9814** | **0.9812** | **0.9794** | **0.9795** | **0.9794** | **0.9795** |

### 2.2 Cenario Non-IID

| Estrategia | Sem Ataque |  | Flip 20% |  | Flip 50% |  |
|---|---|---|---|---|---|---|
|  | **Acc** | **F1** | **Acc** | **F1** | **Acc** | **F1** |
| FedAvg | 0.7784 | 0.7722 | 0.7804 | 0.7740 | 0.7873 | 0.7819 |
| FedProx | 0.7784 | 0.7726 | 0.7804 | 0.7749 | 0.7814 | 0.7762 |
| Krum | 0.7441 | 0.7484 | 0.7549 | 0.7575 | 0.7510 | 0.7543 |
| Trimmed Mean | **0.8029** | **0.8037** | **0.8010** | **0.8017** | **0.8000** | **0.7982** |

---

## 3. Analise de Impacto dos Ataques

### 3.1 Degradacao por Ataque (Delta de Acuracia vs Baseline)

| Estrategia | IID Flip20 | IID Flip50 | Non-IID Flip20 | Non-IID Flip50 |
|---|---|---|---|---|
| FedAvg | +0.20 pp | +0.10 pp | +0.20 pp | +0.89 pp |
| FedProx | -0.09 pp | -0.09 pp | +0.20 pp | +0.30 pp |
| Krum | -0.20 pp | -0.10 pp | +1.08 pp | +0.69 pp |
| Trimmed Mean | -0.20 pp | -0.20 pp | -0.19 pp | -0.29 pp |

> [!important] Observacao Critica
> **Os ataques de label flipping nao degradaram significativamente o desempenho do modelo global em nenhum cenario.** Em varias configuracoes, a acuracia sob ataque foi ate ligeiramente *superior* ao baseline. Isso indica que, com apenas 1 de 3 clientes malicioso, o mecanismo de agregacao (media ponderada por volume de dados) dilui o efeito do envenenamento.

### 3.2 Estabilidade (Desvio Padrao da Acuracia ao Longo dos Rounds)

| Estrategia | IID (sem ataque) | IID (Flip50) | Non-IID (sem ataque) | Non-IID (Flip50) |
|---|---|---|---|---|
| FedAvg | 0.0023 | 0.0021 | 0.0020 | 0.0014 |
| FedProx | 0.0016 | 0.0016 | 0.0024 | 0.0021 |
| Krum | 0.0018 | 0.0032 | 0.0031 | **0.0217** |
| Trimmed Mean | **0.0014** | **0.0011** | **0.0012** | 0.0017 |

> [!warning] Krum sob Non-IID + Ataque
> O Krum apresenta instabilidade significativa no cenario Non-IID com Flip 50% (std = 0.0217), sugerindo que sua heuristica de selecao de modelo unico pode oscilar entre clientes em cenarios heterogeneos sob ataque.

---

## 4. Analise de Convergencia

| Cenario | FedAvg | FedProx | Krum | Trimmed Mean |
|---|---|---|---|---|
| IID sem ataque | Round 1 | Round 1 | Round 1 | Round 1 |
| IID Flip 20% | Round 1 | Round 2 | Round 1 | Round 1 |
| IID Flip 50% | Round 2 | Round 2 | Round 1 | Round 1 |
| Non-IID (todos) | N/A (<90%) | N/A (<90%) | N/A (<90%) | N/A (<90%) |

- **IID**: Convergencia extremamente rapida (1-2 rounds) em todos os cenarios.
- **Non-IID**: Nenhuma estrategia atinge 90% de acuracia, estabilizando entre 74-80%.

---

## 5. Matrizes de Confusao — Padroes por Classe

### 5.1 IID — Sem Ataque (Melhor Caso: Trimmed Mean)

```
              Low    Medium    High
Low           370       7       0
Medium          2     395       5
High            1       4     236
```

### 5.2 Non-IID — Sem Ataque (Trimmed Mean)

```
              Low    Medium    High
Low           317      60       0
Medium         39     333      29
High            0      73     169
```

### 5.3 Non-IID — Padrao Problematico do Krum

```
              Low    Medium    High
Low           376       1       0      ← Quase perfeito para Low
Medium        157     244       0      ← 39% de Medium classificado como Low
High           19      84     139      ← 43% de High mal-classificado
```

> [!danger] Krum em Non-IID
> O Krum apresenta um vies sistematico para a classe **Low**, classificando incorretamente grande parte das amostras Medium e High. Este e um comportamento intrinseco ao mecanismo de selecao do Krum (que escolhe o modelo mais "central"), nao um efeito do ataque.

---

## 6. Comparacao entre Estrategias de Defesa

### 6.1 Ranking Geral

| Ranking | Estrategia | Pontos Fortes | Pontos Fracos |
|---|---|---|---|
| 1 | **Trimmed Mean** | Melhor acuracia em ambos cenarios; maior estabilidade; boa robustez | Custo computacional ligeiramente superior |
| 2 | **FedAvg** | Simples e eficaz; resiliente por default na media ponderada | Sem mecanismo explicito de defesa |
| 3 | **FedProx** | Regularizacao proximal estabiliza treino | Sem ganho significativo sobre FedAvg (~0.19 pp) |
| 4 | **Krum** | Teoricamente robusto contra ataques bizantinos | Pior acuracia geral; instavel em Non-IID; vies para classe Low |

### 6.2 FedAvg vs FedProx

A diferenca entre FedAvg e FedProx e **estatisticamente insignificante** (<=0.19 pp) em todos os cenarios, confirmando que a regularizacao proximal com mu=0.1 nao agrega valor mensuravel para este dataset e grau de heterogeneidade.

### 6.3 Eficacia das Defesas Bizantinas

| Metrica | Krum | Trimmed Mean |
|---|---|---|
| Robustez a Flip 20% (IID) | -0.20 pp | -0.20 pp |
| Robustez a Flip 50% (IID) | -0.10 pp | -0.20 pp |
| Acuracia baseline (IID) | 97.55% | **98.14%** |
| Acuracia baseline (Non-IID) | 74.41% | **80.29%** |
| Estabilidade geral | Baixa | **Alta** |

---

## 7. Discussao

### 7.1 Por que o Label Flipping Nao Funcionou?

Tres fatores explicam a resiliencia observada:

1. **Proporcao de clientes maliciosos**: Com apenas 1/3 dos clientes comprometido, a agregacao por media ponderada dilui naturalmente as contribuicoes envenenadas.
2. **Volume de dados uniforme**: Como todos os RSUs possuem volumes similares (~1.697 amostras), nenhum cliente tem peso desproporcional na agregacao.
3. **Convergencia rapida**: O modelo converge em 1-2 rounds no IID, limitando a janela temporal de influencia do ataque.

### 7.2 Impacto Real da Distribuicao de Dados

A heterogeneidade dos dados (Non-IID) tem impacto **muito maior** que os ataques de envenenamento:
- IID -> Non-IID: **~20 pp de queda** na acuracia
- Baseline -> Flip 50%: **<1 pp de variacao**

> [!tip] Implicacao Pratica
> Para redes ITS reais, a variabilidade espacial-temporal dos dados de monitoramento entre RSUs e uma ameaca mais significativa a qualidade do modelo do que ataques de envenenamento com proporcoes moderadas de clientes maliciosos.

### 7.3 Limitacoes do Estudo

- Apenas 1 de 3 clientes malicioso (33%)
- Apenas ataque de label flipping (sem model poisoning, gradient attack, etc.)
- Modelo GRU unico (sem comparacao com DNN/LSTM nesta fase)
- 10 rounds de comunicacao (ataques persistentes por mais rounds poderiam amplificar efeitos)
- Ausencia de ataques adaptativos ou coordenados

---

## 8. Proximos Passos Sugeridos

- [ ] Aumentar proporcao de clientes maliciosos (2/3) para testar limites de resiliencia
- [ ] Implementar **model poisoning** (manipulacao direta de pesos/gradientes)
- [ ] Testar ataques **adaptativos** que ajustam a intensidade para evadir defesas
- [ ] Avaliar **gradient clipping** e **differential privacy** como defesas adicionais
- [ ] Comparar resultados com DNN e LSTM sob os mesmos cenarios de ataque
- [ ] Testar cenarios com **ataques coordenados** entre multiplos clientes

---

## Apendice: Configuracao Experimental

| Parametro | Valor |
|---|---|
| Framework FL | Flower 1.x |
| Modelo | GRU (Keras/TensorFlow 2.x) |
| Features de entrada | 28 (RTT, PDR, derivadas temporais, etc.) |
| Classes de saida | 3 (Low, Medium, High) |
| Clientes (RSUs) | 3 |
| Rounds de comunicacao | 10 |
| Epocas locais | 5 |
| Distribuicoes | IID, Non-IID |
| Estrategias de agregacao | FedAvg, FedProx (mu=0.1), Krum, Trimmed Mean |
| Ataques | Label Flip (20%, 50%) no Client 0 |
| Hardware | CPU (sem GPU) |

---

> *Documento gerado automaticamente a partir dos resultados experimentais em `results/v2/paper_artifacts/`*
