# Analise dos Resultados - Phase 1b: Vulnerabilidade de Seguranca em FL

> **Data da execucao:** 2026-03-14
> **Experimento:** Phase 1b - Label Flipping com Model Scaling
> **Dataset:** ITS (Intelligent Transportation Systems) - Classificacao de severidade (Low/Medium/High)
> **Modelo:** GRU
> **Configuracao:** 3 clientes, 10 rounds, 5 epocas locais, 1 cliente malicioso (client_0)

---

## 1. Resumo Executivo

Foram executadas **24 configuracoes** experimentais combinando:
- **2 distribuicoes de dados:** IID e Non-IID
- **3 estrategias de agregacao:** FedAvg, Krum, Trimmed Mean
- **4 fatores de escala do ataque:** 1x (sem amplificacao), 5x, 10x, 20x

O ataque consiste em **label flipping a 100%** (flip de High para Low) por um cliente malicioso, com amplificacao opcional do gradiente (model scaling). A metrica critica e a **H2L Rate** (taxa de amostras verdadeiramente High classificadas como Low), que representa o impacto direto na seguranca viaria.

### Principais Achados

- **FedAvg e extremamente vulneravel** a ataques com model scaling, com H2L Rate atingindo 100% em multiplas configuracoes
- **Krum e Trimmed Mean sao robustos**, mantendo H2L Rate proxima de 0% mesmo sob scaling de 20x
- O cenario **Non-IID amplifica a vulnerabilidade** do FedAvg ao ataque
- O ataque de label flipping **sem scaling ja causa degradacao perceptivel** no FedAvg

---

## 2. Resultados Detalhados

### 2.1 Cenario IID

| Estrategia | Escala | Accuracy | Acc Low | Acc Med | Acc High | H2L Rate | H2L Count/Total |
|:-----------|:------:|:--------:|:-------:|:-------:|:--------:|:--------:|:---------------:|
| **FedAvg** | 1x | 0.9529 | 0.979 | 0.993 | 0.846 | 4.15% | 10/241 |
| **FedAvg** | 5x | 0.5784 | 0.732 | 0.781 | **0.000** | **100.0%** | 241/241 |
| **FedAvg** | 10x | 0.4157 | 0.997 | 0.119 | **0.000** | **100.0%** | 241/241 |
| **FedAvg** | 20x | 0.6157 | 0.981 | 0.642 | **0.000** | **90.87%** | 219/241 |
| **Krum** | 1x | 0.9686 | 0.981 | 0.968 | 0.950 | 0.41% | 1/241 |
| **Krum** | 5x | 0.9784 | 0.963 | 0.983 | 0.996 | 0.41% | 1/241 |
| **Krum** | 10x | 0.9745 | 0.981 | 0.970 | 0.971 | 0.41% | 1/241 |
| **Krum** | 20x | 0.9725 | 0.960 | 0.970 | 0.996 | 0.41% | 1/241 |
| **Trimmed Mean** | 1x | 0.9755 | 0.976 | 0.983 | 0.963 | 0.41% | 1/241 |
| **Trimmed Mean** | 5x | 0.9775 | 0.976 | 0.985 | 0.967 | 0.41% | 1/241 |
| **Trimmed Mean** | 10x | 0.9775 | 0.976 | 0.985 | 0.967 | 0.41% | 1/241 |
| **Trimmed Mean** | 20x | 0.9775 | 0.976 | 0.985 | 0.967 | 0.41% | 1/241 |

### 2.2 Cenario Non-IID

| Estrategia | Escala | Accuracy | Acc Low | Acc Med | Acc High | H2L Rate | H2L Count/Total |
|:-----------|:------:|:--------:|:-------:|:-------:|:--------:|:--------:|:---------------:|
| **FedAvg** | 1x | 0.7824 | 0.902 | 0.791 | 0.583 | 5.79% | 14/242 |
| **FedAvg** | 5x | 0.4902 | 0.639 | 0.389 | 0.426 | **50.83%** | 123/242 |
| **FedAvg** | 10x | 0.4245 | 0.995 | 0.145 | **0.000** | **100.0%** | 242/242 |
| **FedAvg** | 20x | 0.3696 | 1.000 | 0.000 | **0.000** | **100.0%** | 242/242 |
| **Krum** | 1x | 0.7471 | 0.997 | 0.623 | 0.562 | 7.02% | 17/242 |
| **Krum** | 5x | 0.8069 | 0.836 | 0.848 | 0.694 | 0.00% | 0/242 |
| **Krum** | 10x | 0.8069 | 0.825 | 0.855 | 0.698 | 0.00% | 0/242 |
| **Krum** | 20x | 0.7951 | 0.828 | 0.825 | 0.694 | 0.00% | 0/242 |
| **Trimmed Mean** | 1x | 0.7824 | 0.931 | 0.763 | 0.583 | 0.83% | 2/242 |
| **Trimmed Mean** | 5x | 0.7824 | 0.923 | 0.768 | 0.587 | 1.24% | 3/242 |
| **Trimmed Mean** | 10x | 0.7843 | 0.920 | 0.773 | 0.591 | 1.24% | 3/242 |
| **Trimmed Mean** | 20x | 0.7833 | 0.918 | 0.773 | 0.591 | 1.24% | 3/242 |

---

## 3. Analise por Estrategia de Agregacao

### 3.1 FedAvg - Altamente Vulneravel

```
Degradacao de Accuracy (IID):
  1x:  95.29% ████████████████████████████████████████████████ (baseline)
  5x:  57.84% █████████████████████████████                    (-37.45pp)
  10x: 41.57% █████████████████████                            (-53.72pp)
  20x: 61.57% ███████████████████████████████                  (-33.72pp)

Degradacao de Accuracy (Non-IID):
  1x:  78.24% ███████████████████████████████████████          (baseline)
  5x:  49.02% █████████████████████████                        (-29.22pp)
  10x: 42.45% █████████████████████                            (-35.79pp)
  20x: 36.96% ███████████████████                              (-41.28pp)
```

**Observacoes criticas:**
- No cenario IID com scale 5x e 10x, a classe **High tem accuracy 0%** - todas as amostras de alta severidade sao classificadas incorretamente
- No cenario Non-IID com scale 20x, o modelo **colapsa completamente**: apenas prediz "Low" (accuracy Low=100%, Med=0%, High=0%)
- O modelo se torna um **classificador trivial** que ignora severidade real, o que em um cenario real de transito pode significar **nao detectar acidentes graves**

### 3.2 Krum - Robusto

```
Accuracy IID (estavel em todas as escalas):
  1x:  96.86% | 5x:  97.84% | 10x: 97.45% | 20x: 97.25%

Accuracy Non-IID:
  1x:  74.71% | 5x:  80.69% | 10x: 80.69% | 20x: 79.51%
```

- **H2L Rate IID: 0.41% constante** (apenas 1 amostra mal classificada em todos os cenarios)
- **H2L Rate Non-IID: 0.00% com scaling** (Krum rejeita completamente o modelo malicioso amplificado)
- Paradoxalmente, Krum com scaling apresenta **melhor accuracy Non-IID** do que sem scaling (80.69% vs 74.71%), pois o modelo malicioso amplificado e mais facil de detectar e rejeitar

### 3.3 Trimmed Mean - Robusto

```
Accuracy IID (estavel em todas as escalas):
  1x:  97.55% | 5x:  97.75% | 10x: 97.75% | 20x: 97.75%

Accuracy Non-IID:
  1x:  78.24% | 5x:  78.24% | 10x: 78.43% | 20x: 78.33%
```

- **H2L Rate quase zero** em todos os cenarios
- **Desempenho mais estavel que Krum** no cenario Non-IID
- O mecanismo de trimming efetivamente neutraliza as atualizacoes extremas do atacante

---

## 4. Impacto do Model Scaling no Ataque

### 4.1 Efeito sobre FedAvg

| Metrica | 1x | 5x | 10x | 20x |
|:--------|:---:|:---:|:---:|:---:|
| H2L Rate IID | 4.15% | 100% | 100% | 90.87% |
| H2L Rate Non-IID | 5.79% | 50.83% | 100% | 100% |
| Acc High IID | 84.6% | 0% | 0% | 0% |
| Acc High Non-IID | 58.3% | 42.6% | 0% | 0% |

O scaling amplifica dramaticamente o impacto do ataque no FedAvg. Mesmo com scale 5x, a classe High e completamente eliminada no cenario IID.

### 4.2 Efeito sobre Krum e Trimmed Mean

O model scaling **nao tem efeito pratico** sobre Krum e Trimmed Mean. Ambas as estrategias mantem performance estavel independentemente do fator de amplificacao, demonstrando que seus mecanismos de defesa sao eficazes contra este tipo de ataque.

---

## 5. Analise de Risco para ITS (Seguranca Viaria)

### Cenario de Risco Critico

No contexto de sistemas inteligentes de transporte, a classificacao incorreta de eventos de **alta severidade como baixa severidade** (H2L) representa um risco direto:

| Cenario | H2L Rate | Impacto |
|:--------|:--------:|:--------|
| FedAvg + IID + scale 10x | **100%** | Todos os eventos graves ignorados |
| FedAvg + Non-IID + scale 20x | **100%** | Colapso total do classificador |
| FedAvg + IID + scale 5x | **100%** | Falha completa na deteccao de severidade alta |
| FedAvg + Non-IID + scale 5x | **50.83%** | Metade dos eventos graves nao detectados |

### Recomendacao

> **FedAvg NAO deve ser utilizado em cenarios FL para ITS sem mecanismos adicionais de defesa.** Krum e Trimmed Mean oferecem protecao intrinseca contra ataques de label flipping com model scaling.

---

## 6. Comparativo de Robustez

```
                    FedAvg         Krum          Trimmed Mean
                 IID   Non-IID   IID  Non-IID   IID   Non-IID
Sem ataque(1x)   ★★★★  ★★★     ★★★★★ ★★★      ★★★★★  ★★★
Scale 5x          ★     ★★       ★★★★★ ★★★★    ★★★★★  ★★★
Scale 10x         ★     ★        ★★★★★ ★★★★    ★★★★★  ★★★
Scale 20x         ★★    ★        ★★★★★ ★★★★    ★★★★★  ★★★

Legenda: ★ = Muito Ruim | ★★★ = Aceitavel | ★★★★★ = Excelente
```

---

## 7. Configuracao Experimental

| Parametro | Valor |
|:----------|:------|
| Modelo | GRU (Gated Recurrent Unit) |
| Num. Clientes | 3 |
| Num. Rounds | 10 |
| Epocas Locais | 5 |
| Batch Size | 32 |
| Clientes Maliciosos | 1 (client_0) |
| Tipo de Ataque | Label Flipping (High -> Low) |
| Fracao do Ataque | 100% dos dados do cliente malicioso |
| Distribuicoes | IID, Non-IID |
| Estrategias | FedAvg, Krum, Trimmed Mean |
| Fatores de Escala | 1x, 5x, 10x, 20x |
| FedProx mu | 0.1 |

---

## 8. Conclusoes

1. **FedAvg e fundamentalmente inseguro** contra ataques de label flipping com model scaling, com a classe alvo (High) sendo completamente suprimida na maioria das configuracoes com scaling
2. **Krum demonstra robustez excepcional**, especialmente quando o scaling e aplicado, pois torna o modelo malicioso mais distinguivel dos modelos honestos
3. **Trimmed Mean e a estrategia mais consistente**, mantendo performance virtualmente identica em todos os niveis de scaling
4. A distribuicao **Non-IID amplifica a vulnerabilidade** do FedAvg, causando degradacao mais severa e progressiva com o aumento do scaling
5. Para aplicacoes criticas de seguranca como ITS, o uso de **agregacao robusta e mandatorio** - nao opcional

---

> **Diretorio dos resultados:** `results/v2b/`
> **Artefatos para paper:** `results/v2b/paper_artifacts/`
> **Logs dos clientes:** `results/v2b/client_logs/`
