# Phase 1b - Relatorio Correcao Krum e Analise de Sensibilidade

> **Data:** 2026-03-14
> **Objetivo:** Corrigir informacao privilegiada no Krum e avaliar sensibilidade a local_epochs

---

## 1. Correcao Aplicada

### Problema
O parametro `num_malicious_clients` do Krum estava configurado como `len(config.malicious_clients)` = 1, fornecendo ao defensor conhecimento exato do numero de adversarios. Isso e irrealista em cenarios reais.

### Correcao
```python
# Antes (privilegiado):
num_malicious_clients=len(config.malicious_clients)  # m=1

# Depois (realista):
num_malicious_clients=0  # m=0, sem conhecimento privilegiado
```

### Impacto na formula Krum
Com `n=3` clientes:
- **Antes (m=1):** score usa `(n - m - 2) = 0` vizinhos mais proximos → scores identicos → selecao arbitraria
- **Depois (m=0):** score usa `(n - 0 - 2) = 1` vizinho mais proximo → scores diferenciados por distancia

---

## 2. Comparativo Krum: m=1 (antigo) vs m=0 (corrigido)

### 2.1 Cenario IID

| Escala | Acc (m=1) | Acc (m=0) | Delta | H2L (m=1) | H2L (m=0) | Delta H2L |
|:------:|:---------:|:---------:|:-----:|:---------:|:---------:|:---------:|
| 1x | 0.9686 | **0.9755** | +0.69pp | 0.41% | 0.41% | = |
| 5x | 0.9784 | 0.9745 | -0.39pp | 0.41% | 0.41% | = |
| 10x | 0.9745 | **0.9784** | +0.39pp | 0.41% | 0.41% | = |
| 20x | 0.9725 | 0.9686 | -0.39pp | 0.41% | 0.41% | = |

**Conclusao IID:** Sem diferenca pratica. Accuracy varia < 0.7pp e H2L permanece identica (1/241).

### 2.2 Cenario Non-IID

| Escala | Acc (m=1) | Acc (m=0) | Delta | H2L (m=1) | H2L (m=0) | Delta H2L |
|:------:|:---------:|:---------:|:-----:|:---------:|:---------:|:---------:|
| 1x | 0.7471 | **0.8049** | **+5.78pp** | 7.02% | **0.00%** | **-7.02pp** |
| 5x | 0.8069 | 0.8039 | -0.30pp | 0.00% | 0.00% | = |
| 10x | 0.8069 | 0.7971 | -0.98pp | 0.00% | 0.00% | = |
| 20x | 0.7951 | 0.7559 | -3.92pp | 0.00% | **6.20%** | **+6.20pp** |

**Conclusao Non-IID:**
- **scale 1x (sem amplificacao):** A correcao **MELHORA** significativamente o Krum (+5.78pp accuracy, H2L de 7.02% → 0.00%). Com m=0 e 1 vizinho, o Krum consegue selecionar o melhor modelo ao inves de uma selecao arbitraria.
- **scale 5x e 10x:** Desempenho praticamente identico. O modelo malicioso amplificado e facilmente detectavel independentemente de m.
- **scale 20x:** **Degradacao** (-3.92pp accuracy, H2L de 0% → 6.20%). Com amplificacao extrema, a diferenca na formula de score pode levar a selecoes subotimas. Ainda assim, o impacto e moderado (15/242 H2L).

### 2.3 Paradoxo Non-IID Revisitado

| Comparacao | m=1 (antigo) | m=0 (corrigido) |
|:-----------|:---:|:---:|
| Acc 1x (baseline) | 0.7471 | 0.8049 |
| Acc 5x | 0.8069 (+5.98pp) | 0.8039 (-0.10pp) |
| Melhora com scaling? | **Sim** (paradoxo) | **Nao** (comportamento esperado) |

> O paradoxo de Krum melhorar sob ataque com scaling **desaparece** com a correcao m=0. O resultado anterior era um artefato da formula degenerada com 0 vizinhos (m=1, n=3), que causava selecao arbitraria no baseline. Com m=0, o baseline ja seleciona o melhor modelo, e o scaling nao melhora a accuracy.

---

## 3. Analise de Sensibilidade: local_epochs=1 vs local_epochs=5

### 3.1 Resultados Comparativos

| Config | e=5, Acc | e=1, Acc | Delta | e=5, H2L | e=1, H2L | Delta H2L |
|:-------|:--------:|:--------:|:-----:|:--------:|:--------:|:---------:|
| IID/FedAvg/5x | 0.5784 | **0.7049** | **+12.65pp** | **100.0%** | 49.79% | **-50.21pp** |
| IID/Krum/10x | 0.9784 | 0.9667 | -1.17pp | 0.41% | 0.41% | = |
| IID/TM/10x | 0.9775 | 0.9676 | -0.99pp | 0.41% | 0.41% | = |
| NonIID/FedAvg/5x | 0.4902 | **0.5520** | **+6.18pp** | 50.83% | **83.88%** | **+33.05pp** |
| NonIID/Krum/10x | 0.8069 | 0.7990 | -0.79pp | 0.00% | 0.00% | = |
| NonIID/TM/10x | 0.7843 | 0.7971 | +1.28pp | 1.24% | **6.61%** | **+5.37pp** |

### 3.2 Interpretacao

#### FedAvg com local_epochs=1
- **IID:** Paradoxalmente, o FedAvg fica **menos** vulneravel com e=1 (H2L cai de 100% para 49.79%). Com menos epocas locais, o modelo malicioso diverge menos do global, e sua atualizacao envenenada e mais moderada.
- **Non-IID:** O FedAvg fica **mais** vulneravel na metrica H2L (50.83% → 83.88%), mas a accuracy global melhora ligeiramente. O ataque se concentra mais efetivamente na classe High.

#### Krum e Trimmed Mean com local_epochs=1
- **Krum:** Praticamente invariante. Accuracy cai ~1pp (convergencia mais lenta), mas H2L permanece identica.
- **Trimmed Mean IID:** Invariante.
- **Trimmed Mean Non-IID:** Ligeira degradacao (H2L 1.24% → 6.61%). Com epocas reduzidas, as atualizacoes dos clientes sao mais similares, dificultando o trimming.

### 3.3 Insight Chave

> O parametro `local_epochs` tem efeito duplo: mais epocas **amplificam** a divergencia maliciosa mas tambem **fortalecem** o sinal de treino honesto. Com e=1, o ataque e mais sutil (mais dificil de detectar por defesas), mas tambem menos potente (menor magnitude da atualizacao envenenada). Este trade-off explica os resultados contraditarios entre IID e Non-IID.

---

## 4. Tabela Consolidada Final (Resultados Corretos)

### 4.1 Todos os resultados Phase 1b com Krum corrigido (m=0)

| Dist | Estrategia | Escala | Accuracy | Acc High | H2L Rate | H2L Count |
|:----:|:-----------|:------:|:--------:|:--------:|:--------:|:---------:|
| IID | FedAvg | 1x | 0.9529 | 0.846 | 4.15% | 10/241 |
| IID | FedAvg | 5x | 0.5784 | 0.000 | **100.0%** | 241/241 |
| IID | FedAvg | 10x | 0.4157 | 0.000 | **100.0%** | 241/241 |
| IID | FedAvg | 20x | 0.6157 | 0.000 | **90.87%** | 219/241 |
| IID | Krum | 1x | 0.9755 | 0.996 | 0.41% | 1/241 |
| IID | Krum | 5x | 0.9745 | 0.996 | 0.41% | 1/241 |
| IID | Krum | 10x | 0.9784 | 0.996 | 0.41% | 1/241 |
| IID | Krum | 20x | 0.9686 | 0.946 | 0.41% | 1/241 |
| IID | Trimmed Mean | 1x | 0.9755 | 0.963 | 0.41% | 1/241 |
| IID | Trimmed Mean | 5x | 0.9775 | 0.967 | 0.41% | 1/241 |
| IID | Trimmed Mean | 10x | 0.9775 | 0.967 | 0.41% | 1/241 |
| IID | Trimmed Mean | 20x | 0.9775 | 0.967 | 0.41% | 1/241 |
| NonIID | FedAvg | 1x | 0.7824 | 0.583 | 5.79% | 14/242 |
| NonIID | FedAvg | 5x | 0.4902 | 0.426 | **50.83%** | 123/242 |
| NonIID | FedAvg | 10x | 0.4245 | 0.000 | **100.0%** | 242/242 |
| NonIID | FedAvg | 20x | 0.3696 | 0.000 | **100.0%** | 242/242 |
| NonIID | Krum | 1x | 0.8049 | 0.694 | 0.00% | 0/242 |
| NonIID | Krum | 5x | 0.8039 | 0.694 | 0.00% | 0/242 |
| NonIID | Krum | 10x | 0.7971 | 0.694 | 0.00% | 0/242 |
| NonIID | Krum | 20x | 0.7559 | 0.579 | 6.20% | 15/242 |
| NonIID | Trimmed Mean | 1x | 0.7824 | 0.583 | 0.83% | 2/242 |
| NonIID | Trimmed Mean | 5x | 0.7824 | 0.587 | 1.24% | 3/242 |
| NonIID | Trimmed Mean | 10x | 0.7843 | 0.591 | 1.24% | 3/242 |
| NonIID | Trimmed Mean | 20x | 0.7833 | 0.591 | 1.24% | 3/242 |

### 4.2 Analise de Sensibilidade (local_epochs=1)

| Config | Acc (e=5) | Acc (e=1) | H2L (e=5) | H2L (e=1) |
|:-------|:---------:|:---------:|:---------:|:---------:|
| IID/FedAvg/5x | 0.5784 | 0.7049 | 100.0% | 49.79% |
| IID/Krum/10x | 0.9784 | 0.9667 | 0.41% | 0.41% |
| IID/TM/10x | 0.9775 | 0.9676 | 0.41% | 0.41% |
| NonIID/FedAvg/5x | 0.4902 | 0.5520 | 50.83% | 83.88% |
| NonIID/Krum/10x | 0.8069 | 0.7990 | 0.00% | 0.00% |
| NonIID/TM/10x | 0.7843 | 0.7971 | 1.24% | 6.61% |

---

## 5. Conclusoes Atualizadas

1. **A correcao m=0 nao afeta a conclusao principal:** Krum continua robusto contra label-flipping com model scaling. A H2L Rate permanece negligivel na grande maioria dos cenarios.

2. **O paradoxo Non-IID do Krum era um artefato:** Com m=1 e n=3, a formula de Krum degenerava (0 vizinhos). O comportamento paradoxal (melhoria sob ataque) desaparece com m=0.

3. **Krum com m=0 e scale 20x Non-IID mostra leve vulnerabilidade** (H2L=6.20%), sugerindo que escalas extremas podem contornar parcialmente a defesa mesmo sem informacao privilegiada. Isso e um resultado mais realista e informativo para o paper.

4. **A sensibilidade a local_epochs e moderada:** Krum e robusto independentemente de e. FedAvg mostra um trade-off complexo onde e=1 pode ser pior (NonIID) ou melhor (IID) dependendo da distribuicao.

5. **Trimmed Mean com e=1 no NonIID apresenta leve degradacao** (H2L 1.24% → 6.61%), indicando que o trimming e menos eficaz quando as atualizacoes dos clientes sao mais similares.

---

## 6. Diretorios dos Resultados

| Conteudo | Diretorio |
|:---------|:----------|
| Phase 1b original (FedAvg + TM validos) | `results/v2b/` |
| Krum corrigido (m=0) | `results/v2b_krum_fixed/` |
| Sensibilidade local_epochs=1 | `results/v2b_sensitivity_epochs/` |
