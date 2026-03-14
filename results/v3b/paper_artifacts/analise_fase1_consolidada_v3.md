# Fase 1 Consolidada (v3) — Resultados Definitivos

> **Data de execucao:** 2026-03-14
> **Versao:** v3 (scaler corrigido, Krum m=0, FedProx incluso em Phase 1b)
> **Correcoes aplicadas:** (1) StandardScaler fit apenas no treino, (2) Krum num_malicious_clients=0, (3) FedProx adicionado a Phase 1b

---

## 1. Correcoes Aplicadas nesta Versao

### 1.1 StandardScaler — Data Leakage Corrigido

**Antes (v2):** `scaler.fit_transform(all_data)` seguido de `train_test_split()` — o scaler via estatisticas do teste.

**Depois (v3):** `train_test_split()` primeiro, depois `scaler.fit_transform(X_train)` e `scaler.transform(X_test)`.

### 1.2 Krum — Informacao Privilegiada Removida

**Antes (v2):** `num_malicious_clients=1` — o defensor sabia quantos adversarios existiam.

**Depois (v3):** `num_malicious_clients=0` — sem informacao privilegiada. Score por 1 vizinho mais proximo (n-0-2=1).

### 1.3 FedProx — Adicionado a Phase 1b

**Antes (v2b):** 24 configs (FedAvg, Krum, TM × 4 scales × 2 dists).

**Depois (v3b):** 32 configs (FedAvg, **FedProx**, Krum, TM × 4 scales × 2 dists).

---

## 2. Impacto da Correcao do Scaler (v2 vs v3)

### 2.1 Comparativo de configs-chave

| Config | v2 Acc | v3 Acc | Delta | v2 H2L | v3 H2L | Delta H2L |
|:-------|:------:|:------:|:-----:|:------:|:------:|:---------:|
| IID/FedAvg/no_attack | 0.9784 | 0.9775 | -0.09pp | 0.41% | 0.41% | = |
| IID/FedAvg/scale1x | 0.9529 | 0.9510 | -0.19pp | 4.15% | 4.15% | = |
| IID/FedAvg/scale5x | 0.5784 | 0.5784 | 0.00pp | 100.0% | 100.0% | = |
| IID/FedAvg/scale10x | 0.4157 | 0.3696 | -4.61pp | 100.0% | 100.0% | = |
| IID/Krum/scale10x (m=0) | 0.9784 | 0.9676 | -1.08pp | 0.41% | 0.41% | = |
| IID/TM/scale10x | 0.9775 | 0.9755 | -0.20pp | 0.41% | 0.41% | = |
| NonIID/FedAvg/scale10x | 0.4245 | 0.4029 | -2.16pp | 100.0% | 100.0% | = |
| NonIID/Krum/scale10x (m=0) | 0.7971 | 0.7441 | -5.30pp | 0.00% | 8.26% | +8.26pp |
| NonIID/TM/scale10x | 0.7843 | 0.7843 | 0.00pp | 1.24% | 1.65% | +0.41pp |

> **Conclusao:** A correcao do scaler causa queda de 0-5pp na accuracy, confirmando o data leakage estimado. As conclusoes qualitativas se mantem: FedAvg colapsa, defesas funcionam. A maior mudanca e no Krum NonIID/10x, onde H2L sobe de 0% para 8.26% — resultado mais realista.

---

## 3. Phase 1a — Label-flipping puro (24 configs)

### 3.1 Resultados

| Dist | Ataque | FedAvg Acc | FedProx Acc | Krum Acc | TM Acc |
|:----:|:------:|:----------:|:-----------:|:--------:|:------:|
| IID | none | 0.9775 | 0.9775 | 0.9735 | 0.9775 |
| IID | flip 20% | 0.9765 | 0.9775 | 0.9676 | 0.9775 |
| IID | flip 50% | 0.9775 | 0.9755 | 0.9706 | 0.9755 |
| NonIID | none | 0.7784 | 0.7814 | 0.7461 | **0.8010** |
| NonIID | flip 20% | 0.7804 | 0.7794 | 0.7402 | **0.8020** |
| NonIID | flip 50% | 0.7824 | 0.7794 | 0.7451 | **0.7961** |

### 3.2 H2L Rate (Phase 1a)

| Dist | Ataque | FedAvg | FedProx | Krum | TM |
|:----:|:------:|:------:|:-------:|:----:|:--:|
| IID | none | 0.41% | 0.41% | 0.41% | 0.41% |
| IID | flip 20% | 0.41% | 0.41% | 0.41% | 0.41% |
| IID | flip 50% | 0.41% | 0.41% | 0.41% | 0.41% |
| NonIID | none | 0.00% | 0.00% | **7.85%** | 0.00% |
| NonIID | flip 20% | 0.00% | 0.00% | **7.85%** | 0.00% |
| NonIID | flip 50% | 0.00% | 0.00% | **7.85%** | 0.00% |

### 3.3 Conclusao Phase 1a

Label-flipping puro (20%, 50%) com 1/3 dos clientes **nao produz degradacao mensuravel**. Confirmacao robusta apos correcao do scaler. O Krum apresenta H2L intrinseco de 7.85% em NonIID mesmo SEM ataque — isso e artefato da selecao por distancia em distribuicao heterogenea.

---

## 4. Phase 1b — Gradient Scaling (32 configs)

### 4.1 IID

| Estrategia | Scale | Accuracy | F1 | Acc High | H2L Rate | H2L Count |
|:-----------|:-----:|:--------:|:--:|:--------:|:--------:|:---------:|
| FedAvg | 1x | 0.9510 | 0.9455 | 0.842 | 4.15% | 10/241 |
| FedAvg | 5x | 0.5784 | 0.4435 | 0.000 | **100.0%** | 241/241 |
| FedAvg | 10x | 0.3696 | 0.1798 | 0.000 | **100.0%** | 241/241 |
| FedAvg | 20x | 0.5422 | 0.4085 | 0.000 | **99.17%** | 239/241 |
| FedProx | 1x | 0.9588 | 0.9555 | 0.876 | 2.07% | 5/241 |
| FedProx | 5x | 0.6118 | 0.4707 | 0.000 | **100.0%** | 241/241 |
| FedProx | 10x | 0.2892 | 0.1496 | 0.000 | **99.59%** | 240/241 |
| FedProx | 20x | 0.4676 | 0.3261 | 0.000 | **99.59%** | 240/241 |
| Krum | 1x | 0.9735 | 0.9741 | 0.996 | 0.41% | 1/241 |
| Krum | 5x | 0.9676 | 0.9673 | 0.942 | 0.41% | 1/241 |
| Krum | 10x | 0.9676 | 0.9667 | 0.929 | 0.41% | 1/241 |
| Krum | 20x | 0.9725 | 0.9732 | 0.996 | 0.41% | 1/241 |
| TM | 1x | 0.9745 | 0.9747 | 0.963 | 0.41% | 1/241 |
| TM | 5x | 0.9775 | 0.9772 | 0.963 | 0.41% | 1/241 |
| TM | 10x | 0.9755 | 0.9755 | 0.963 | 0.41% | 1/241 |
| TM | 20x | 0.9765 | 0.9764 | 0.963 | 0.41% | 1/241 |

### 4.2 Non-IID

| Estrategia | Scale | Accuracy | F1 | Acc High | H2L Rate | H2L Count |
|:-----------|:-----:|:--------:|:--:|:--------:|:--------:|:---------:|
| FedAvg | 1x | 0.7833 | 0.7799 | 0.574 | 5.79% | 14/242 |
| FedAvg | 5x | 0.4941 | 0.5348 | 0.413 | **53.31%** | 129/242 |
| FedAvg | 10x | 0.4029 | 0.2263 | 0.000 | **100.0%** | 242/242 |
| FedAvg | 20x | 0.3706 | 0.1775 | 0.004 | **99.59%** | 241/242 |
| FedProx | 1x | 0.7863 | 0.7844 | 0.599 | 1.65% | 4/242 |
| FedProx | 5x | 0.4088 | 0.3014 | 0.000 | **91.74%** | 222/242 |
| FedProx | 10x | 0.4549 | 0.2950 | 0.000 | **99.59%** | 241/242 |
| FedProx | 20x | 0.3696 | 0.1734 | 0.000 | **100.0%** | 242/242 |
| Krum | 1x | 0.7461 | 0.7488 | 0.566 | 7.85% | 19/242 |
| Krum | 5x | **0.8049** | 0.8072 | 0.698 | **0.00%** | 0/242 |
| Krum | 10x | 0.7441 | 0.7467 | 0.570 | 8.26% | 20/242 |
| Krum | 20x | 0.7549 | 0.7581 | 0.587 | 6.61% | 16/242 |
| TM | 1x | 0.7775 | 0.7789 | 0.583 | 1.65% | 4/242 |
| TM | 5x | 0.7824 | 0.7853 | 0.591 | 1.65% | 4/242 |
| TM | 10x | 0.7843 | 0.7863 | 0.587 | 1.65% | 4/242 |
| TM | 20x | 0.7843 | 0.7865 | 0.587 | 1.65% | 4/242 |

---

## 5. FedProx sob Gradient Scaling — Novo Resultado

### 5.1 FedProx vs FedAvg

| Config | FedAvg Acc | FedProx Acc | FedAvg H2L | FedProx H2L |
|:-------|:---------:|:----------:|:---------:|:----------:|
| IID/1x | 0.9510 | 0.9588 | 4.15% | 2.07% |
| IID/5x | 0.5784 | 0.6118 | 100.0% | 100.0% |
| IID/10x | 0.3696 | 0.2892 | 100.0% | 99.59% |
| IID/20x | 0.5422 | 0.4676 | 99.17% | 99.59% |
| NonIID/1x | 0.7833 | 0.7863 | 5.79% | 1.65% |
| NonIID/5x | 0.4941 | 0.4088 | 53.31% | **91.74%** |
| NonIID/10x | 0.4029 | 0.4549 | 100.0% | 99.59% |
| NonIID/20x | 0.3706 | 0.3696 | 99.59% | 100.0% |

### 5.2 Conclusao FedProx

> **FedProx NAO oferece protecao contra gradient scaling.** Comportamento muito similar ao FedAvg: H2L ≥ 99% para scales ≥10x. Em NonIID/5x, FedProx e **pior** que FedAvg (H2L 91.74% vs 53.31%). A regularizacao proximal (mu=0.1) nao e mecanismo de defesa contra model poisoning — confirma a decisao de excluir FedProx na versao original, mas agora com dados concretos.

---

## 6. Analise das Defesas (v3 definitivo)

### 6.1 Krum (m=0) — Perfil Atualizado

```
IID:    H2L constante em 0.41% (1/241) em TODOS os scales — defesa perfeita
NonIID: H2L entre 0-8.26%, com variacao por scale:
  1x:  7.85% (19/242) — H2L intrinseco (nao causado pelo ataque)
  5x:  0.00% (0/242)  — scaling moderado facilmente rejeitado
  10x: 8.26% (20/242) — similar ao baseline, scaling alto causa instabilidade
  20x: 6.61% (16/242) — idem
```

**Observacao importante sobre NonIID:** O Krum tem H2L de ~7.85% mesmo SEM ataque (Phase 1a). Isso e intrinseco ao algoritmo de selecao por distancia em distribuicoes heterogeneas — nao e causado pelo ataque. O scale 5x paradoxalmente melhora porque o modelo malicioso amplificado e mais facilmente rejeitado.

### 6.2 Trimmed Mean — Defesa mais estavel

```
IID:    H2L constante em 0.41% (1/241) em TODOS os scales
NonIID: H2L constante em 1.65% (4/242) em TODOS os scales
```

**TM e a defesa mais previsivel.** H2L completamente estavel independentemente do fator de escala. Perde em accuracy pura para Krum em NonIID/5x (78.24% vs 80.49%) mas oferece garantia de consistencia.

### 6.3 Comparativo Final de Defesas (NonIID, scale 10x)

| Metrica | FedAvg | FedProx | Krum (m=0) | TM |
|:--------|:------:|:-------:|:----------:|:--:|
| Accuracy | 40.29% | 45.49% | **74.41%** | **78.43%** |
| H2L Rate | 100.0% | 99.59% | 8.26% | **1.65%** |
| Acc High | 0.0% | 0.0% | 57.0% | **58.7%** |
| Defense Recovery | — | — | 91.74% | **98.35%** |

---

## 7. Analise de Sensibilidade (local_epochs=1)

### 7.1 Resultados Comparativos (e=5 vs e=1)

| Config | Acc (e=5) | Acc (e=1) | Delta | H2L (e=5) | H2L (e=1) | Delta H2L |
|:-------|:---------:|:---------:|:-----:|:---------:|:---------:|:---------:|
| IID/FedAvg/5x | 0.5784 | **0.7069** | +12.85pp | 100.0% | **49.38%** | -50.62pp |
| IID/FedProx/5x | 0.6118 | 0.6961 | +8.43pp | 100.0% | **61.00%** | -39.00pp |
| IID/Krum/10x | 0.9676 | 0.9608 | -0.68pp | 0.41% | 0.41% | = |
| IID/TM/10x | 0.9755 | 0.9676 | -0.79pp | 0.41% | 0.41% | = |
| NonIID/FedAvg/5x | 0.4941 | **0.5539** | +5.98pp | 53.31% | **84.71%** | +31.40pp |
| NonIID/FedProx/5x | 0.4088 | 0.4971 | +8.83pp | 91.74% | 77.69% | -14.05pp |
| NonIID/Krum/10x | 0.7441 | 0.8020 | +5.79pp | 8.26% | **0.00%** | -8.26pp |
| NonIID/TM/10x | 0.7843 | 0.7951 | +1.08pp | 1.65% | **6.61%** | +4.96pp |

### 7.2 Interpretacao

**FedAvg/FedProx com e=1:**
- IID: ataque **menos** eficaz com e=1 (H2L cai ~50pp). Menos epocas = atualizacao maliciosa mais moderada.
- NonIID FedAvg: ataque **mais** direcionado com e=1 (H2L sobe 53%→85%). O ataque se concentra na classe High.

**Krum com e=1:**
- Invariante ao parametro. H2L 0.41% (IID) e 0.00% (NonIID). Defesa robusta independentemente da intensidade do treino local.
- NonIID com e=1 tem accuracy melhor (80.20% vs 74.41%) — com menos divergencia local, a selecao e mais estavel.

**TM com e=1:**
- IID: invariante.
- NonIID: leve degradacao (H2L 1.65% → 6.61%). Com atualizacoes mais similares, o trimming tem menor margem.

---

## 8. Comparativo Consolidado: Robustez das Estrategias

```
                    FedAvg        FedProx       Krum          Trimmed Mean
                 IID   Non-IID   IID  Non-IID   IID  Non-IID   IID   Non-IID
Sem ataque(1x)   ★★★★  ★★★     ★★★★  ★★★     ★★★★★ ★★★      ★★★★★  ★★★★
Scale 5x          ★     ★★       ★★    ★       ★★★★★ ★★★★★   ★★★★★  ★★★★
Scale 10x         ★     ★        ★     ★       ★★★★★ ★★★      ★★★★★  ★★★★
Scale 20x         ★★    ★        ★     ★       ★★★★★ ★★★      ★★★★★  ★★★★

Legenda: ★ = Muito Ruim | ★★★ = Aceitavel | ★★★★★ = Excelente
```

---

## 9. Conclusoes Definitivas (v3)

### 9.1 Resultados Confirmados

1. **FedAvg e catastroficamente vulneravel** a gradient scaling ≥5x. H2L = 100% em IID/5x e NonIID/10x+. Resultado robusto, nao afetado por nenhuma correcao.

2. **FedProx NAO e defesa.** Comportamento identico ao FedAvg sob scaling. Em NonIID/5x, e **pior** que FedAvg (H2L 91.74% vs 53.31%). A regularizacao proximal nao neutraliza atualizacoes maliciosas amplificadas.

3. **Trimmed Mean e a defesa mais consistente.** H2L ≤ 1.65% em TODOS os 32 cenarios. Accuracy estavel (97.45-97.75% IID, 77.75-78.43% NonIID). Recomendado como default para deployments ITS.

4. **Krum tem perfil complementar.** Excelente em IID (H2L = 0.41% constante). Em NonIID, H2L intrinseco de ~7-8% mesmo sem ataque. Melhor accuracy que TM em NonIID/5x (80.49% vs 78.24%). Escolha adequada quando accuracy e prioritaria e NonIID e moderado.

5. **Label-flipping puro e ineficaz.** Confirmado em 24 configs sem nenhuma degradacao mensuravel.

### 9.2 Resultados Novos (nao presentes em v2)

- **FedProx sob scaling:** Vulneravel como FedAvg. Dados concretos para justificar sua exclusao.
- **Krum NonIID com scaler correto:** H2L sobe de 0% (v2) para ~8% (v3) em scales 10x/20x. Resultado mais realista.
- **Impacto real do data leakage:** Confirmado em <5pp na maioria dos cenarios. Nao muda conclusoes.

### 9.3 Recomendacao para ITS

| Cenario | Estrategia Recomendada | Razao |
|:--------|:----------------------:|:------|
| Default (producao) | **Trimmed Mean** | Consistencia maxima, H2L ≤ 1.65% |
| NonIID com accuracy prioritaria | **Krum** | Melhor accuracy em NonIID moderado |
| Sem ameaca conhecida | FedAvg | Performance otima sem overhead |
| FedProx | **Nao recomendado** | Sem beneficio defensivo, overhead desnecessario |

---

## 10. Diretorios dos Resultados

| Conteudo | Diretorio | Configs | Status |
|:---------|:----------|:-------:|:------:|
| Phase 1a (v3) | `results/v3/` | 24 | Definitivo |
| Phase 1b (v3) | `results/v3b/` | 32 | Definitivo |
| Sensibilidade e=1 (v3) | `results/v3b_sensitivity_epochs/` | 8 | Definitivo |
| Quick validation (v3) | `results/v3b_quick/` | 4 | Validacao |
| Phase 1a (v2, leaky) | `results/v2/` | 24 | Descartado |
| Phase 1b (v2, leaky) | `results/v2b/` | 24 | Descartado |
| Krum rerun (v2, leaky) | `results/v2b_krum_fixed/` | 8 | Descartado |
| Sensibilidade (v2, leaky) | `results/v2b_sensitivity_epochs/` | 6 | Descartado |

---

## 11. Configuracao Experimental

| Parametro | Valor |
|:----------|:------|
| Modelo | GRU (64→32), ~17K parametros |
| Clientes | 3 RSUs |
| Rounds | 10 |
| Epocas locais | 5 (e=1 na sensibilidade) |
| Batch size | 32 |
| Dataset | AIMS ITS — 5.093 amostras, 28 features, 3 classes |
| Framework | Flower 1.x + TensorFlow/Keras |
| Seed | 42 (+ client_id) |
| Cliente malicioso | Client 0 (RSU 0) |
| FedProx mu | 0.1 |
| Krum | m=0, k=0 (selecao classica) |
| TM beta | 0.34 (mediana coordenada com 3 clientes) |
| StandardScaler | Fit no treino, transform em treino+teste |
