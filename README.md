# Vertical Federated Learning — Agnostic Inference Attack

Implementação experimental do **Agnostic Inference Attack (AIA)** em **Vertical Federated Learning (VFL)**, baseada no trabalho *Privacy Against Agnostic Inference Attacks in Vertical Federated Learning*.

O projeto utiliza o dataset **Bank Marketing**, do UCI Machine Learning Repository, para reproduzir o fluxo de treinamento VFL, treinamento do **Adversary Model (AM)** e reconstrução das features do participante passivo utilizando o algoritmo **Half***.

---

## 1. Visão geral

O objetivo do projeto é investigar um cenário de **Vertical Federated Learning** no qual os dados de uma mesma amostra estão distribuídos verticalmente entre dois participantes:

* **Active Party** — possui as features ativas e o label;
* **Passive Party** — possui as features passivas.

O ataque explora a informação produzida pelo modelo VFL para tentar inferir/reconstruir as features pertencentes à parte passiva.

O fluxo geral da implementação é:

```text
                    Bank Marketing
                           │
                           ▼
                   Remove "duration"
                           │
                           ▼
                       19 features
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
        10 categóricas             9 numéricas
              │                         │
              ▼                         │
     Target Mean Encoding               │
              │                         │
              └────────────┬────────────┘
                           ▼
                    MinMaxScaler
                           │
                           ▼
                 Features ∈ [0,1]
                           │
                           ▼
                     Divisão 14/5
                    ┌──────┴──────┐
                    ▼             ▼
                  Active        Passive
                    14              5
```

Depois do pré-processamento, o experimento segue três níveis principais:

```text
Nível 1 — Baseline VFL

       Y + X
         │
         ▼
        VFL
         │
         ▼
      Accuracy


Nível 2 — Adversary Model

         Y
         │
         ▼
         AM
         │
         ▼
         ĉ
         │
         ▼
  comparar com c real


Nível 3 — Feature Reconstruction Attack

         Y
         │
         ▼
         AM
         │
         ▼
         ĉ
         │
         ▼
         ĉ'
         │
         ▼
       Half*
         │
         ▼
         X̂
         │
         ▼
      MSE(X, X̂)
```

Além disso, é executado um **oracle attack**, utilizando o confidence score real do VFL em vez da estimativa produzida pelo AM:

```text
       c real
         │
         ▼
       Half*
         │
         ▼
         X̂
```

Esse resultado serve como referência para avaliar o limite do ataque quando não existe erro na estimativa do confidence score.

---

# 2. Estrutura do projeto

A implementação foi organizada de forma modular. O antigo código concentrado em `preprocessing_clean.py` foi transformado em módulos independentes, enquanto `VFL.py` passou a funcionar como o **orquestrador principal do experimento**.

```text
.
├── VFL.py
├── adversary_model.py
├── vfl_model.py
├── logit.py
├── metrics.py
├── half.py
│
├── data/
│   └── bankmarketing/
│       └── bank-additional-full.csv
│
└── results/
    ├── attack_results_YYYY-MM-DD_HH-MM-SS.csv
    └── ...
```

### Responsabilidade de cada arquivo

| Arquivo              | Responsabilidade                                      |
| -------------------- | ----------------------------------------------------- |
| `VFL.py`             | Orquestração completa do experimento                  |
| `vfl_model.py`       | Treinamento e inferência do modelo VFL                |
| `adversary_model.py` | Treinamento e inferência do Adversary Model           |
| `logit.py`           | Conversão de probabilidade para logit                 |
| `metrics.py`         | Cálculo das métricas utilizadas                       |
| `half.py`            | Implementação do algoritmo Half*                      |
| `data/`              | Dataset utilizado nos experimentos                    |
| `results/`           | Resultados gerados automaticamente pelos experimentos |

A separação foi feita para evitar que toda a lógica experimental fique concentrada em um único arquivo e para facilitar a manutenção, validação e futura extensão do projeto.

---

# 3. `VFL.py`

O arquivo `VFL.py` é o **ponto de entrada principal** da implementação.

Ele é responsável por coordenar as diferentes etapas do experimento:

```text
Dataset
   │
   ▼
Pré-processamento
   │
   ▼
Feature scenarios
   │
   ▼
Train/Test split
   │
   ▼
VFL
   │
   ▼
Adversary Model
   │
   ▼
Confidence → Logit
   │
   ├───────────────┐
   ▼               ▼
 Half*          Oracle Half*
   │               │
   ▼               ▼
 X̂              X̂_real
   │               │
   └───────┬───────┘
           ▼
        Metrics
           │
           ▼
       CSV results
```

O `VFL.py` também é responsável por executar os **19 cenários diferentes de divisão das features** entre Active e Passive.

Ao final de cada execução, os resultados são armazenados em um arquivo CSV dentro da pasta `results/`.

O nome do arquivo contém um timestamp:

```text
attack_results_2026-09-11_13-30-00.csv
```

Dessa forma, cada execução gera um novo arquivo sem sobrescrever os resultados anteriores.

---

# 4. Pré-processamento

O dataset utilizado é:

```text
data/bankmarketing/bank-additional-full.csv
```

O dataset possui originalmente 20 features, porém a feature `duration` é removida, resultando em **19 features**.

## 4.1 Remoção de `duration`

A feature:

```text
duration
```

é removida antes do treinamento.

Essa remoção segue a metodologia utilizada no experimento de referência.

---

## 4.2 Target

A variável alvo é:

```text
y
```

com a seguinte conversão:

```text
no  → 0
yes → 1
```

---

## 4.3 Features categóricas

As 10 features categóricas são transformadas utilizando **Target Mean Encoding**:

```text
job
marital
education
default
housing
loan
contact
month
day_of_week
poutcome
```

As 9 features numéricas permanecem como valores numéricos:

```text
age
campaign
pdays
previous
emp.var.rate
cons.price.idx
cons.conf.idx
euribor3m
nr.employed
```

---

## 4.4 Normalização

Após o encoding, todas as 19 features são normalizadas utilizando `MinMaxScaler`, colocando os valores no intervalo:

```text
[0, 1]
```

Portanto:

```text
10 categóricas
      +
9 numéricas
      ↓
19 features
      ↓
MinMaxScaler
      ↓
19 features ∈ [0,1]
```

---

# 5. Divisão Active / Passive

Para cada cenário, as 19 features são divididas em:

```text
Active Party  → 14 features
Passive Party → 5 features
```

A Active Party possui também o label `Y`.

Assim:

```text
Active:
    Y + 14 features

Passive:
    5 features
```

São avaliadas **19 diferentes configurações** de divisão, utilizando uma janela de cinco features passivas que percorre circularmente a lista de features.

Isso permite avaliar o comportamento do ataque em diferentes combinações de features passivas.

---

# 6. Divisão dos dados

A divisão dos dados ocorre em duas etapas.

Primeiramente:

```text
Dataset completo
      │
      ├── 80% treinamento
      │
      └── 20% teste
```

O conjunto de treinamento é então dividido novamente:

```text
80% treinamento
      │
      ├── 80% → treinamento efetivo
      │
      └── 20% → conjunto intermediário
```

A divisão utiliza:

```python
random_state=42
stratify=y
```

Essa estrutura de divisão é mantida para reproduzir o procedimento utilizado na implementação experimental.

---

# 7. Modelo VFL

O módulo:

```text
vfl_model.py
```

é responsável pelo treinamento e inferência do modelo VFL.

O modelo utilizado é uma regressão logística:

```python
LogisticRegression(max_iter=1000)
```

O treinamento utiliza as features das duas partes:

```text
Y + X
```

ou, mais precisamente:

```text
[features_active | features_passive]
```

O modelo fornece:

* classificação;
* probabilidades;
* coeficientes;
* bias.

A saída de probabilidade positiva do VFL é denominada:

```text
c
```

e representa o **confidence score real** produzido pelo modelo VFL.

---

# 8. Adversary Model

O módulo:

```text
adversary_model.py
```

implementa o **Adversary Model (AM)**.

O AM representa o cenário no qual o atacante possui acesso às informações da Active Party e tenta estimar o confidence score produzido pelo VFL.

O treinamento do AM utiliza apenas as features da Active Party:

```text
Y
  │
  ▼
 AM
  │
  ▼
ĉ
```

onde:

* `c` = confidence score real produzido pelo VFL;
* `ĉ` = confidence score estimado pelo Adversary Model.

O objetivo do AM é aproximar:

```text
ĉ ≈ c
```

A qualidade dessa aproximação é avaliada através do:

```text
Confidence-score MSE
```

---

# 9. Conversão para Logit

O módulo:

```text
logit.py
```

é responsável pela conversão entre probabilidade e logit.

A probabilidade produzida pelo modelo é:

```text
c
```

e é convertida para:

```text
c' = log(c / (1 - c))
```

Na implementação, as probabilidades são limitadas ao intervalo:

```text
[1e-15, 1 - 1e-15]
```

antes da aplicação da função logit, evitando problemas numéricos associados a:

```text
log(0)
```

O fluxo do ataque passa então a ser:

```text
AM
 │
 ▼
ĉ
 │
 ▼
logit
 │
 ▼
ĉ'
```

O mesmo procedimento é aplicado ao confidence score real para gerar o valor utilizado no ataque oracle.

---

# 10. Half*

O arquivo:

```text
half.py
```

contém a implementação do algoritmo **Half*** utilizada no experimento.

A implementação recebe:

```text
W_pas
W_act
Y_test
ĉ'
b
```

e utiliza os pesos da parte passiva, os pesos da parte ativa, as features ativas, o logit estimado e o bias para reconstruir uma estimativa das features passivas.

O resultado é:

```text
X̂
```

ou seja, uma estimativa das features pertencentes à Passive Party.

O fluxo completo é:

```text
Y_test
   │
   ▼
Adversary Model
   │
   ▼
ĉ
   │
   ▼
Logit
   │
   ▼
ĉ'
   │
   ▼
Half*
   │
   ▼
X̂
```

O `half.py` é mantido como um módulo independente porque sua implementação representa especificamente o algoritmo de reconstrução utilizado no ataque.

---

# 11. Oracle Attack

Além do ataque normal, a implementação executa uma versão oracle.

No ataque normal:

```text
c
 │
 ▼
AM
 │
 ▼
ĉ
 │
 ▼
Logit
 │
 ▼
ĉ'
 │
 ▼
Half*
 │
 ▼
X̂
```

No oracle:

```text
c real
 │
 ▼
Logit
 │
 ▼
c'
 │
 ▼
Half*
 │
 ▼
X̂_real
```

A diferença é que o oracle não possui o erro introduzido pelo Adversary Model.

Portanto, ele serve como uma referência para avaliar o impacto do erro do AM sobre a reconstrução.

---

# 12. Métricas

O módulo:

```text
metrics.py
```

concentra o cálculo das métricas utilizadas na avaliação.

## 12.1 VFL Accuracy

Mede a acurácia do modelo VFL:

```text
VFL Accuracy
```

---

## 12.2 AM Accuracy

Mede a acurácia do Adversary Model:

```text
AM Accuracy
```

---

## 12.3 Confidence-score MSE

Compara o confidence score real com o estimado pelo AM:

```text
MSE(c, ĉ)
```

Essa métrica mede o quanto o AM consegue reproduzir a saída do VFL.

---

## 12.4 Logit MSE

Compara os logits real e estimado:

```text
MSE(c', ĉ')
```

Essa métrica permite avaliar diretamente o erro introduzido antes da etapa Half*.

---

## 12.5 Half* Reconstruction MSE

Compara as features passivas reais com as reconstruídas:

```text
MSE(X, X̂)
```

Essa é a principal métrica utilizada para avaliar a qualidade da reconstrução.

Quanto menor:

```text
MSE(X, X̂)
```

melhor a reconstrução das features passivas.

---

## 12.6 Oracle Reconstruction MSE

Também é calculado:

```text
MSE(X, X̂_real)
```

utilizando o confidence score real.

Esse resultado representa o desempenho do Half* quando o erro do Adversary Model é removido.

---

## 12.7 Attack Gap

É calculada também a diferença entre o ataque utilizando `ĉ` e o oracle:

```text
Attack Gap =
Half* MSE − Oracle Half* MSE
```

Essa métrica permite observar quanto da perda de desempenho da reconstrução pode ser associada ao erro na estimativa do confidence score.

Também é calculado o percentual relativo:

```text
Attack Gap (%) =
Attack Gap / Oracle MSE × 100
```

---

# 13. Resultados armazenados

Cada cenário produz resultados por feature passiva.

O CSV final contém as seguintes informações:

```text
scenario
passive_features
feature
vfl_accuracy
am_accuracy
confidence_mse
logit_mse
half_star_mse
half_star_real_c_mse
attack_gap
attack_gap_percent
weight
abs_weight
feature_mse
feature_variance
feature_std
feature_nrmse
```

Como são avaliados:

```text
19 cenários × 5 features passivas
```

o resultado final contém:

```text
95 linhas
```

---

# 14. Execução

Com o dataset localizado em:

```text
data/bankmarketing/bank-additional-full.csv
```

a execução principal pode ser realizada através de:

```bash
python VFL.py
```

O programa executará os cenários e, ao final, criará automaticamente a pasta:

```text
results/
```

caso ela ainda não exista.

Um novo CSV será então criado:

```text
results/attack_results_YYYY-MM-DD_HH-MM-SS.csv
```

O timestamp permite executar o experimento várias vezes sem sobrescrever os resultados anteriores.

---

# 15. Baseline atual

A implementação atual possui como baseline uma configuração de:

```text
14 Active features
5 Passive features
```

Para a configuração de referência utilizada anteriormente, foram obtidos:

```text
VFL accuracy:              0.9007587253
AM accuracy:               0.8983308042
Confidence-score MSE:      0.0044701060
Half* reconstruction MSE:  0.1137887976
Half* MSE using REAL c:    0.1121233845
```

Esses resultados devem ser considerados a **baseline atual da implementação**.

É importante destacar que essa baseline corresponde a **uma das possíveis alocações 14/5 das 19 features**. O experimento completo avalia as 19 configurações de divisão descritas anteriormente.

---

# 16. Organização conceitual do código

A separação atual pode ser resumida da seguinte forma:

```text
                         VFL.py
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
     vfl_model.py   adversary_model.py   logit.py
          │                │                │
          ▼                ▼                ▼
         VFL              AM             Logit
          │                │                │
          └────────────────┼────────────────┘
                           │
                           ▼
                        half.py
                           │
                           ▼
                        X̂ / X̂real
                           │
                           ▼
                      metrics.py
                           │
                           ▼
                      CSV Results
```

A ideia dessa organização é manter cada componente com uma responsabilidade específica:

### `VFL.py`

Responsável por **orquestrar** o experimento.

### `vfl_model.py`

Responsável exclusivamente pelo **modelo VFL**.

### `adversary_model.py`

Responsável exclusivamente pelo **Adversary Model**.

### `logit.py`

Responsável pela transformação de **probabilidade → logit**.

### `half.py`

Responsável pela **reconstrução das features passivas**.

### `metrics.py`

Responsável pelas **métricas de avaliação**.

Essa estrutura facilita a substituição ou evolução individual dos componentes sem alterar toda a implementação experimental.

---

# 17. Fluxo completo da implementação

Considerando todos os componentes, o pipeline completo é:

```text
                    ┌──────────────────┐
                    │  Bank Marketing  │
                    └────────┬─────────┘
                             │
                             ▼
                    Remove "duration"
                             │
                             ▼
                       19 features
                             │
                             ▼
                   Target Mean Encoding
                             │
                             ▼
                       MinMaxScaler
                             │
                             ▼
                  ┌──────────┴──────────┐
                  │                     │
                  ▼                     ▼
              Active (14)           Passive (5)
                  │                     │
                  └──────────┬──────────┘
                             │
                             ▼
                       Train/Test
                             │
                             ▼
                    ┌────────────────┐
                    │   VFL Model    │
                    └───────┬────────┘
                            │
                            ▼
                            c
                            │
                 ┌──────────┴──────────┐
                 │                     │
                 ▼                     ▼
          Adversary Model          Oracle
                 │                     │
                 ▼                     │
                ĉ                      │
                 │                     │
                 ▼                     │
               Logit                   │
                 │                     │
                 ▼                     ▼
                ĉ'                    c'
                 │                     │
                 ▼                     ▼
               Half*                 Half*
                 │                     │
                 ▼                     ▼
                 X̂                   X̂_real
                 │                     │
                 └──────────┬──────────┘
                            │
                            ▼
                         Metrics
                            │
                            ▼
                      CSV Results
```

---

# 18. Objetivo dos experimentos

A partir dessa implementação, os experimentos podem ser utilizados para analisar principalmente:

1. **Desempenho do VFL**

   * VFL Accuracy.

2. **Capacidade do adversário de reproduzir o confidence score**

   * AM Accuracy;
   * Confidence-score MSE;
   * Logit MSE.

3. **Capacidade de reconstrução das features passivas**

   * Half* Reconstruction MSE;
   * MSE por feature;
   * NRMSE por feature.

4. **Impacto do erro do Adversary Model**

   * comparação entre Half* utilizando `ĉ'` e Half* utilizando `c'` real;
   * Attack Gap;
   * Attack Gap Percentage.

5. **Influência das diferentes partições Active/Passive**

   * comparação dos 19 cenários 14/5;
   * análise individual das features passivas;
   * relação entre pesos do modelo e erro de reconstrução;
   * análise de variância e desvio padrão das features.

---

# 19. Próximos experimentos

A estrutura modular permite adicionar novos experimentos sem modificar significativamente o pipeline principal.

Possíveis extensões incluem:

* avaliação das 19 configurações Active/Passive;
* comparação entre diferentes modelos VFL;
* avaliação de diferentes modelos para o Adversary Model;
* análise mais detalhada do impacto do erro de confiança na reconstrução;
* comparação entre diferentes métodos de normalização;
* análise de diferentes estratégias de particionamento;
* estudos adicionais sobre a relação entre pesos do modelo e reconstrução;
* avaliação de mecanismos de defesa contra o ataque.

---

# 20. Resumo

A implementação está organizada em uma arquitetura modular na qual:

```text
VFL.py
   │
   ├── vfl_model.py
   │      └── VFL
   │
   ├── adversary_model.py
   │      └── AM
   │
   ├── logit.py
   │      └── Probability → Logit
   │
   ├── half.py
   │      └── Feature Reconstruction
   │
   └── metrics.py
          └── Evaluation
```

O fluxo experimental completo é:

```text
Dataset
   ↓
Preprocessing
   ↓
14 Active / 5 Passive
   ↓
VFL
   ↓
Confidence score c
   ↓
Adversary Model
   ↓
Estimated confidence ĉ
   ↓
Logit ĉ'
   ↓
Half*
   ↓
Reconstructed Passive Features X̂
   ↓
Metrics
   ↓
CSV
```

Essa organização separa claramente **pré-processamento, treinamento, ataque, reconstrução e avaliação**, permitindo que cada etapa seja testada e modificada de forma independente.
