# Agnostic Inference Attack — VFL / Half*

## 1. Contexto

Este projeto busca reproduzir, sobre o dataset **Bank Marketing**, a abordagem de ataque apresentada no artigo:

> *Privacy Against Agnostic Inference Attacks in Vertical Federated Learning*

O objetivo é estudar um **Agnostic Inference Attack** em Vertical Federated Learning (VFL), no qual um adversário estima o confidence score produzido pelo modelo VFL usando somente as features ativas disponíveis para ele. A partir dessa estimativa, é utilizado o **Half\*** para investigar a reconstrução das features passivas.

> **Estado atual:** a implementação de `Half` foi deixada de lado temporariamente. Os experimentos atuais utilizam somente `Half*`.

---

## 2. Dataset

Arquivo utilizado:

`data/bankmarketing/bank-additional-full.csv`

- Separador: `;`
- 41.188 registros
- 20 atributos originais
- `duration` foi removida
- 19 features restantes
- Target: `no → 0` e `yes → 1`

---

## 3. Pré-processamento

O pré-processamento foi ajustado para seguir a metodologia descrita no artigo.

### Target Mean Encoding

Features categóricas:

- `job`
- `marital`
- `education`
- `default`
- `housing`
- `loan`
- `contact`
- `month`
- `day_of_week`
- `poutcome`

Features numéricas mantidas:

- `age`
- `campaign`
- `pdays`
- `previous`
- `emp.var.rate`
- `cons.price.idx`
- `cons.conf.idx`
- `euribor3m`
- `nr.employed`

Após o encoding, foi aplicado `MinMaxScaler` para o intervalo `[0, 1]` antes da divisão dos dados, conforme a metodologia utilizada no artigo.

---

## 4. Ordem das features

| Índice | Feature |
|---:|---|
| 0 | `age` |
| 1 | `job` |
| 2 | `marital` |
| 3 | `education` |
| 4 | `default` |
| 5 | `housing` |
| 6 | `loan` |
| 7 | `contact` |
| 8 | `month` |
| 9 | `day_of_week` |
| 10 | `campaign` |
| 11 | `pdays` |
| 12 | `previous` |
| 13 | `poutcome` |
| 14 | `emp.var.rate` |
| 15 | `cons.price.idx` |
| 16 | `cons.conf.idx` |
| 17 | `euribor3m` |
| 18 | `nr.employed` |

Total: **19 features**, sendo **14 ativas** e **5 passivas**.

---

## 5. Divisão dos dados

Foi utilizada uma divisão 80/20 e, sobre a parte de treino, uma segunda divisão 80/20 para os conjuntos utilizados no experimento VFL.

Parâmetros:

```python
random_state=42
stratify=y
```

---

## 6. Modelo VFL

Para cada cenário:

- `Y` = features ativas
- `X` = features passivas

O modelo VFL é uma `LogisticRegression(max_iter=1000)` treinada com as 19 features.

O confidence score real é obtido por:

```python
c_real = vfl_model.predict_proba(
    np.hstack((Y_test, X_test))
)[:, 1]
```

---

## 7. Adversary Model (AM)

O AM utiliza somente as 14 features ativas para estimar o confidence score do VFL.

Também é utilizada:

```python
LogisticRegression(max_iter=1000)
```

O confidence score estimado é:

```python
c_hat = am_model.predict_proba(Y_test)[:, 1]
```

O erro de confidence score é:

```python
confidence_mse = np.mean((c_real - c_hat) ** 2)
```

Também são calculados os logits `z_real` e `z_hat`, com clipping das probabilidades para evitar problemas numéricos.

---

## 8. Half*

Os pesos do modelo VFL são separados em:

```text
W_act = pesos das features ativas
W_pas = pesos das features passivas
```

A reconstrução é realizada usando:

- `W_pas`
- `W_act`
- `Y_test`
- `z_hat`
- bias do VFL

A reconstrução gera `X_hat`.

---

## 9. Reconstrução Oracle

Para separar o erro provocado pelo AM do erro inerente ao Half*, também foi executada a reconstrução utilizando o logit real:

```text
Half*(z_hat) → ataque completo
Half*(z_real) → cenário oracle
```

Isso permite observar quanto o erro adicional de reconstrução está relacionado à estimativa do AM.

---

## 10. Experimento com 19 cenários

Foram executados **19 cenários**, utilizando uma janela móvel de 5 features passivas sobre as 19 features disponíveis.

Em todos os cenários:

```text
5 features passivas
14 features ativas
```

Exemplos:

**Cenário 1**

```text
age, job, marital, education, default
```

**Cenário 2**

```text
job, marital, education, default, housing
```

**Cenário 19**

```text
nr.employed, age, job, marital, education
```

---

## 11. Resultados agregados

A média dos 19 cenários foi:

| Métrica | Média |
|---|---:|
| Half* MSE | **0.1612160844** |
| Half* MSE usando `z_real` | **0.1551121720** |
| Diferença | **0.0061039124** |
| Diferença relativa | **~3,94%** |

Assim, substituir o logit real pelo logit estimado pelo AM aumentou o MSE médio da reconstrução em aproximadamente **3,94%**.

Isso indica que, nos experimentos realizados, uma parcela relativamente pequena do erro total de reconstrução é explicada pela imperfeição da estimativa do AM.

---

## 12. Accuracy

A accuracy do VFL permaneceu próxima de:

```text
~0.90
```

nos diferentes cenários.

A accuracy do AM variou aproximadamente entre:

```text
0.8857 e 0.8997
```

A accuracy, entretanto, não é suficiente para caracterizar a qualidade da reconstrução, pois o ataque depende do confidence score/logit e não somente da classe prevista.

---

## 13. Resultados por feature

| Feature | Mean MSE | Mean NRMSE | Std | |Weight| |
|---|---:|---:|---:|---:|
| `pdays` | 0.287664 | 2.844532 | 0.187098 | 1.368350 |
| `poutcome` | 0.242278 | 2.732043 | 0.180044 | 0.362018 |
| `loan` | 0.223982 | 1.545465 | 0.306225 | 0.031333 |
| `housing` | 0.214057 | 1.010569 | 0.457824 | 0.017779 |
| `default` | 0.212701 | 1.875811 | 0.243910 | 0.477397 |
| `month` | 0.201647 | 2.254174 | 0.196661 | 0.936115 |
| `contact` | 0.201457 | 0.929584 | 0.481507 | 0.829221 |
| `previous` | 0.197357 | 6.177145 | 0.070700 | 1.643118 |
| `marital` | 0.190277 | 1.224153 | 0.356280 | 0.154447 |
| `euribor3m` | 0.173667 | 1.058367 | 0.393210 | 0.617869 |
| `nr.employed` | 0.159428 | 1.456546 | 0.273163 | 0.850902 |
| `campaign` | 0.155228 | **7.619211** | 0.050364 | 1.666093 |
| `day_of_week` | 0.145738 | 1.039991 | 0.367018 | 0.218064 |
| `job` | 0.114238 | 1.712529 | 0.196872 | 0.417443 |
| `emp.var.rate` | 0.092728 | 0.922171 | 0.327283 | 3.121859 |
| `education` | 0.087017 | 1.929678 | 0.150366 | 0.402520 |
| `cons.price.idx` | 0.069822 | 1.126737 | 0.225581 | 3.101683 |
| `age` | 0.054379 | 1.808089 | 0.128657 | 0.263407 |
| `cons.conf.idx` | 0.039440 | **1.025004** | 0.193648 | 0.992895 |

---

## 14. MSE absoluto vs. NRMSE

Foi observado que o MSE absoluto pode produzir uma interpretação diferente do erro relativo.

O NRMSE utilizado foi:

```text
NRMSE = sqrt(MSE) / std
```

Exemplo:

```text
campaign
MSE     = 0.155228
std     = 0.050364
NRMSE   = 7.619211
```

Enquanto:

```text
contact
MSE     = 0.201457
std     = 0.481507
NRMSE   = 0.929584
```

Portanto, `campaign` possui MSE absoluto menor que `contact`, mas erro muito maior em relação à sua própria variabilidade.

---

## 15. Correlações analisadas

Foram testadas várias relações.

| Relação | Correlação |
|---|---:|
| `abs_weight` × `feature_mse` | **-0.259767** |
| `confidence_mse` × `Half* MSE` | **0.158505** |
| `max_active_corr` × `feature_mse` | **0.028611** |
| `1 / |W|` × `feature_mse` | **0.240411** |
| `1 / W²` × `feature_mse` | **0.217220** |
| `logit_mse` × `feature_mse` | **0.082306** |
| `predicted_mse` × `feature_mse` | **0.195194** |
| `feature_variance` × `feature_mse` | **0.206875** |
| `feature_variance` × `relative_mse` | **-0.406675** |
| `std` × `feature_mse` | **0.185904** |
| `std` × `NRMSE` | **-0.675659** |

O resultado mais forte foi:

```text
std × NRMSE = -0.675659
```

---

## 16. Principais observações

### 16.1 Peso do modelo

A relação entre `abs_weight` e MSE foi fraca:

```text
-0.259767
```

Também foram testadas `1/|W|` e `1/W²`, mas as correlações permaneceram relativamente fracas.

Portanto, o módulo do peso, isoladamente, não parece explicar a maior parte da variação do erro observado.

### 16.2 Erro do AM

A relação entre `confidence_mse` e o MSE do Half* também foi fraca:

```text
0.158505
```

Isso indica que maior erro na estimativa do confidence score não necessariamente produz aumento proporcional no erro de reconstrução.

### 16.3 Correlação entre features ativas e passivas

A relação entre `max_active_corr` e `feature_mse` foi:

```text
0.028611
```

Não foi observada uma relação linear relevante entre essas duas métricas nos experimentos realizados.

### 16.4 Variabilidade da feature

A relação entre `std` e NRMSE foi consideravelmente mais forte:

```text
-0.675659
```

Isso sugere que features com menor variabilidade tendem a apresentar maior erro relativo de reconstrução.

---

## 17. Regressão exploratória

Foi ajustada uma regressão linear para explicar o NRMSE utilizando:

```text
std
abs_weight
logit_mse
```

As variáveis independentes foram padronizadas.

Resultado:

```text
R² = 0.4697171101
```

Portanto, o modelo explica aproximadamente **46,97% da variação observada no NRMSE**.

Coeficientes padronizados:

```text
std         = -1.253157
abs_weight  =  0.020524
logit_mse   =  0.204034
```

### Interpretação

O maior coeficiente em magnitude foi o de `std`:

```text
-1.253157
```

seguido por:

```text
logit_mse = 0.204034
abs_weight = 0.020524
```

Isso reforça a hipótese de que a variabilidade da feature possui uma associação mais forte com o erro relativo de reconstrução do que o módulo do peso ou o erro do logit, dentro dos experimentos atuais.

Essa regressão deve ser considerada **exploratória**, e não uma análise causal ou inferencial definitiva.

---

## 18. Hipótese atual

A hipótese de trabalho neste ponto é:

> **A variabilidade da feature passiva parece ser um dos fatores mais importantes associados à qualidade relativa da reconstrução realizada pelo Half*.**

Features com baixa variabilidade podem apresentar NRMSE elevado mesmo quando o MSE absoluto não é o maior.

---

## 19. Limitações

- Os 19 cenários possuem features passivas sobrepostas.
- As 95 observações (`19 × 5`) não são completamente independentes.
- A regressão realizada é exploratória.
- `std` e `variance` são diretamente relacionados e não devem ser tratados como variáveis independentes na mesma regressão.
- `abs_weight`, `1/abs_weight` e `1/weight²` são transformações relacionadas.
- Correlação linear não captura necessariamente relações não lineares.
- O problema de reconstrução é restrito: existem múltiplas features passivas, enquanto o logit fornece uma única equação.
- As correlações calculadas até agora não demonstraram relação forte entre correlação ativa-passiva e erro de reconstrução.

---

## 20. Próximos passos possíveis

A análise foi **pausada neste ponto**.

Quando for retomada, algumas possibilidades são:

1. Avaliar a correlação entre as próprias 5 features passivas em cada cenário.
2. Comparar correlação média/máxima entre features passivas com o `Half* MSE`.
3. Incorporar corretamente `max_active_corr` e `mean_active_corr` ao `feature_df`.
4. Fazer uma análise multivariada mais completa.
5. Analisar individualmente quais características das features tornam a reconstrução mais ou menos efetiva.

---

## 21. Estado atual

### Implementado

- [x] Carregamento do Bank Marketing
- [x] Remoção de `duration`
- [x] Target Mean Encoding
- [x] MinMax Scaling
- [x] Divisão entre features ativas e passivas
- [x] Modelo VFL
- [x] Adversary Model
- [x] Estimativa do confidence score
- [x] Conversão para logit
- [x] Half*
- [x] Reconstrução das features passivas
- [x] Reconstrução oracle utilizando o logit real
- [x] Execução de 19 cenários
- [x] Avaliação de MSE por feature
- [x] Avaliação de NRMSE
- [x] Análise de pesos
- [x] Análise de correlações
- [x] Regressão exploratória

### Temporariamente deixado de lado

- [ ] Implementação/avaliação do Half original
- [ ] Análise da correlação entre as próprias features passivas
- [ ] Regressão com `max_active_corr` e `mean_active_corr`

---

## 22. Resumo

Até o momento, os experimentos indicam que:

1. O AM consegue aproximar razoavelmente o confidence score do modelo VFL.
2. A utilização de `z_hat` em vez de `z_real` aumentou o MSE médio da reconstrução em aproximadamente **3,94%**.
3. O erro absoluto de reconstrução varia significativamente entre as features.
4. O MSE absoluto não é suficiente para comparar a qualidade da reconstrução entre features.
5. O NRMSE revelou uma relação mais forte com a variabilidade da feature.
6. A correlação `std × NRMSE` foi de aproximadamente **-0,676**.
7. Uma regressão utilizando `std`, `abs_weight` e `logit_mse` obteve **R² ≈ 0,470**.
8. `std` apresentou o maior coeficiente padronizado em magnitude.
9. O módulo do peso apresentou relação relativamente fraca com a qualidade da reconstrução.
10. A análise foi interrompida neste ponto para ser retomada posteriormente.
