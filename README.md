O pipeline que melhor representa o procedimento descrito no artigo fica:

Bank Marketing
      │
      ▼
Remove duration
      │
      ▼
19 features
      │
      ├─────────────────────┐
      │                     │
      ▼                     ▼
10 categóricas          9 numéricas
      │                     │
      ▼                     │
Target Mean Encoding        │
      │                     │
      └──────────┬──────────┘
                 ▼
          MinMaxScaler
                 │
                 ▼
       Todas as features ∈ [0,1]
                 │
                 ▼
          Divisão 14 / 5
          ┌──────┴──────┐
          ▼             ▼
       Active        Passive
        (14)            (5)


1. Remover duration
2. Separar target
3. Aplicar target mean encoding somente às 10 categóricas
4. Manter as 9 numéricas como numéricas
5. Normalizar todas as 19 features para [0,1]
6. Separar 14 features para Active e 5 para Passive
7. Dividir 80/20
8. Dividir novamente os 80% em 80/20
9. Treinar VFL
10. Treinar AM
11. Utilizar ĉ no Half*

Com isso, os resultados obtidos anteriormente:

VFL accuracy:                  0.9007587253
AM accuracy:                   0.8983308042
Confidence-score MSE:          0.0044701060
Half* reconstruction MSE:      0.1137887976
Half* MSE using REAL c:        0.1121233845

devem ser considerados a baseline correta da nossa implementação atual do artigo, com a ressalva de que estamos avaliando somente uma das possíveis alocações 14/5 das features.


Nível 1 — Baseline VFL
Y + X
 ↓
VFL
 ↓
accuracy
Nível 2 — Adversary Model
Y
 ↓
AM
 ↓
ĉ
 ↓
comparar com c

Métricas:

AM accuracy
Confidence-score MSE
Nível 3 — Feature Reconstruction Attack
Y
 ↓
AM
 ↓
ĉ
 ↓
ĉ'
 ↓
Half*
 ↓
X̂
 ↓
MSE(X, X̂)

E ainda teremos o oracle attack:

c real
 ↓
Half*
 ↓
X̂

para saber o limite superior do ataque.