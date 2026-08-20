# Previsão de Preços de Combustíveis — ANP

Pipeline completo de Data Science para previsão do preço semanal de venda de combustíveis (Gasolina, Etanol, Diesel e GLP) por estado brasileiro, a partir de dados públicos da ANP enriquecidos com indicadores macroeconômicos.

---

## Objetivo

Construir um pipeline end-to-end — da ingestão à predição — capaz de prever o preço médio semanal de combustíveis por estado, combinando o histórico de preços (lags, médias e desvios móveis) com variáveis macroeconômicas exógenas (dólar, Brent, IPCA, SELIC, PIB mensal).

O projeto foi conduzido com ênfase em rigor estatístico: cada decisão de modelagem — corte temporal, escolha de lags, transformação de target, escolha de teste estatístico — foi validada empiricamente antes de ser aplicada, e resultados negativos foram documentados com a mesma transparência que os positivos.

---

## Fonte de dados

- **ANP** — Levantamento de Preços de Combustíveis (~22,5 milhões de registros, granularidade diária, por posto).
- **Enriquecimento macroeconômico:**
  - Dólar comercial (compra e venda)
  - Petróleo Brent (Yahoo Finance, ticker `BZ=F`)
  - IPCA (inflação)
  - SELIC (taxa básica de juros)
  - PIB mensal

---

## Pipeline

```
Ingestão → EDA → Testes de Hipótese → Feature Engineering → Baseline → Modelagem → Diagnóstico Crítico
```

### 1. Ingestão e tratamento
- Agregação semanal (`W-SUN`) por Estado × Produto, usando **mediana** (não média) do `Valor de Venda`, dado o skewness positivo confirmado estatisticamente em todos os produtos.
- Reindex de todas as combinações Estado × Produto para garantir série temporal contígua (909 semanas), com `ffill` **segmentado por grupo** para tratar gaps pontuais.
- Feature de dispersão entre postos (`price_sale_std`) e cobertura amostral (`n_postos`), com flag `baixa_amostra` para semanas com menos de 2 postos reportando.

### 2. Análise Exploratória (EDA)
- **Distribuição de preços:** assimétrica à direita em todos os produtos (skewness entre 0.51 e 1.15), confirmada via teste de Kolmogorov-Smirnov (p ≈ 0.0000 em todos os produtos) — descartando o uso de testes paramétricos.
- **Diferença entre estados:** Kruskal-Wallis + post-hoc de Dunn confirmam diferenças estatisticamente significativas entre estados e regiões (ex.: SP como outlier de preço mais baixo; clusters regionais de preço identificados, como Norte/Centro-Oeste).
- **Correlação com exógenas:** Spearman (não-paramétrico) — dólar e PIB mensal com correlação forte (ρ ≈ 0.6–0.7) em todos os produtos.
- **Decomposição STL, ACF/PACF:** utilizados para definir a estrutura de lags e avaliar força de sazonalidade.

### 3. Quebras de regime identificadas

Duas mudanças estruturais de mercado foram identificadas e tratadas de forma diferente:

**a) Mudança da política de preços da Petrobras (2016)**

| Período | Gasolina | Etanol | Diesel | GLP |
|---|---|---|---|---|
| Antes de 2015 | -0.058 | +0.075 | -0.230 | -0.063 |
| 2016 em diante | +0.718 | +0.635 | +0.780 | +0.705 |

A correlação entre o Brent e o preço nacional muda de fraca/negativa para forte e positiva a partir de 2016 — coincidindo com a adoção da Política de Preços de Paridade de Importação (PPI) pela Petrobras. **Decisão:** treino iniciado em 2016-01, descartando o período de preços administrados.

**b) Salto de patamar entre 2024 e 2025-2026**

O preço médio de todos os produtos subiu de forma real e expressiva no período mais recente (ex.: GLP de ~R$84 para ~R$115). O Brent, em particular, saiu de uma faixa quase estática (~R$65, desvio padrão 0.70 nas últimas 52 semanas de treino) para uma faixa ampla e volátil (R$60–114, desvio padrão 17.3) no período de teste — **fora do range de valores visto durante o treino**. Diferente da quebra de 2016, essa não pôde ser corrigida por corte de dados, pois é justamente o período que se deseja prever.

### 4. Feature Engineering
- **Lags** definidos por PACF sobre a série diferenciada (estacionariedade confirmada via teste ADF): `lag_1, lag_2, lag_3, lag_4, lag_8, lag_27` — cobrindo ciclos semanal, mensal e semestral.
- **Rolling mean / std** em janelas de 4, 8 e 12 semanas, mantidas como features candidatas múltiplas (decisão final de relevância deixada para o `feature_importance` do modelo).
- **Target:** variação semanal (`price_sale_median − price_sale_median_lag_1`), não o nível absoluto — decisão tomada após diagnóstico de que o modelo, ao prever o nível absoluto, ignorava quase totalmente as variáveis exógenas em favor de features de curtíssimo prazo (lag_1 e rolling_mean concentrando >90% do `feature_importance`).

### 5. Divisão temporal

```
Treino:      2016-01 até 2025-04
Validação:   2025-05 até 2025-08
Teste final: 2025-09 até 2026-08
```

Validação cruzada temporal (`TimeSeriesSplit`, 8 folds) usada durante o tuning de hiperparâmetros, sempre respeitando a ordem cronológica.

### 6. Baseline

Baseline ingênuo: preço da próxima semana = preço da semana atual (`lag_1`). Referência mínima de qualidade que qualquer modelo precisa superar para agregar valor real.

| Produto | MAE baseline (teste) |
|---|---|
| Gasolina | 0.0597 |
| Etanol | 0.0552 |
| Diesel | 0.0815 |
| GLP | 1.0851 |

### 7. Modelagem

Modelos independentes por produto (XGBoost e LightGBM), com tuning bayesiano via **Optuna** (50 trials, otimizando o MAE reconstruído na escala real do preço).

**Resultado consolidado — melhora sobre o baseline no teste final:**

| Produto | XGBoost | LightGBM |
|---|---|---|
| GLP | -4.4% | -4.5% |
| Gasolina | -13.8% | -19.0% |
| Etanol | -12.3% | -9.2% |
| Diesel | -18.5% | -37.9% |

---

## Principal achado

**Nenhum dos dois algoritmos supera o baseline de persistência de forma consistente no período de teste, em nenhum dos quatro produtos.**

Investigação do resíduo previsto vs. real revelou viés sistemático direcional nas previsões, causado pela mudança de patamar do Brent entre treino e teste (seção 3b). Uma tentativa de mitigação — normalizar o Brent como variação percentual em vez de nível absoluto — foi testada e **piorou** o resultado, descartando essa hipótese de correção simples.

A confirmação cruzada com dois algoritmos independentes (XGBoost e LightGBM), que reproduziram o mesmo padrão de degradação com magnitude semelhante, indica que a limitação não é de implementação específica de um framework, mas sim uma característica estrutural de **modelos baseados em árvore de decisão**: dificuldade de extrapolar previsões além do range de valores observado durante o treino, quando o mercado real passa por uma mudança de regime.

Essa conclusão não invalida o pipeline construído — ao contrário, valida a importância de um processo de validação temporal rigoroso e de comparação sistemática contra um baseline bem definido, prática que evitou reportar um resultado enganosamente otimista.

---

## Estrutura do projeto

```
├── data/
│   └── dados_anp_modelado.parquet
├── src/
│   ├── tratamento_iniciais.py     # classe TratamentoIniciaisDF (split temporal, dummies)
│   ├── pipeline_modelagem.py      # orquestração: Optuna, treino, avaliação, produção
│   └── previsao.py                # previsão recursiva N semanas à frente
├── models/
│   ├── modelo_validacao_<produto>.json
│   ├── modelo_producao_<produto>.json
│   └── colunas_<produto>.json
├── resultados/
│   └── metricas_por_produto.csv
├── notebooks/
│   └── notebook_previsao_combustiveis.ipynb
└── README.md
```

---

## Stack técnica

`Python` · `pandas` · `scikit-learn` · `XGBoost` · `LightGBM` · `Optuna` · `scipy.stats` · `statsmodels` · `matplotlib` / `seaborn`

---

## Trabalhos futuros

- Avaliar modelos com componente de tendência explícito (regressão linear regularizada, SARIMA, Prophet) para lidar melhor com mudanças de patamar fora do range histórico.
- Explorar abordagens de ensemble entre o baseline de persistência e os modelos baseados em árvore.
- Plataforma de visualização (Streamlit) com integração a LLM para consulta em linguagem natural das previsões, exibindo lado a lado a previsão do modelo e do baseline para transparência sobre a incerteza envolvida.

---

## Autor

Projeto desenvolvido como parte de portfólio em Ciência de Dados, com foco em rigor metodológico: cada decisão — do corte temporal à escolha do teste estatístico — foi validada empiricamente antes de aplicada.