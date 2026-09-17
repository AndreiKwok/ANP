"""
App Streamlit — Previsão de Preços de Combustíveis (ANP)

Consome a tabela pré-computada em resultados/previsoes_horizonte.csv
(gerada por src/inferencia_previsao.ipynb) e o histórico tratado em
data/dados_anp_modelado.parquet, para exibir por produto e estado:
- o preço mais recente e o MAE histórico do modelo e do baseline;
- o histórico recente + a previsão de 4 semanas (modelo vs. baseline);
- a comparação de erro (MAE) entre estados, para o produto selecionado.

Roda com: streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).parent
PATH_PREVISOES = BASE_DIR / "resultados" / "previsoes_horizonte.csv"
PATH_HISTORICO = BASE_DIR / "data" / "dados_anp_modelado.parquet"

# paleta — ver skill de dataviz: séries categóricas em ordem fixa,
# histórico real em tinta neutra (não é uma "série" de previsão)
COR_HISTORICO = "#0b0b0b"
COR_MODELO = "#2a78d6"
COR_BASELINE = "#eb6834"
COR_GRADE = "#e1e0d9"
COR_EIXO_MUTED = "#898781"
COR_SEQUENCIAL = "#2a78d6"

PRODUTOS = ["GLP", "GASOLINA", "ETANOL", "DIESEL"]

st.set_page_config(
    page_title="Previsão de Preços de Combustíveis — ANP",
    page_icon="⛽",
    layout="wide",
)


@st.cache_data
def carregar_previsoes() -> pd.DataFrame:
    return pd.read_csv(PATH_PREVISOES)


@st.cache_data
def carregar_historico() -> pd.DataFrame:
    df = pd.read_parquet(PATH_HISTORICO)
    df.index = df["dt_week"]
    df = df.drop(columns=["dt_week"]).sort_index()
    return df


def semana_para_timestamp(col_semana: pd.Series) -> pd.Series:
    return pd.PeriodIndex(col_semana.str.split("/").str[0], freq="W-SUN").to_timestamp()


if not PATH_PREVISOES.exists():
    st.error(
        f"Não encontrei {PATH_PREVISOES.relative_to(BASE_DIR)}. "
        "Rode src/modelagem_06_final_treino.ipynb e depois "
        "src/inferencia_previsao.ipynb antes de abrir o app."
    )
    st.stop()

df_previsoes = carregar_previsoes()
df_historico = carregar_historico()

st.title("⛽ Previsão de Preços de Combustíveis — ANP")
st.caption(
    "Preço mediano semanal por estado, com previsão de 4 semanas via LightGBM "
    "(segmentação por cluster) — sempre lado a lado com o baseline de persistência."
)

with st.sidebar:
    st.header("Filtros")
    produto = st.selectbox("Produto", PRODUTOS)
    estados_disponiveis = sorted(df_previsoes.loc[df_previsoes["produto"] == produto, "estado"].unique())
    estado = st.selectbox("Estado", estados_disponiveis)

    st.divider()
    st.caption(
        "**Nota metodológica:** nenhum modelo testado neste projeto superou o "
        "baseline de persistência (último preço conhecido) de forma consistente "
        "no período de teste. O modelo é exibido lado a lado com o baseline por "
        "transparência — não como substituto comprovadamente melhor."
    )

prev_sel = df_previsoes[
    (df_previsoes["produto"] == produto) & (df_previsoes["estado"] == estado)
].sort_values("semana")

hist_sel = df_historico[
    (df_historico["Produto"] == produto) & (df_historico["Estado - Sigla"] == estado)
].sort_index()

preco_atual = hist_sel["price_sale_median"].iloc[-1]
mae_modelo = prev_sel["mae_modelo_estado"].iloc[0]
mae_baseline = prev_sel["mae_baseline_estado"].iloc[0]
melhora_pct = (1 - mae_modelo / mae_baseline) * 100 if mae_baseline else None

col1, col2, col3, col4 = st.columns(4)
col1.metric("Preço mais recente", f"R$ {preco_atual:,.3f}")
col2.metric("MAE modelo (estado)", f"{mae_modelo:.4f}")
col3.metric("MAE baseline (estado)", f"{mae_baseline:.4f}")
col4.metric(
    "Modelo vs. baseline",
    f"{melhora_pct:+.1f}%",
    help="Negativo = baseline erra menos que o modelo neste estado, no período de teste.",
)

st.subheader(f"Histórico + previsão — {produto} / {estado}")

hist_plot = hist_sel.tail(26)
x_hist = hist_plot.index.to_timestamp()
x_prev = semana_para_timestamp(prev_sel["semana"])

# ponte entre o último ponto real e o primeiro previsto, para as linhas
# de previsão não começarem "soltas" no gráfico
x_modelo = pd.concat([pd.Series([x_hist[-1]]), pd.Series(x_prev)], ignore_index=True)
y_modelo = pd.concat(
    [pd.Series([hist_plot["price_sale_median"].iloc[-1]]), prev_sel["preco_modelo"].reset_index(drop=True)],
    ignore_index=True,
)
y_baseline = pd.concat(
    [pd.Series([hist_plot["price_sale_median"].iloc[-1]]), prev_sel["preco_baseline"].reset_index(drop=True)],
    ignore_index=True,
)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=x_hist, y=hist_plot["price_sale_median"],
    mode="lines", name="Histórico real",
    line=dict(color=COR_HISTORICO, width=2),
))
fig.add_trace(go.Scatter(
    x=x_modelo, y=y_modelo,
    mode="lines+markers", name="Previsão — modelo (LGBM)",
    line=dict(color=COR_MODELO, width=2, dash="dash"),
    marker=dict(size=8),
))
fig.add_trace(go.Scatter(
    x=x_modelo, y=y_baseline,
    mode="lines+markers", name="Previsão — baseline (persistência)",
    line=dict(color=COR_BASELINE, width=2, dash="dash"),
    marker=dict(size=8),
))
fig.update_layout(
    height=440,
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(
        title=None, gridcolor=COR_GRADE, linecolor=COR_EIXO_MUTED,
        tickfont=dict(color=COR_HISTORICO),
    ),
    yaxis=dict(
        title=dict(text="Preço mediano (R$)", font=dict(color=COR_HISTORICO)),
        gridcolor=COR_GRADE, linecolor=COR_EIXO_MUTED,
        tickfont=dict(color=COR_HISTORICO),
    ),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
        font=dict(color=COR_HISTORICO),
    ),
    plot_bgcolor="#fcfcfb",
    paper_bgcolor="#fcfcfb",
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Exógenas (dólar, Brent, IPCA, SELIC, PIB) congeladas no último valor conhecido "
    "durante todo o horizonte de previsão — limitação documentada no README."
)

st.subheader("Previsão detalhada (4 semanas)")
tabela = prev_sel[["semana", "preco_modelo", "preco_baseline"]].rename(columns={
    "semana": "Semana",
    "preco_modelo": "Previsão — modelo",
    "preco_baseline": "Previsão — baseline",
})
st.dataframe(tabela, hide_index=True, use_container_width=True)

with st.expander(f"Comparar erro (MAE) entre estados — {produto}"):
    mae_produto = (
        df_previsoes[df_previsoes["produto"] == produto]
        .drop_duplicates(subset="estado")[["estado", "mae_modelo_estado"]]
        .sort_values("mae_modelo_estado")
    )
    cores_barras = [
        "#184f95" if e == estado else COR_SEQUENCIAL for e in mae_produto["estado"]
    ]
    fig_mae = go.Figure(go.Bar(
        x=mae_produto["mae_modelo_estado"], y=mae_produto["estado"],
        orientation="h", marker_color=cores_barras,
    ))
    fig_mae.update_layout(
        height=650,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(
            title=dict(text="MAE do modelo (menor é melhor)", font=dict(color=COR_HISTORICO)),
            gridcolor=COR_GRADE, tickfont=dict(color=COR_HISTORICO),
        ),
        yaxis=dict(
            title=None, gridcolor=COR_GRADE, autorange="reversed",
            tickfont=dict(color=COR_HISTORICO),
        ),
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
    )
    st.plotly_chart(fig_mae, use_container_width=True)
    st.caption(f"Estado selecionado ({estado}) destacado em azul escuro.")
