"""
Genie — agente de IA sobre a tabela de previsões (resultados/).

Usa tool-calling controlado: o LLM (Groq, modelos gratuitos com tool-use)
só pode chamar as funções Python definidas em `TOOLS`/`FERRAMENTAS` abaixo,
nunca executar código arbitrário. Cada ferramenta lê os DataFrames já
carregados pelo app.py e devolve um dict pequeno e serializável.

Chave: variável de ambiente GROQ_API_KEY (grátis em console.groq.com).
"""

import json
from typing import Any

from groq import Groq

MODELO_GROQ = "openai/gpt-oss-120b"

REGIOES = {
    "Norte": ["AC", "AP", "AM", "PA", "RO", "RR", "TO"],
    "Nordeste": ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
    "Centro-Oeste": ["DF", "GO", "MS", "MT"],
    "Sudeste": ["ES", "MG", "RJ", "SP"],
    "Sul": ["PR", "RS", "SC"],
}

SYSTEM_PROMPT = """Você é o Genie, um assistente que responde perguntas sobre as previsões \
de preços de combustíveis (GLP, GASOLINA, ETANOL, DIESEL) por estado brasileiro, geradas \
pelo pipeline de ML deste projeto.

Regras:
- Responda SEMPRE em português do Brasil, de forma direta e objetiva.
- Use as ferramentas disponíveis para consultar os dados — nunca invente números.
- Produtos válidos: GLP, GASOLINA, ETANOL, DIESEL. Estados são siglas de 2 letras (ex: SP, RJ).
- Para perguntas sobre uma região (Norte, Nordeste, Centro-Oeste, Sudeste, Sul) ou sobre vários \
estados de uma vez (comparações, rankings regionais), use a ferramenta comparar_estados — ela \
resolve a região para os estados e traz previsão + MAE de todos em UMA única chamada. Não chame \
obter_previsao/obter_mae_estado estado por estado para esse tipo de pergunta.
- Se o usuário citar um estado ou produto que não existir nos dados, avise e sugira as opções \
válidas (use a ferramenta listar_opcoes).
- Contexto importante do projeto: no período de teste, NENHUM modelo testado superou de forma \
consistente o baseline de persistência (repetir o último preço conhecido). Sempre que fizer \
sentido, mencione o baseline ao lado do modelo, não apenas o modelo isoladamente.
- As previsões usam variáveis exógenas (dólar, Brent, IPCA, SELIC, PIB) congeladas no último \
valor conhecido durante todo o horizonte — é uma limitação conhecida, cite-a se for relevante \
para a pergunta (ex: perguntas sobre confiabilidade de longo prazo).
- Se a pergunta não puder ser respondida com as ferramentas disponíveis, diga isso claramente \
em vez de especular.
- Seja eficiente com as ferramentas: use o mínimo necessário para responder, não repita a mesma \
chamada com os mesmos argumentos, e responda em texto assim que tiver informação suficiente.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "listar_opcoes",
            "description": "Lista os produtos e estados disponíveis nos dados.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "obter_previsao",
            "description": (
                "Retorna a previsão de 4 semanas (modelo e baseline) para um produto e estado."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "produto": {
                        "type": "string",
                        "description": "GLP, GASOLINA, ETANOL ou DIESEL",
                    },
                    "estado": {
                        "type": "string",
                        "description": "sigla do estado, ex: SP",
                    },
                },
                "required": ["produto", "estado"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "obter_historico_recente",
            "description": "Retorna o preço mediano real das últimas N semanas para um produto e estado.",
            "parameters": {
                "type": "object",
                "properties": {
                    "produto": {"type": "string"},
                    "estado": {"type": "string"},
                    "semanas": {
                        "type": "integer",
                        "description": "quantidade de semanas (padrão 8)",
                    },
                },
                "required": ["produto", "estado"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "obter_mae_estado",
            "description": (
                "Retorna o erro (MAE) do modelo e do baseline no período de teste, "
                "para um produto e estado específicos, com a melhora percentual."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "produto": {"type": "string"},
                    "estado": {"type": "string"},
                },
                "required": ["produto", "estado"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ranking_estados_por_mae",
            "description": (
                "Rankeia os estados por MAE do modelo para um produto, do melhor (menor erro) "
                "para o pior ou vice-versa."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "produto": {"type": "string"},
                    "ordem": {
                        "type": "string",
                        "enum": ["melhor", "pior"],
                        "description": "'melhor' = menor MAE primeiro, 'pior' = maior MAE primeiro",
                    },
                    "limite": {
                        "type": "integer",
                        "description": "quantos estados retornar (padrão 5)",
                    },
                },
                "required": ["produto"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "obter_metricas_modelo",
            "description": (
                "Retorna as métricas gerais (não por estado) do modelo de produção para um "
                "produto: MAE de validação/teste do modelo e do baseline, e melhora percentual."
            ),
            "parameters": {
                "type": "object",
                "properties": {"produto": {"type": "string"}},
                "required": ["produto"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "obter_cluster",
            "description": (
                "Retorna o cluster geoeconômico ao qual um estado pertence, para um produto, "
                "e quais outros estados estão no mesmo cluster."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "produto": {"type": "string"},
                    "estado": {"type": "string"},
                },
                "required": ["produto", "estado"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "comparar_estados",
            "description": (
                "Compara previsão (modelo e baseline) e MAE de vários estados de uma vez, para "
                "um produto — use para perguntas sobre uma região ou sobre múltiplos estados. "
                "Informe 'regiao' OU 'estados' (lista de siglas)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "produto": {"type": "string"},
                    "regiao": {
                        "type": "string",
                        "enum": ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"],
                    },
                    "estados": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "lista de siglas de estado, ex: ['SP', 'RJ']",
                    },
                },
                "required": ["produto"],
            },
        },
    },
]


def _normalizar_produto(contexto: dict, produto: str) -> str | None:
    produto = (produto or "").strip().upper()
    produtos = contexto["df_previsoes"]["produto"].unique()
    return produto if produto in produtos else None


def _normalizar_estado(estado: str) -> str:
    return (estado or "").strip().upper()


def _listar_opcoes(contexto: dict, **_) -> dict:
    df = contexto["df_previsoes"]
    return {
        "produtos": sorted(df["produto"].unique().tolist()),
        "estados": sorted(df["estado"].unique().tolist()),
    }


def _obter_previsao(contexto: dict, produto: str, estado: str, **_) -> dict:
    produto_norm = _normalizar_produto(contexto, produto)
    estado_norm = _normalizar_estado(estado)
    if produto_norm is None:
        return {
            "erro": f"produto '{produto}' não encontrado",
            **_listar_opcoes(contexto),
        }

    df = contexto["df_previsoes"]
    sel = df[
        (df["produto"] == produto_norm) & (df["estado"] == estado_norm)
    ].sort_values("semana")
    if sel.empty:
        return {
            "erro": f"sem previsão para {produto_norm}/{estado_norm}",
            **_listar_opcoes(contexto),
        }

    return {
        "produto": produto_norm,
        "estado": estado_norm,
        "previsao_semanas": [
            {
                "semana": row["semana"],
                "preco_modelo": round(float(row["preco_modelo"]), 4),
                "preco_baseline": round(float(row["preco_baseline"]), 4),
            }
            for _, row in sel.iterrows()
        ],
    }


def _obter_historico_recente(
    contexto: dict, produto: str, estado: str, semanas: int = 8, **_
) -> dict:
    produto_norm = _normalizar_produto(contexto, produto)
    estado_norm = _normalizar_estado(estado)
    if produto_norm is None:
        return {
            "erro": f"produto '{produto}' não encontrado",
            **_listar_opcoes(contexto),
        }

    df = contexto["df_historico"]
    sel = df[
        (df["Produto"] == produto_norm) & (df["Estado - Sigla"] == estado_norm)
    ].sort_index()
    if sel.empty:
        return {"erro": f"sem histórico para {produto_norm}/{estado_norm}"}

    sel = sel.tail(max(1, min(int(semanas or 8), 52)))
    return {
        "produto": produto_norm,
        "estado": estado_norm,
        "historico": [
            {"semana": str(idx), "preco_mediano": round(float(preco), 4)}
            for idx, preco in sel["price_sale_median"].items()
        ],
    }


def _obter_mae_estado(contexto: dict, produto: str, estado: str, **_) -> dict:
    produto_norm = _normalizar_produto(contexto, produto)
    estado_norm = _normalizar_estado(estado)
    if produto_norm is None:
        return {
            "erro": f"produto '{produto}' não encontrado",
            **_listar_opcoes(contexto),
        }

    df = contexto["df_mae_estado"]
    sel = df[(df["produto"] == produto_norm) & (df["estado"] == estado_norm)]
    if sel.empty:
        return {"erro": f"sem dados de MAE para {produto_norm}/{estado_norm}"}

    row = sel.iloc[0]
    mae_modelo = float(row["mae_estado"])
    mae_baseline = float(row["mae_baseline_estado"])
    melhora_pct = (1 - mae_modelo / mae_baseline) * 100 if mae_baseline else None
    return {
        "produto": produto_norm,
        "estado": estado_norm,
        "mae_modelo": round(mae_modelo, 4),
        "mae_baseline": round(mae_baseline, 4),
        "melhora_pct": round(melhora_pct, 2) if melhora_pct is not None else None,
        "interpretacao": (
            "negativo = baseline erra menos que o modelo neste estado, no período de teste"
        ),
    }


def _ranking_estados_por_mae(
    contexto: dict, produto: str, ordem: str = "melhor", limite: int = 5, **_
) -> dict:
    produto_norm = _normalizar_produto(contexto, produto)
    if produto_norm is None:
        return {
            "erro": f"produto '{produto}' não encontrado",
            **_listar_opcoes(contexto),
        }

    df = contexto["df_mae_estado"]
    sel = df[df["produto"] == produto_norm].sort_values(
        "mae_estado", ascending=(ordem != "pior")
    )
    limite = max(1, min(int(limite or 5), len(sel)))
    sel = sel.head(limite)
    return {
        "produto": produto_norm,
        "ordem": ordem,
        "ranking": [
            {"estado": r["estado"], "mae_modelo": round(float(r["mae_estado"]), 4)}
            for _, r in sel.iterrows()
        ],
    }


def _obter_metricas_modelo(contexto: dict, produto: str, **_) -> dict:
    produto_norm = _normalizar_produto(contexto, produto)
    if produto_norm is None:
        return {
            "erro": f"produto '{produto}' não encontrado",
            **_listar_opcoes(contexto),
        }

    df = contexto["df_metricas"]
    sel = df[df["produto"] == produto_norm]
    if sel.empty:
        return {"erro": f"sem métricas gerais para {produto_norm}"}

    row = sel.iloc[0]
    return {
        "produto": produto_norm,
        "mae_baseline_validacao": round(float(row["mae_baseline_val"]), 4),
        "mae_modelo_validacao": round(float(row["mae_modelo_val"]), 4),
        "mae_baseline_teste": round(float(row["mae_baseline_teste"]), 4),
        "mae_modelo_teste": round(float(row["mae_modelo_teste"]), 4),
        "melhora_teste_pct": round(float(row["melhora_teste_%"]), 2),
        "interpretacao": (
            "melhora_teste_pct negativo = o modelo errou mais que o baseline de persistência "
            "no período de teste"
        ),
    }


def _obter_cluster(contexto: dict, produto: str, estado: str, **_) -> dict:
    produto_norm = _normalizar_produto(contexto, produto)
    estado_norm = _normalizar_estado(estado)
    if produto_norm is None:
        return {
            "erro": f"produto '{produto}' não encontrado",
            **_listar_opcoes(contexto),
        }

    clusters = contexto["clusters"].get(produto_norm, {})
    cluster_estado = clusters.get(estado_norm)
    if cluster_estado is None:
        return {
            "erro": f"estado '{estado_norm}' não encontrado nos clusters de {produto_norm}"
        }

    mesmo_cluster = sorted(
        e for e, c in clusters.items() if c == cluster_estado and e != estado_norm
    )
    return {
        "produto": produto_norm,
        "estado": estado_norm,
        "cluster": cluster_estado,
        "outros_estados_no_mesmo_cluster": mesmo_cluster,
    }


def _normalizar_regiao(regiao: str) -> str | None:
    regiao = (regiao or "").strip().lower()
    return next((r for r in REGIOES if r.lower() == regiao), None)


def _comparar_estados(
    contexto: dict,
    produto: str,
    regiao: str | None = None,
    estados: list[str] | None = None,
    **_,
) -> dict:
    produto_norm = _normalizar_produto(contexto, produto)
    if produto_norm is None:
        return {"erro": f"produto '{produto}' não encontrado", **_listar_opcoes(contexto)}

    lista_estados: set[str] = set()
    if regiao:
        regiao_norm = _normalizar_regiao(regiao)
        if regiao_norm is None:
            return {"erro": f"região '{regiao}' inválida", "regioes_validas": list(REGIOES)}
        lista_estados.update(REGIOES[regiao_norm])
    else:
        regiao_norm = None
    if estados:
        lista_estados.update(_normalizar_estado(e) for e in estados)

    if not lista_estados:
        return {"erro": "informe 'regiao' ou 'estados'"}

    df_prev = contexto["df_previsoes"]
    df_mae = contexto["df_mae_estado"]

    comparacao = []
    for estado in sorted(lista_estados):
        prev_sel = df_prev[(df_prev["produto"] == produto_norm) & (df_prev["estado"] == estado)]
        mae_sel = df_mae[(df_mae["produto"] == produto_norm) & (df_mae["estado"] == estado)]
        if prev_sel.empty or mae_sel.empty:
            comparacao.append({"estado": estado, "erro": "sem dados"})
            continue

        mae_row = mae_sel.iloc[0]
        mae_modelo = float(mae_row["mae_estado"])
        mae_baseline = float(mae_row["mae_baseline_estado"])
        comparacao.append(
            {
                "estado": estado,
                "preco_modelo_medio_previsto": round(float(prev_sel["preco_modelo"].mean()), 4),
                "preco_baseline_medio_previsto": round(float(prev_sel["preco_baseline"].mean()), 4),
                "mae_modelo": round(mae_modelo, 4),
                "mae_baseline": round(mae_baseline, 4),
                "melhora_pct": (
                    round((1 - mae_modelo / mae_baseline) * 100, 2) if mae_baseline else None
                ),
            }
        )

    return {"produto": produto_norm, "regiao": regiao_norm, "comparacao": comparacao}


_DISPATCH = {
    "listar_opcoes": _listar_opcoes,
    "obter_previsao": _obter_previsao,
    "obter_historico_recente": _obter_historico_recente,
    "obter_mae_estado": _obter_mae_estado,
    "ranking_estados_por_mae": _ranking_estados_por_mae,
    "obter_metricas_modelo": _obter_metricas_modelo,
    "obter_cluster": _obter_cluster,
    "comparar_estados": _comparar_estados,
}


def executar_tool(nome: str, argumentos: dict, contexto: dict) -> dict:
    funcao = _DISPATCH.get(nome)
    if funcao is None:
        return {"erro": f"ferramenta desconhecida: {nome}"}
    try:
        return funcao(contexto, **argumentos)
    except Exception as exc:  # dados inesperados não devem derrubar o chat
        return {"erro": f"falha ao executar {nome}: {exc}"}


def responder(
    pergunta: str, historico: list[dict], contexto: dict, api_key: str
) -> str:
    """Roda o loop de tool-calling no Groq e devolve a resposta final em texto."""
    client = Groq(api_key=api_key)

    mensagens: list[dict[str, Any]] = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + historico
        + [{"role": "user", "content": pergunta}]
    )

    max_rodadas_com_tools = 8
    for rodada in range(max_rodadas_com_tools + 1):
        # na última rodada, força uma resposta em texto com o que já foi
        # coletado, em vez de deixar o modelo pedir mais uma ferramenta
        forcar_final = rodada == max_rodadas_com_tools
        resposta = client.chat.completions.create(
            model=MODELO_GROQ,
            messages=mensagens,
            tools=TOOLS,
            tool_choice="none" if forcar_final else "auto",
            temperature=0.2,
        )
        msg = resposta.choices[0].message

        if not msg.tool_calls:
            return msg.content or "Não consegui gerar uma resposta."

        mensagens.append(
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
        )
        for tc in msg.tool_calls:
            argumentos = json.loads(tc.function.arguments or "{}")
            resultado = executar_tool(tc.function.name, argumentos, contexto)
            mensagens.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(resultado, ensure_ascii=False),
                }
            )

    return "Não consegui concluir a resposta a tempo — tente reformular a pergunta."
