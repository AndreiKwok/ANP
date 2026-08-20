"""
Pipeline de modelagem — treino, tuning (Optuna) e avaliação
segmentado por produto (GLP, GASOLINA, ETANOL, DIESEL).


"""

import os
import json
from functools import partial
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import optuna
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils.tools_optimize import *

# Configuração central — única fonte de verdade para as colunas do modelo


COLUNAS_REMOVER_BASE = [
    "price_sale_median",
    "price_sale_std",
    "n_postos",
    "baixa_amostra",
    "estado_AP",
    "estado_DF",
    "price_sale_median_rolling_mean_2",
    "rolling_std_2",
]

COL_TARGET = "price_sale_median"
COL_LAG1 = "price_sale_median_lag_1"


@dataclass
class ResultadoProduto:
    produto: str
    melhores_params: dict = field(default_factory=dict)

    mae_baseline_val: float = None
    mae_modelo_val: float = None

    mae_baseline_teste: float = None
    mae_modelo_teste: float = None
    rmse_modelo_teste: float = None

    colunas_modelo: list = field(default_factory=list)
    modelo_validacao: object = None  # treinado só no df_train (2016 até 2025-04)
    modelo_producao: object = None  # treinado com train+val+teste

    def melhora_pct(self, mae_modelo, mae_baseline):
        if mae_baseline is None or mae_baseline == 0:
            return None
        return (1 - mae_modelo / mae_baseline) * 100

    def resumo(self):
        return {
            "produto": self.produto,
            "mae_baseline_val": (
                round(self.mae_baseline_val, 4) if self.mae_baseline_val else None
            ),
            "mae_modelo_val": (
                round(self.mae_modelo_val, 4) if self.mae_modelo_val else None
            ),
            "melhora_val_%": (
                round(self.melhora_pct(self.mae_modelo_val, self.mae_baseline_val), 2)
                if self.mae_modelo_val
                else None
            ),
            "mae_baseline_teste": (
                round(self.mae_baseline_teste, 4) if self.mae_baseline_teste else None
            ),
            "mae_modelo_teste": (
                round(self.mae_modelo_teste, 4) if self.mae_modelo_teste else None
            ),
            "melhora_teste_%": (
                round(
                    self.melhora_pct(self.mae_modelo_teste, self.mae_baseline_teste), 2
                )
                if self.mae_modelo_teste
                else None
            ),
            "rmse_modelo_teste": (
                round(self.rmse_modelo_teste, 4) if self.rmse_modelo_teste else None
            ),
            "melhores_params": self.melhores_params,
        }


def alinhar_colunas(X: pd.DataFrame, colunas_referencia: list) -> pd.DataFrame:
    """
    Garante que X tenha exatamente as mesmas colunas, na mesma ordem,
    que o modelo foi treinado com. Evita o erro de 'feature_names mismatch'.
    """
    faltantes = [c for c in colunas_referencia if c not in X.columns]
    if faltantes:
        raise ValueError(f"Colunas ausentes em X: {faltantes}")
    extras = [c for c in X.columns if c not in colunas_referencia]
    if extras:
        X = X.drop(columns=extras)
    return X[colunas_referencia]


# Resultado estruturado de cada produto
class TratamentoIniciaisDF:
    def __init__(
        self, path_df: pd.DataFrame = rf"{os.getcwd()}\data\dados_anp_modelado.parquet"
    ):

        self.df = pd.read_parquet(path_df)
        self.df.index = self.df["dt_week"]
        self.df = self.df.drop(columns=["dt_week"])

    def apply_dummy_cat_var(self):
        col_cat = [
            coluna
            for coluna, tipo in self.df.dtypes.items()
            if tipo not in ["float64", "int64", "float32", "int32"]
        ]
        print(col_cat)
        self.df_dummy = pd.get_dummies(
            self.df, columns=col_cat, prefix=col_cat, prefix_sep="_"
        )
        # drop em columns exogenas que não servirao muito
        self.df_dummy = self.df_dummy.drop(
            columns=["High_BZ=F", "Low_BZ=F", "Open_BZ=F", "Volume_BZ=F"]
        )
        cols_bool = self.df_dummy.select_dtypes(include="bool").columns
        self.df_dummy[cols_bool] = self.df_dummy[cols_bool].astype(int)
        return self.df_dummy

    def apply_filters_date(self):
        """Aplicando o filtros de datas para definir os df`s train, val e test"""
        self.df_dummy = self.apply_dummy_cat_var()

        self.df_train = self.df_dummy[
            ((self.df_dummy.index.year >= 2016) & (self.df_dummy.index.year < 2025))
            | ((self.df_dummy.index.year == 2025) & (self.df_dummy.index.month <= 4))
        ]

        self.df_val = self.df_dummy[
            (self.df_dummy.index.year == 2025)
            & (self.df_dummy.index.month >= 5)
            & (self.df_dummy.index.month <= 8)
        ]

        self.df_test = self.df_dummy[
            ((self.df_dummy.index.year == 2025) & (self.df_dummy.index.month >= 9))
            | ((self.df_dummy.index.year == 2026) & (self.df_dummy.index.month <= 8))
        ]

        return self.df_train, self.df_val, self.df_test

    def prepara_df_produto(self, df: pd.DataFrame, produto: str):
        df_produto = df[df[f"Produto_{produto}"] == True].copy()
        cols_produto = [c for c in df_produto.columns if c.startswith("Produto_")]
        df_produto = df_produto.drop(columns=cols_produto)
        return df_produto


# Função objetivo do Optuna (genérica, reaproveitável entre produtos)
def objetivo_optuna(X_train, y_residuo_train, df_train, tscv, trial, modelo="XGB"):
    if modelo == "XGB":
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "n_estimators": trial.suggest_int(
                "n_estimators", 200, 800, step=100
            ),  # n_estimators
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
            "random_state": 42,
        }
    else:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": 42,
            "verbosity": -1,
        }
    maes_fold = []

    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_residuo_train.iloc[train_idx], y_residuo_train.iloc[val_idx]
        if modelo == "XGB":
            modelo = XGBRegressor(**params)
        else:
            modelo = lgb.LGBMRegressor(**params)
        modelo.fit(X_tr, y_tr)

        pred_residuo = modelo.predict(X_val)

        lag1_val = df_train.iloc[val_idx][COL_LAG1].values
        preco_real = df_train.iloc[val_idx][COL_TARGET].values
        preco_pred = lag1_val + pred_residuo

        mae = mean_absolute_error(preco_real, preco_pred)
        maes_fold.append(mae)

    return float(np.mean(maes_fold))


# Orquestrador principal — roda o pipeline completo para um produto
class PipelineProduto:
    """
    Executa, para um único produto:
      1. Otimização de hiperparâmetros via Optuna (TimeSeriesSplit no treino)
      2. Treino do modelo de validação (só no df_train)
      3. Avaliação no df_val (comparação com baseline)
      4. Avaliação única no df_test (resultado final)
      5. Treino do modelo de produção (train + val + teste combinados)
    """

    def __init__(
        self,
        produto: str,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
        modelo: str = "XGB",
        n_splits: int = 8,
        n_trials: int = 50,
        verbose: bool = True,
    ):
        self.produto = produto
        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.n_trials = n_trials
        self.verbose = verbose
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
        self.resultado = ResultadoProduto(produto=produto)
        self.modelo = modelo

    def log(self, msg):
        if self.verbose:
            print(f"[{self.produto}] {msg}")

    # ---- Etapa 1: Optuna -------------------------------------------------
    def _rodar_optuna(self):
        X_train, y_train_real, lag1_train = prepara_X_y(
            self.df_train, colunas_remover=COLUNAS_REMOVER_BASE, train_or_test="train"
        )
        y_residuo_train = y_train_real - lag1_train

        self.resultado.colunas_modelo = X_train.columns.tolist()

        objetivo_fn = partial(
            objetivo_optuna, X_train, y_residuo_train, self.df_train, self.tscv
        )

        study = optuna.create_study(direction="minimize")
        study.optimize(
            objetivo_fn, n_trials=self.n_trials, show_progress_bar=self.verbose
        )

        self.resultado.melhores_params = study.best_params
        self.log(f"Melhores hiperparâmetros: {study.best_params}")
        self.log(f"Melhor MAE (CV): {study.best_value:.4f}")

        return X_train, y_train_real, y_residuo_train, lag1_train

    # ---- Etapa 2: treino de validação -------------------------------------
    def _treinar_modelo_validacao(self, X_train, y_residuo_train, modelo="XGB"):
        params = dict(self.resultado.melhores_params)
        params["random_state"] = 42
        if modelo == "XGB":
            modelo = XGBRegressor(**params)  #
        else:
            modelo = lgb.LGBMRegressor(**params)
        modelo.fit(X_train, y_residuo_train)

        self.resultado.modelo_validacao = modelo
        return modelo

    # ---- Etapa 3: avaliação em df_val -------------------------------------
    def _avaliar_validacao(self, modelo):
        X_val, y_val_real, lag1_val = prepara_X_y(
            self.df_val, colunas_remover=COLUNAS_REMOVER_BASE
        )
        X_val = alinhar_colunas(X_val, self.resultado.colunas_modelo)

        pred_residuo = modelo.predict(X_val)
        pred_preco = lag1_val.values + pred_residuo

        mae_modelo = mean_absolute_error(y_val_real, pred_preco)
        mae_baseline = (lag1_val - y_val_real).abs().mean()

        self.resultado.mae_modelo_val = mae_modelo
        self.resultado.mae_baseline_val = mae_baseline

        self.log(
            f"Validação — MAE baseline: {mae_baseline:.4f} | MAE modelo: {mae_modelo:.4f} "
            f"| melhora: {self.resultado.melhora_pct(mae_modelo, mae_baseline):+.1f}%"
        )

    # ---- Etapa 4: avaliação final em df_test -------------------------------
    def _avaliar_teste(self, modelo):
        X_teste, y_teste_real, lag1_teste = prepara_X_y(
            self.df_test, colunas_remover=COLUNAS_REMOVER_BASE
        )
        X_teste = alinhar_colunas(X_teste, self.resultado.colunas_modelo)

        pred_residuo = modelo.predict(X_teste)
        pred_preco = lag1_teste.values + pred_residuo

        mae_modelo = mean_absolute_error(y_teste_real, pred_preco)
        rmse_modelo = mean_squared_error(y_teste_real, pred_preco) ** 0.5
        mae_baseline = (lag1_teste - y_teste_real).abs().mean()

        self.resultado.mae_modelo_teste = mae_modelo
        self.resultado.rmse_modelo_teste = rmse_modelo
        self.resultado.mae_baseline_teste = mae_baseline

        self.log(
            f"TESTE FINAL — MAE baseline: {mae_baseline:.4f} | MAE modelo: {mae_modelo:.4f} "
            f"| RMSE: {rmse_modelo:.4f} | melhora: "
            f"{self.resultado.melhora_pct(mae_modelo, mae_baseline):+.1f}%"
        )

    # ---- Etapa 5: modelo de produção (treina com tudo) ----------------------
    def _treinar_modelo_producao(self, modelo="XGB"):
        df_producao = pd.concat([self.df_train, self.df_val, self.df_test])
        X_prod, y_prod_real, lag1_prod = prepara_X_y(
            df_producao, colunas_remover=COLUNAS_REMOVER_BASE
        )
        y_prod_residuo = y_prod_real - lag1_prod

        params = dict(self.resultado.melhores_params)
        params["random_state"] = 42
        if modelo == "XGB":
            modelo_producao = XGBRegressor(**params)
        else:
            modelo_producao = lgb.LGBMRegressor(**params)
        modelo_producao.fit(X_prod, y_prod_residuo)

        self.resultado.modelo_producao = modelo_producao
        self.log("Modelo de produção treinado com train+val+teste combinados.")

    # ---- Execução completa ---------------------------------------------------
    def rodar(self) -> ResultadoProduto:
        self.log("Iniciando pipeline...")
        X_train, y_train_real, y_residuo_train, lag1_train = self._rodar_optuna()

        modelo_val = self._treinar_modelo_validacao(X_train, y_residuo_train)
        self._avaliar_validacao(modelo_val)
        self._avaliar_teste(modelo_val)

        self._treinar_modelo_producao()

        self.log("Pipeline concluído.\n")
        return self.resultado


# Orquestrador geral — roda os 4 produtos e consolida os resultados
def rodar_pipeline_completo(
    tratamento,
    produtos=("GLP", "GASOLINA", "ETANOL", "DIESEL"),
    n_splits=8,
    n_trials=50,
    salvar_modelos=True,
    pasta_modelos="models",
    pasta_resultados="resultados",
):
    """
    tratamento: instância já inicializada de TratamentoIniciaisDF,
                com apply_filters_date() já executado (ou será executado aqui).
    """
    df_train, df_val, df_test = tratamento.apply_filters_date()

    os.makedirs(pasta_modelos, exist_ok=True)
    os.makedirs(pasta_resultados, exist_ok=True)

    resultados = {}

    for produto in produtos:
        df_train_p = tratamento.prepara_df_produto(df_train, produto)
        df_val_p = tratamento.prepara_df_produto(df_val, produto)
        df_test_p = tratamento.prepara_df_produto(df_test, produto)

        pipeline = PipelineProduto(
            produto=produto,
            df_train=df_train_p,
            df_val=df_val_p,
            df_test=df_test_p,
            n_splits=n_splits,
            n_trials=n_trials,
        )
        resultado = pipeline.rodar()
        resultados[produto] = resultado

        if salvar_modelos:
            resultado.modelo_producao.save_model(
                os.path.join(pasta_modelos, f"modelo_producao_{produto.lower()}.json")
            )
            resultado.modelo_validacao.save_model(
                os.path.join(pasta_modelos, f"modelo_validacao_{produto.lower()}.json")
            )
            with open(
                os.path.join(pasta_modelos, f"colunas_{produto.lower()}.json"), "w"
            ) as f:
                json.dump(resultado.colunas_modelo, f, indent=2)

    # Consolida métricas de todos os produtos numa tabela só
    df_resumo = pd.DataFrame([r.resumo() for r in resultados.values()])
    df_resumo.to_csv(
        os.path.join(pasta_resultados, "metricas_por_produto.csv"), index=False
    )

    print("\n=== RESUMO FINAL ===")
    print(df_resumo.to_string(index=False))

    return resultados, df_resumo


if __name__ == "__main__":
    tratamento = TratamentoIniciaisDF()
    resultados, df_resumo = rodar_pipeline_completo(
        tratamento,
        produtos=("GLP", "ETANOL", "DIESEL", "GASOLINA"),
        n_splits=8,
        n_trials=50,
        salvar_modelos=False,
    )
