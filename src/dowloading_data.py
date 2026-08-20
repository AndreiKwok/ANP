import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import requests
import urllib3  # Para capturar erros de protocolo
from dateutil.relativedelta import relativedelta
from datetime import datetime
import pandas as pd
import numpy as np
import zipfile
from datetime import date
import matplotlib.pyplot as plt
from io import StringIO, BytesIO
from bs4 import BeautifulSoup
import shutil
import os
import re
import time
import easygui

# --- 1. Configurações Iniciais ---
plt.rcParams.update({"figure.figsize": (22, 8), "figure.dpi": 120})

# --- 2. Definição de Paths ---
base_path = rf"{os.getcwd()}\data\download_gov"
path_csv = os.path.join(base_path, "csv")
path_zip = os.path.join(base_path, "zip")
path_temp = os.path.join(base_path, "temp")  # Path definido para temp

# Criação de pastas
if not os.path.exists(path_csv):
    os.makedirs(path_csv)

if not os.path.exists(path_zip):
    os.makedirs(path_zip)

# Limpeza de pasta temp
if os.path.exists(path_temp):
    shutil.rmtree(path_temp)

# --- 3. Funções Auxiliares de Processamento ---
print(base_path)
print(path_csv)
print(path_zip)
print(path_temp)


def validate_and_convert_date_updated(date_str):
    """Valida e converte datas de 'dd/mm/aa' para 'dd/mm/yyyy'."""
    pattern_two_digit_year = r"^\d{2}/\d{2}/\d{2}$"
    pattern_four_digit_year = r"^\d{2}/\d{2}/\d{4}$"

    if re.match(pattern_two_digit_year, date_str):
        day, month, year = date_str.split("/")
        year = "20" + year
        return "/".join([day, month, year])

    elif re.match(pattern_four_digit_year, date_str):
        return date_str

    else:
        return float("nan")


def processar_dataframe(df):
    """
    Limpa, formata e processa o DataFrame de entrada.
    Esta é a etapa mais importante para garantir a concatenação correta.
    """

    # Limpa nomes das colunas ANTES de qualquer outra coisa
    df.columns = df.columns.str.replace("ï»¿", "")  # Remove BOM
    df.columns = (
        df.columns.str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("ascii")
    )  # Remove acentos
    df.columns = df.columns.str.strip()  # Remove espaços extras

    # Processamento de Datas
    if "Data da Coleta" in df.columns:
        df["Data da Coleta"] = df["Data da Coleta"].astype(str)
        df["Data da Coleta"] = df["Data da Coleta"].apply(
            validate_and_convert_date_updated
        )
        df = df.dropna(subset=["Data da Coleta"])
        df["Data da Coleta"] = pd.to_datetime(df["Data da Coleta"], format="%d/%m/%Y")
        df = df.sort_values(by=["CNPJ da Revenda", "Data da Coleta"])

    # Processamento de Valores (Venda)
    if "Valor de Venda" in df.columns:
        df["Valor de Venda"] = df["Valor de Venda"].astype(str)
        df["Valor de Venda"] = (
            df["Valor de Venda"]
            .str.replace(".", "", regex=False)
            .str.replace(",", ".")
            .astype(float)
        )
        df["Valor de Venda"].interpolate(method="linear", inplace=True)

    # Processamento de Valores (Compra)
    if "Valor de Compra" in df.columns:
        df["Valor de Compra"] = df["Valor de Compra"].astype(str)
        df["Valor de Compra"] = (
            df["Valor de Compra"]
            .str.replace(".", "", regex=False)
            .str.replace(",", ".")
            .astype(float)
        )
        df["Valor de Compra"].interpolate(method="linear", inplace=True)

    return df


# region get_last_date
# Arquivo bruto, sem variaveis exogenas
df_og = pd.read_parquet(f"{base_path}\\anp.parquet")
print("leu o arquivo original")


class ExtractLastDate:
    def __init__(self, df: pd.DataFrame, column_name: str = "Data da Coleta"):
        self.df = df
        self.column_name = column_name
        self._last_date_cache = None

    def get_last_date(self) -> pd.Timestamp:
        if self._last_date_cache is not None:
            return self._last_date_cache

        # Extrai a coluna (Sem sobrescrever a variável final)
        col_data = self.df[self.column_name]

        # Verifica se já é datetime de forma segura
        if not pd.api.types.is_datetime64_any_dtype(col_data):
            # 3. Converte corretamente (Passando a Series, e não self.df[Series])
            # errors='coerce' é útil para transformar erros de conversão em NaT (Not a Time)
            col_data = pd.to_datetime(col_data, format="%Y-%m-%d", errors="coerce")

        # Calcula e guarda o valor máximo real
        self._last_date_cache = col_data.max()

        print(self._last_date_cache)
        return self._last_date_cache

    def get_last_year(self) -> int:
        return self.get_last_date().year

    def get_last_month(self) -> int:
        return self.get_last_date().month


# endregion


# region extract year and month from href
def extract_year_month(href: str) -> tuple[int, int]:
    if href.endswith(".csv"):
        pattern = r"/(\d{4})/[^/]+-(\d+)\.csv"
        if str(datetime.today().year) in href:
            pattern = r"(\d{4})/(\d{2})-"

        elif "ultimas" in href or "semanas" in href:
            return ("ultimas", "semanas")

    else:
        pattern = r"/(\d{4})/[^/]+-(\d+)\.zip"

    match = re.search(pattern, href)
    if match:
        return (int(match.group(1)), int(match.group(2)))  # return: year, month


# endregion

# --- 4. Loop de Download (Scraping) ---
# region Scraping data
url = "https://www.gov.br/anp/pt-br/centrais-de-conteudo/dados-abertos/serie-historica-de-precos-de-combustiveis"
MAX_TENTATIVAS = 3

print("Iniciando verificação e download de arquivos...")
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")

ELD = ExtractLastDate(df_og)
last_year_from_df = ELD.get_last_year()
last_month_from_df = ELD.get_last_month()
all_years_needed = [year for year in range(last_year_from_df, date.today().year + 1)]

# duas palavras adicionadas em uma tupla, pois em 2026 a url possui ultimas e semanas nela
dates = [("ultimas", "semanas")]
current = ELD.get_last_date().replace(day=1)

while current <= datetime.today():
    dates.append((current.year, current.month))
    current += relativedelta(months=1)


# for link in soup.find_all("a", href=True):
#     href = link["href"]
#     print("HREF:", href)

#     # Unificar lógica de download para CSV e ZIP
#     if href.endswith(".csv") or href.endswith(".zip"):
#         year_month_from_href = extract_year_month(href)
#         # ((str(last_year_from_df) in str(href)) or str(date.today().year) in str(href)):
#         # CORRIGIR LOGICA PARA EXTRAIR DADOS DE ACORDO COM O ULTIMO REGISTRO NO ARQUIVO .CSV OU PARQUET - INFO 20/03
#         print(f"{year_month_from_href} in {dates}")
#         if year_month_from_href in dates:
#             # Determina o path de destino
#             if href.endswith(".csv"):
#                 file_path = os.path.join(path_csv, os.path.basename(href))
#                 tipo_arquivo = "CSV"

#                 print("href:", href)

#             else:  # .zip
#                 file_path = os.path.join(path_zip, os.path.basename(href))
#                 tipo_arquivo = "ZIP"

#                 print("href:", href)

#             # Lógica para pular download
#             if os.path.exists(file_path) and ("precos" not in href):
#                 print(
#                     f"Arquivo {tipo_arquivo} {os.path.basename(href)} já existe. Pulando."
#                 )

#                 print("href:", href)

#                 continue

#             print(f"Baixando {tipo_arquivo}: {href}")

#             # --- LÓGICA DE STREAMING E RETRY (Correção do IncompleteRead) ---
#             sucesso = False
#             for tentativa in range(MAX_TENTATIVAS):
#                 try:
#                     # Usamos stream=True e um timeout
#                     with requests.get(href, stream=True, timeout=30) as r:
#                         r.raise_for_status()  # Verifica se a requisição foi OK (erro 200)

#                         with open(file_path, "wb") as f:
#                             # Baixa em pedaços (chunks)
#                             for chunk in r.iter_content(
#                                 chunk_size=8192 * 1024
#                             ):  # chunks de 8MB
#                                 if chunk:
#                                     f.write(chunk)

#                     print(f"Download de {os.path.basename(href)} concluído.")
#                     sucesso = True
#                     break  # Sai do loop de tentativas

#                 # Captura os erros de conexão
#                 except (
#                     requests.exceptions.RequestException,
#                     urllib3.exceptions.ProtocolError,
#                 ) as e:
#                     print(
#                         f"  [Tentativa {tentativa + 1}/{MAX_TENTATIVAS}] Falha no download: {e}"
#                     )
#                     if tentativa < MAX_TENTATIVAS - 1:
#                         time.sleep(5)  # Espera 5s antes de tentar de novo
#                     else:
#                         print(
#                             f"ERRO: Falha ao baixar {href} após {MAX_TENTATIVAS} tentativas."
#                         )

#             if sucesso:
#                 # Pausa educada entre downloads
#                 time.sleep(5.0)

print("Downloads concluídos! Iniciando processamento...")
# endregion
# --- 5. Loop de Processamento e Concatenação ---

dfs = []

# Loop através de todos os arquivos na pasta CSV
for filename in os.listdir(path_csv):
    if filename.endswith(".csv"):
        print(f"Processando CSV: {filename}")
        file_path = os.path.join(path_csv, filename)
        try:
            df = pd.read_csv(file_path, sep=";", encoding="latin1", low_memory=False)
            df_processado = processar_dataframe(df)  # Usa a função refatorada
            dfs.append(df_processado)
        except Exception as e:
            print(f"Erro ao processar {filename}: {e}")

# Loop através de todos os arquivos na pasta ZIP
for filename in os.listdir(path_zip):
    if filename.endswith(".zip"):
        print(f"Processando ZIP: {filename}")
        file_path = os.path.join(path_zip, filename)

        try:
            with zipfile.ZipFile(file_path, "r") as z:
                # Lógica robusta para achar o CSV dentro do ZIP
                csv_filename = None
                for name in z.namelist():
                    if name.endswith(".csv"):
                        csv_filename = name
                        break  # Pega o primeiro CSV que encontrar

                if csv_filename:
                    with z.open(csv_filename) as f:
                        df = pd.read_csv(
                            f, sep=";", encoding="latin1", low_memory=False
                        )
                        df_processado = processar_dataframe(
                            df
                        )  # Usa a função refatorada
                        dfs.append(df_processado)
                else:
                    print(f"Nenhum arquivo .csv encontrado em {filename}")
        except Exception as e:
            print(f"Erro ao processar {filename}: {e}")

print("Processamento concluído. Concatenando DataFrames...")

# --- 6. Finalização ---
if dfs:  # Garante que a lista não está vazia
    df_extract = pd.concat(dfs, ignore_index=True)

    # Passos finais de limpeza
    df_extract = df_extract.drop_duplicates()
    df_extract = df_extract.sort_values(
        by=["Data da Coleta", "Produto", "Cep", "CNPJ da Revenda"]
    )

    # concatenando df extraido com o ultimo arquivo .parquet(df_og)
    final_df = pd.concat([df_og, df_extract], ignore_index=True)
    print("Salvando arquivo final anp.parquet...")
    try:
        final_df.to_parquet(f"{base_path}\\anp.parquet", index=False)
    except Exception as e:
        print("ERRO AO TENTAR SALVAR ARQUIVO: anp.parquet:", e)
    print("Processo finalizado!")
    final_df.info()
else:
    print("Nenhum dado foi processado. Verifique os downloads e os caminhos.")
