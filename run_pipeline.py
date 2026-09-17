"""
Orquestrador do pipeline: dados brutos -> features -> treino -> inferência.

Ordem de execução:
    dowloading_data.py -> FE1.ipynb -> FE2.ipynb -> FE3.ipynb ->
    modelagem_06_final_treino.ipynb -> inferencia_previsao.ipynb

`src/extract_dolar.ipynb` é deliberadamente pulado: sua célula de merge lê
`data/anp_parquet.parquet`, um arquivo que não existe mais no repo, e nada
depois dele consome sua saída (`FE1.ipynb` refaz a mesma extração de
dólar/Brent/IPCA/SELIC/PIB de forma independente e corrigida). Ver
CLAUDE.md / .claude/PROGRESS.md para o histórico dessa decisão.

Cada etapa roda até o fim antes da próxima começar; qualquer falha
interrompe o pipeline imediatamente (fail-fast) para não gerar artefatos
com dados parciais/inconsistentes.

Rodar a partir da raiz do repo:
    python run_pipeline.py
"""

import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent
SRC_DIR = REPO_ROOT / "src"
KERNEL = "anp-anaconda"
PYTHON = sys.executable

NOTEBOOKS = [
    "FE1.ipynb",
    "FE2.ipynb",
    "FE3.ipynb",
    "modelagem_06_final_treino.ipynb",
    "inferencia_previsao.ipynb",
]

AVISOS = [
    "src/extract_dolar.ipynb foi deliberadamente pulado neste pipeline: lê "
    "um arquivo inexistente (data/anp_parquet.parquet) e sua saída não é "
    "consumida por nenhuma etapa seguinte. FE1.ipynb refaz a mesma "
    "extração de forma corrigida e independente.",

    "dowloading_data.py reconcatena os CSVs/ZIPs já presentes em "
    "data/download_gov/csv e /zip com o anp.parquet existente, sem "
    "deduplicar o resultado final — rodar mais de uma vez sem limpar essas "
    "pastas pode duplicar linhas. Note também que o laço de scraping do "
    "site da ANP está comentado no script: ele não baixa arquivos novos "
    "sozinho, só reprocessa o que já estiver nas pastas csv/zip.",

    "FE1.ipynb faz chamadas de rede ao vivo (API do Banco Central e "
    "yfinance para o Brent) — precisa de internet e pode falhar por "
    "instabilidade externa, sem relação com o código.",

    "Esta execução sobrescreve arquivos gitignorados em data/, models/ e "
    "resultados/ (dados/modelos/artefatos anteriores não ficam em "
    "histórico de git) — sem rede de segurança além de backup manual.",
]


def rodar_script(caminho: Path, cwd: Path):
    print(f"\n{'=' * 70}\n> {caminho.name} (script)\n{'=' * 70}")
    inicio = time.time()
    resultado = subprocess.run([PYTHON, str(caminho)], cwd=cwd)
    if resultado.returncode != 0:
        raise RuntimeError(f"{caminho.name} falhou (exit code {resultado.returncode})")
    print(f"OK — {caminho.name} em {time.time() - inicio:.1f}s")


def rodar_notebook(nome: str):
    print(f"\n{'=' * 70}\n> {nome} (notebook)\n{'=' * 70}")
    inicio = time.time()
    resultado = subprocess.run(
        [
            PYTHON, "-m", "jupyter", "nbconvert",
            "--to", "notebook", "--execute", "--inplace",
            "--ExecutePreprocessor.timeout=1800",
            f"--ExecutePreprocessor.kernel_name={KERNEL}",
            nome,
        ],
        cwd=SRC_DIR,
    )
    if resultado.returncode != 0:
        raise RuntimeError(f"{nome} falhou (exit code {resultado.returncode})")
    print(f"OK — {nome} em {time.time() - inicio:.1f}s")


def main():
    inicio_total = time.time()

    print("AVISOS antes de começar:")
    for aviso in AVISOS:
        print(f" - {aviso}")

    rodar_script(SRC_DIR / "dowloading_data.py", cwd=REPO_ROOT)

    for nb in NOTEBOOKS:
        rodar_notebook(nb)

    print(f"\nPipeline completo em {(time.time() - inicio_total) / 60:.1f} min.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(f"\nPIPELINE INTERROMPIDO: {e}")
        sys.exit(1)
