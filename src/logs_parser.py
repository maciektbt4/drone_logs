# src/parse_log.py
import re
import csv
from pathlib import Path

# ─── 0) Definicja ścieżek bazowych ───────────────────────────────
DATA_DIR   = Path("../data")    # zakładamy, że parse_log.py jest w folderze src/
OUTPUT_DIR = Path("../output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── 1) Przygotowanie regex-u (jak wcześniej) ──────────────────
pattern = re.compile(r"""
    ^\s*
    .*? -\s+Iter:\s+(\d+)\s*/\s*(\d+)\s+      # 1=Step  2=Episode
    ([A-Z][0-9]-[A-Z][0-9])\s+ -\s+           # 3=Decision
    (?:Rand|Pred)\s+Eps:\s+([+-]?\d+(?:\.\d+)?)   # 4=Eps
    \s+lr:\s+([+-]?\d+(?:\.\d+)?)                 # 5=lr
    \s+Ret\s*=\s*([+-]?\d+(?:\.\d+)?)             # 6=Ret
    \s+Last\s+Crash\s*=\s*(\d+)                   # 7=Last Crash
    \s+t=([+-]?\d+(?:\.\d+)?)                     # 8=t
    \s+SF\s*=\s*([+-]?\d+(?:\.\d+)?)              # 9=SF
    \s+Seen=\s*([01])                             # 10=Found
    \s+Reward:\s*([+-]?\d+(?:\.\d+)?)             # 11=Reward
""", re.VERBOSE)

# Nagłówki kolumn w CSV
HEADER = [
    "Step", "Episode", "Decision", "Eps", "lr",
    "Ret", "Last Crash", "t", "SF", "Found", "Reward"
]

def parse_one_run(run_name: str, run_input_dir: Path, run_output_dir: Path):
    """
    Parsuje wszystkie pliki .txt w katalogu run_input_dir (np. trainlog.txt, inne .txt).
    Generuje dwa pliki CSV w run_output_dir: trainlog.csv oraz best_results.csv.
    """
    # Upewnij się, że katalog docelowy istnieje
    run_output_dir.mkdir(exist_ok=True, parents=True)

    # Ścieżki plików wynikowych
    csv_all_path  = run_output_dir / "trainlog.csv"
    csv_best_path = run_output_dir / "best_results.csv"

    best_by_episode = {}  # słownik: ep -> (ret_value, row_tuple)

    total, parsed = 0, 0

    # Otwieramy plik CSV do zapisu (wszystkie rekordy)
    with csv_all_path.open("w", newline="", encoding="utf-8") as f_all:
        writer_all = csv.writer(f_all)
        writer_all.writerow(HEADER)

        # Dla każdego pliku *.txt w run_input_dir
        for txt_file in run_input_dir.glob("*.txt"):
            with txt_file.open(encoding="utf-8") as f_in:
                for line in f_in:
                    total += 1
                    m = pattern.match(line)
                    if not m:
                        continue
                    row = list(m.groups())       # 11 pól, wszystkie jako string
                    # zamiana pola "Found" (0/1) na bool, jeśli chcesz – albo trzymaj '0'/'1'
                    row[9] = bool(int(row[9]))   # Found = True/False

                    # Zapisz w CSV "all"
                    writer_all.writerow(row)
                    parsed += 1

                    episode = row[1]
                    ret_val = float(row[5])
                    # aktualizacja best_by_episode
                    if (episode not in best_by_episode) or (ret_val > best_by_episode[episode][0]):
                        best_by_episode[episode] = (ret_val, row)

    # Po przejściu wszystkich plików *.txt – generujemy drugi CSV (best_results.csv)
    with csv_best_path.open("w", newline="", encoding="utf-8") as f_best:
        writer_best = csv.writer(f_best)
        writer_best.writerow(HEADER)
        # zapisz w kolejności alfabetycznej runów (czyli sortujemy po kluczu epizodu)
        for ep, (ret_val, row) in sorted(best_by_episode.items(), key=lambda kv: int(kv[0])):
            writer_best.writerow(row)

    print(f"Run '{run_name}': przetworzono {parsed}/{total} wierszy -> "
          f"{csv_all_path.name}, {csv_best_path.name}")


def main():
    """
    Główna funkcja: przejdź po każdym podfolderze w data/, wywołaj parse_one_run.
    """
    if not DATA_DIR.exists() or not DATA_DIR.is_dir():
        print(f"Nie znaleziono katalogu {DATA_DIR}. Włóż tam podfoldery z logami.")
        return

    for run_dir in sorted(DATA_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        run_name = run_dir.name
        out_dir  = OUTPUT_DIR / run_name
        print(f">>> Parsuję run: '{run_name}' …")
        parse_one_run(run_name, run_dir, out_dir)

if __name__ == "__main__":
    main()
