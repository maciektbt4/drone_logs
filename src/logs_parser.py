# src/parse_log.py
import re
import csv
import configparser
from pathlib import Path

# ─── 0) Ścieżki bazowe ───────────────────────────────────────────────
# Zakładamy, że uruchamiasz ten plik z katalogu "src/"
DATA_DIR   = Path("../data")    # surowe logi, z podfolderami np. data/run1/
OUTPUT_DIR = Path("../output")  # tutaj stworzymy output/run1/, run2/, …

OUTPUT_DIR.mkdir(exist_ok=True)

# ─── 1) Regex do parsowania linii logów ──────────────────────────────
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
    \s+Seen=\s*([01])                             # 10=Found (0 lub 1)
    \s+Reward:\s*([+-]?\d+(?:\.\d+)?)             # 11=Reward
""", re.VERBOSE)

HEADER = [
    "Step", "Episode", "Decision", "Eps", "lr",
    "Ret", "Last Crash", "t", "SF", "Found", "Reward"
]


def parse_one_run(run_name: str, run_input_dir: Path, run_output_dir: Path):
    """
    Parsuje wszystkie *.txt w katalogu run_input_dir → tworzy:
      - run_output_dir/trainlog.csv
      - run_output_dir/best_results.csv
      - run_output_dir/config.csv  (parametry z plików .cfg)
    """
    print(f">>> Rozpoczynam parsowanie runu '{run_name}' …")
    run_output_dir.mkdir(exist_ok=True, parents=True)

    # ─── 1a) Parsowanie logów do trainlog.csv i best_results.csv ─────────
    csv_all_path  = run_output_dir / "trainlog.csv"
    csv_best_path = run_output_dir / "best_results.csv"

    best_by_episode = {}
    total, parsed = 0, 0

    with csv_all_path.open("w", newline="", encoding="utf-8") as f_all:
        writer_all = csv.writer(f_all)
        writer_all.writerow(HEADER)

        # dla każdego pliku .txt w katalogu run_input_dir
        for txt_file in run_input_dir.glob("*.txt"):
            with txt_file.open(encoding="utf-8") as f_in:
                for line in f_in:
                    total += 1
                    m = pattern.match(line)
                    if not m:
                        continue
                    row = list(m.groups())
                    # konwersja pola "Found" z '0'/'1' do boolean
                    row[9] = bool(int(row[9]))
                    writer_all.writerow(row)
                    parsed += 1

                    ep = row[1]
                    ret_val = float(row[5])
                    # jeśli pierwszy raz lub Ret większy → aktualizujemy best_by_episode
                    if (ep not in best_by_episode) or (ret_val > best_by_episode[ep][0]):
                        best_by_episode[ep] = (ret_val, row)

    # Zapis najlepszych per Episode
    with csv_best_path.open("w", newline="", encoding="utf-8") as f_best:
        writer_best = csv.writer(f_best)
        writer_best.writerow(HEADER)
        # sortujemy po numerze ep (transformacja do int)
        for ep, (ret_val, row) in sorted(best_by_episode.items(), key=lambda kv: int(kv[0])):
            writer_best.writerow(row)

    print(f"    • Zapisano '{csv_all_path.name}' ({parsed}/{total} wierszy).")
    print(f"    • Zapisano '{csv_best_path.name}' (best per Episode).")

    # ─── 1b) Parsowanie wszystkich plików .cfg → config.csv ──────────────
    config_params = {}  # słownik param → wartość, np. "general_params.run_name"→"Tello_indoor"

    # Szukamy plików o rozszerzeniu .cfg w run_input_dir
    for cfg_file in run_input_dir.glob("*.cfg"):
        # używamy ConfigParser, by pominąć linie komentarzy # lub ;
        parser = configparser.ConfigParser()
        # wymuszamy ignorowanie wielkości liter w nazwach kluczy (jeśli ktoś czasem użyje innego stylu)
        parser.optionxform = str  

        parser.read(cfg_file, encoding="utf-8")
        for section in parser.sections():
            for key, val in parser.items(section):
                # wskutek mapowania na lower-case klucze są w jednakowym formacie
                param_name = f"{section}.{key}"
                config_params[param_name] = val

    # Jeżeli znaleziono jakiekolwiek parametry, zapiszemy je do CSV
    if config_params:
        csv_cfg_path = run_output_dir / "config.csv"
        with csv_cfg_path.open("w", newline="", encoding="utf-8") as f_cfg:
            writer_cfg = csv.writer(f_cfg)
            writer_cfg.writerow(["parameter", "value"])
            # sortujemy po nazwie parametru, dla czytelności
            for param, value in sorted(config_params.items()):
                writer_cfg.writerow([param, value])
        print(f"    • Zapisano '{csv_cfg_path.name}' ({len(config_params)} parametrów).")
    else:
        print("    • Nie znaleziono plików .cfg w tym runie.")

    print(f"<<< Zakończono parsowanie runu '{run_name}'.\n")


def main():
    """
    Przechodzimy rekurencyjnie po każdym podfolderze w 'data/'.
    Dla każdego tworzymy odpowiednik w 'output/' i wywołujemy parse_one_run.
    """
    if not DATA_DIR.exists() or not DATA_DIR.is_dir():
        print(f"Błąd: nie znaleziono katalogu {DATA_DIR}. Włóż tam swoje runy (podfoldery).")
        return

    for run_dir in sorted(DATA_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        run_name     = run_dir.name
        out_run_dir  = OUTPUT_DIR / run_name
        parse_one_run(run_name, run_dir, out_run_dir)


if __name__ == "__main__":
    main()
