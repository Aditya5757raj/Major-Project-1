import os
import subprocess

LOG_FILE = "ingestion_log.txt"

def load_log():
    """Load already ingested file paths from log file."""
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f)
    return set()

def append_log(pdf_path):
    """Append a processed file path to the log file."""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(pdf_path + "\n")

def ingest_pdfs(root_folder):
    """
    Iterates through subfolders (year-wise) inside the root folder
    and ingests each PDF file with rag1.py using --append.
    Skips files already processed (from log).
    """
    print(f"Processing PDFs in: {root_folder}")

    processed_files = load_log()

    for year_folder in os.listdir(root_folder):
        year_path = os.path.join(root_folder, year_folder)

        # Only process valid year subfolders
        if os.path.isdir(year_path) and year_folder.isdigit() and 1950 <= int(year_folder) <= 2025:
            print(f"  Processing year folder: {year_folder}")

            for filename in os.listdir(year_path):
                if filename.lower().endswith(".pdf"):
                    pdf_path = os.path.join(year_path, filename)

                    if pdf_path in processed_files:
                        print(f"    Skipping (already ingested): {pdf_path}")
                        continue

                    cmd = ["python", "rag1.py", "--ingest", pdf_path, "--append"]
                    print("Running:", " ".join(cmd))

                    try:
                        subprocess.run(cmd, check=True)  # wait until finished
                        append_log(pdf_path)
                        print(f"    ✅ Ingested and logged: {pdf_path}")
                    except subprocess.CalledProcessError as e:
                        print(f"    ❌ Error ingesting {pdf_path}: {e}")
                        return  # stop if error occurs

    print("Finished ingestion of all PDFs.")

if __name__ == "__main__":
    root_directory = r"C:\Users\adity\Downloads\archive (8)\supreme_court_judgments"
    ingest_pdfs(root_directory)
