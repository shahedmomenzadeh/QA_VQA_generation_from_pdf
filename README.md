# VQA from Book - Cataract Surgery Data Generation Pipeline

## Overview
This project provides a robust pipeline for generating Question-Answer (QA), Visual Question-Answer (VQA), and Table QA data from medical textbooks in PDF format, specifically focused on Cataract Surgery. The script processes PDF books, extracts text, figures, and tables, and uses large language models (LLMs) and vision-language models (VLMs) to generate high-quality Q&A pairs for research and training purposes.

## Features
- Extracts text, figures, and tables from PDF textbooks
- Generates general and visual Q&A pairs using LLMs and VLMs
- Performs OCR on tables and figures for QA generation
- Saves results as CSV and JSONL files for downstream use
- Modular and configurable workflow

## Requirements
- Python 3.8+
- The following Python packages (install with pip):

```
pip install -r requirements.txt
```

## Installation
1. Clone or download this repository.
2. Place your PDF books in the `Books/` directory (e.g., `Books/book1.pdf`).
3. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```
4. (Optional) For local OCR fallback, ensure Tesseract is installed and available in your system PATH.

## Usage
Run the main script from the project directory:

```
python VQA_from_book.py
```

### What the Script Does
- Scans the `Books/` directory for PDF files.
- For each book, extracts text, figures, and tables.
- Uses LLMs/VLMs to generate:
  - General QA pairs from text
  - Visual QA pairs from figures/images
  - Table QA pairs from tables (using OCR if needed)
- Saves generated QA and VQA pairs to `output/qa_pairs.csv`, `output/vqa_pairs.csv`, and their corresponding `.jsonl` files.
- Caches extracted images and tables in `output/images/` and `output/tables/`.

### Output
- `output/qa_pairs.csv` and `output/qa_pairs.jsonl`: General and table-based QA pairs
- `output/vqa_pairs.csv` and `output/vqa_pairs.jsonl`: Visual QA pairs (with image paths)
- `output/images/`: Extracted figure images
- `output/tables/`: Extracted table images

## Configuration
You can adjust pipeline options (e.g., enable/disable text QA, VQA, table QA, or change model names) by editing the `Config` class at the top of `VQA_from_book.py`.

## Notes
- The script is designed for research and data generation purposes, especially for medical AI and VQA tasks.
- For best results, ensure your PDFs are high quality and contain clear figures and tables.
- The script uses LangChain and LangGraph for workflow orchestration and model integration.

## Troubleshooting
- If you encounter errors related to missing models or packages, ensure all dependencies are installed and configured.
- For OCR issues, verify that Tesseract is installed and accessible.

## License
This project is for academic and research use. Please cite appropriately if used in publications.
