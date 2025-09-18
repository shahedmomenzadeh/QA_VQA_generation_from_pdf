"""
VLM Data Generation Pipeline for Cataract Surgery Textbooks

Rewritten full script with robust table and figure handling and fallbacks.
Key fixes included:
- Save table and figure images as ABSOLUTE paths and immediately verify existence.
- Do NOT rely on file:// URIs; prefer plain absolute path or base64 image payloads.
- VLM invocation: try path-mode, then base64-mode, then local pytesseract fallback.
- Added helpful logging so you can trace missing file issues.
- Kept original LangGraph synchronous workflow structure.
- Modified for table QA: added 3-page context window + table name, relevance check, no table name or path in QA, combined with text QA in one JSONL.
- Modified for VQA: no figure name in QA, general questions for VLM finetuning phrased with 'in this figure' or 'in the image', include image_path in separate JSONL.
- Added quality check nodes for QA and VQA to filter non-medical content (e.g., authors, non-medical topics).

NOTE: Adjust the VLM "HumanMessage" payload fields (image_url / image_base64) to match
your ChatOllama / VLM client's expected schema. Look for comments marked: "ADJUST HERE".
"""

import os
import re
import json
import base64
from pathlib import Path
from typing import TypedDict, List, Tuple

import fitz  # PyMuPDF
import pandas as pd

# Optional local OCR fallback
try:
    from PIL import Image
    import pytesseract
    HAS_PYTESSERACT = True
except Exception:
    HAS_PYTESSERACT = False

# LangGraph and LangChain components (keep as in your environment)
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage


# -----------------------------------------------------------------------------
# 1. Configuration and Setup
# -----------------------------------------------------------------------------

class Config:
    """Holds all configuration for the data generation pipeline."""
    # --- Module Controls ---
    RUN_TEXT_QA = True  # Enabled for text QA
    RUN_VQA = True
    RUN_TABLE_QA = True

    # --- Model Configuration ---
    LLM_MODEL = "qwen2.5vl:7b"
    VLM_MODEL = "qwen2.5vl:7b"

    # Instantiate models in a try/except so local runs don't fail silently
    try:
        LLM = ChatOllama(model=LLM_MODEL, temperature=0.0)
        VLM = ChatOllama(model=VLM_MODEL, temperature=0.0)
    except Exception as e:
        print("⚠️ Warning: could not instantiate ChatOllama models:", e)
        LLM = None
        VLM = None

    # --- Directory Configuration ---
    BASE_DIR = os.getcwd()
    BOOKS_DIR = os.path.join(BASE_DIR, "Books")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    IMAGE_CACHE_DIR = os.path.join(OUTPUT_DIR, "images")
    TABLE_CACHE_DIR = os.path.join(OUTPUT_DIR, "tables")


# -----------------------------------------------------------------------------
# 2. Utility helpers
# -----------------------------------------------------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# -----------------------------------------------------------------------------
# 3. Custom Parsing Function (unchanged semantics)
# -----------------------------------------------------------------------------

def parse_qa_from_string(text_response: str) -> List[dict]:
    """Parses a string with QUESTION: and ANSWER: format into a list of dicts."""
    pairs = []
    if not text_response:
        return pairs
    if "no content for" in text_response.lower() or not text_response.strip():
        return pairs

    # Split on QUESTION: tokens
    blocks = text_response.strip().split('QUESTION:')[1:]

    for block in blocks:
        if 'ANSWER:' in block:
            question_part, answer_part = block.split('ANSWER:', 1)
            question = question_part.strip()
            answer = answer_part.strip().split('QUESTION:')[0].strip()

            if question and answer:
                pairs.append({"question": question, "answer": answer})
    return pairs


# -----------------------------------------------------------------------------
# 4. Prompts (updated with quality check prompts)
# -----------------------------------------------------------------------------

class Prompts:
    TEXT_QA = PromptTemplate(
        template="""You are a meticulous medical data extractor specializing in Cataract surgery. Your task is to generate 2-4 general question-answer pairs from the following text excerpt.

Rules:
- Questions must be general and not refer to page numbers (e.g., about techniques, complications, or procedures).
- Answers must be complete and thorough, including all key details from the text related to the question. Do not summarize aggressively.
- All information in the answer must be derived exclusively from the provided text. Do not add external knowledge.
- If the text is irrelevant, respond with 'No content for Q&A.'
- Use the exact format below for each pair.

QUESTION: [your question here]
ANSWER: [your answer here]

Text excerpt:
---
{text_chunk}
---""",
        input_variables=["text_chunk"],
    )

    IMAGE_RELEVANCE = PromptTemplate(
        template="""You are a Cataract surgery expert evaluating an image's suitability for VQA. Based on the page text, decide if the image with index {img_index} is valuable for training.

Valuable images: Diagnostic visuals (slit-lamp photos), surgical illustrations, anatomical diagrams.
Not valuable: Generic figures, logos, non-clinical content.

Respond ONLY with 'true' if valuable, or 'false' if not.

Page text:
---
{full_page_text}
---""",
        input_variables=["full_page_text", "img_index"],
    )

    VQA_GENERATION = PromptTemplate(
        template="""You are a research assistant generating a Visual Question-Answering pair about a figure from a medical textbook on Cataract surgery.

Your task is to:
1. Carefully read the provided multi-page text context.
2. Locate all sentences that describe or refer to the figure.
3. Generate ONE high-quality, general question a student would ask about the medical content shown in the figure, phrased to refer to 'this figure', 'the image', or similar (e.g., 'What does this figure show about ...?', 'In the image, what is ...?').
4. Provide a complete and thorough answer based on all relevant information in the text. Do not summarize aggressively; include all key details mentioned in the description of the figure.
5. The answer must be derived strictly and exclusively from the provided text. Do not add external knowledge.
6. Do NOT mention the figure's specific name (e.g., "Figure 12-1") in the question or answer. Refer to the figure generally (e.g., "this figure" or "the image").
7. Make the answer descriptive and general. Focus on the medical condition, technique, or finding shown. Omit case-specific details like patient age or gender.
8. If the context does not contain a clear description of the figure, respond with 'No content for Q&A'.
9. Use the exact format below.

QUESTION: [your question here]
ANSWER: [your answer here]

--- TEXT CONTEXT (PREVIOUS, CURRENT, AND NEXT PAGE) ---
{context_text}
---""",
        input_variables=["context_text"],
    )

    TABLE_OCR = "You are an expert OCR engine. Extract all text and structure from the following table image and format it as a clean, complete Markdown table."

    TABLE_RELEVANCE = PromptTemplate(
        template="""You are a Cataract surgery expert evaluating a table's suitability for QA. Based on the page text, decide if the table named "{table_name}" is valuable for training.

Valuable tables: Data on techniques, complications, procedures, outcomes, classifications.
Not valuable: Table of contents, acknowledgments, generic lists, non-clinical content.

Respond ONLY with 'true' if valuable, or 'false' if not.

Page text:
---
{full_page_text}
---""",
        input_variables=["full_page_text", "table_name"],
    )

    TABLE_QA_GENERATION = PromptTemplate(
        template="""You are a research assistant generating Question-Answering pairs about a table from a medical textbook on Cataract surgery.

Your task is to:
1. Carefully read the provided multi-page text context and the table data.
2. Locate all sentences in the context that describe or refer to the table's content.
3. Generate 2-3 high-quality, general questions a student would ask about the medical content in the table (e.g., techniques, complications, or procedures).
4. For each question, provide a complete and thorough answer based on the table data and relevant information from the text context. Do not summarize aggressively; include all key details.
5. The answer must be derived strictly and exclusively from the provided text and table data. Do not add external knowledge.
6. Do NOT mention the table's specific name (e.g., "Table 12-1") in the questions or answers. Refer to the data generally (e.g., "settings for phacoemulsification").
7. If the context or table does not contain sufficient information, respond with 'No content for Q&A'.
8. Use the exact format below for each pair.

QUESTION: [your question here]
ANSWER: [your answer here]

--- TEXT CONTEXT (PREVIOUS, CURRENT, AND NEXT PAGE) ---
{context_text}

--- TABLE DATA (MARKDOWN) ---
{markdown_table}
---""",
        input_variables=["context_text", "markdown_table"],
    )

    QA_QUALITY_CHECK = PromptTemplate(
        template="""You are a Cataract surgery expert evaluating the quality of a question-answer pair for medical training data.

Given the question: "{question}"

Is this question specifically about medical content related to Cataract surgery (e.g., techniques, complications, procedures, anatomy, outcomes)? It should not be about non-medical topics like authors, acknowledgments, page numbers, or general book info.

Respond ONLY with 'true' if it is a high-quality medical QA pair, or 'false' if not.""",
        input_variables=["question"],
    )

    VQA_QUALITY_CHECK = PromptTemplate(
        template="""You are a Cataract surgery expert evaluating the quality of a Visual Question-Answering pair for medical training data.

Given the question: "{question}"

Is this question specifically about medical content related to Cataract surgery (e.g., techniques, complications, procedures, anatomy, outcomes shown in the figure/image)? It should be phrased referring to 'this figure', 'the image', etc., and not about non-medical topics like authors, acknowledgments, or general book info.

Respond ONLY with 'true' if it is a high-quality medical VQA pair, or 'false' if not.""",
        input_variables=["question"],
    )


# -----------------------------------------------------------------------------
# 5. Pipeline State type
# -----------------------------------------------------------------------------

class PipelineState(TypedDict):
    pdf_paths: List[str]
    current_book_index: int
    text_chunks: List[dict]
    images_with_page_text: List[Tuple[str, str, str, str, str, int, int]]
    table_data: List[Tuple[str, str, str, str, int]]
    qa_pairs: List[dict]
    vqa_pairs: List[dict]
    qa_count: int
    vqa_count: int
    qa_count_per_book: int
    vqa_count_per_book: int


# -----------------------------------------------------------------------------
# 6. PDF parsing (with ABSOLUTE table/figure image saving, verification, and context)
# -----------------------------------------------------------------------------

def parse_pdf(pdf_path: str) -> Tuple[List[dict], List[Tuple], List[Tuple]]:
    """Parses a PDF to extract text chunks, image-caption groups, and tables."""
    print(f"📄 Parsing PDF with enhanced figure and table detection: {pdf_path}")
    doc = fitz.open(pdf_path)

    all_page_texts = [page.get_text("text").strip() for page in doc]
    text_chunks = [{"page": i, "text": text} for i, text in enumerate(all_page_texts) if len(text) > 100]

    final_images_data = []
    final_tables_data = []

    figure_pattern = re.compile(
        r'^(Figure|Fig\.?|Image|Img\.?|Plate|Illustration|Diagram|Chart|Graph|Photo|Photograph|Scheme|Map)\s+(\d+(?:[.-]\d+)*[A-Z]?)',
        re.IGNORECASE,
    )
    table_pattern = re.compile(r'^(Table)\s+(\d+(?:[.-]\d+)*[A-Z]?)', re.IGNORECASE)

    for page_num, page in enumerate(doc):
        blocks = page.get_text("blocks")
        caption_anchors = []
        page_tables_info = []
        processed_block_indices = set()

        # First pass on the page to find all table and figure labels
        for i, block in enumerate(blocks):
            if i in processed_block_indices:
                continue
            text = block[4].strip()

            # Find and expand tables
            match_table = table_pattern.match(text)
            if match_table:
                table_name = f"{match_table.group(1)} {match_table.group(2)}"
                table_rect = fitz.Rect(block[:4])
                processed_block_indices.add(i)

                for j in range(i + 1, len(blocks)):
                    next_block = blocks[j]
                    next_block_text = next_block[4].strip()

                    if figure_pattern.match(next_block_text) or table_pattern.match(next_block_text):
                        break

                    # heuristic: stop if the next block is far below (new section)
                    if next_block[1] - table_rect.y1 > 40:
                        break

                    table_rect |= fitz.Rect(next_block[:4])
                    processed_block_indices.add(j)

                page_tables_info.append({"name": table_name, "rect": table_rect})

            # Find figure labels
            match_fig = figure_pattern.match(text)
            if match_fig:
                figure_name = f"{match_fig.group(1)} {match_fig.group(2)}"
                caption_anchors.append({"name": figure_name, "text": text, "y": block[1]})

        # Save table images found on this page (ensure absolute paths) and add context
        for table_index, table_info in enumerate(page_tables_info):
            pix = page.get_pixmap(clip=table_info["rect"])
            filename = f"book_{os.path.basename(pdf_path).replace('.pdf', '')}_p{page_num}_t{table_index}_{table_info['name'].replace(' ', '_')}.png"
            table_path = os.path.join(Config.TABLE_CACHE_DIR, filename)
            ensure_dir(os.path.dirname(table_path))
            pix.save(table_path)

            table_path = os.path.abspath(table_path)
            print(f"    → Saved table image: {table_path} (exists? {os.path.exists(table_path)})")

            if os.path.exists(table_path):
                start, end = max(0, page_num - 1), min(len(all_page_texts), page_num + 2)
                context_text = "\n\n---\n\n".join(all_page_texts[start:end])
                full_page_text = all_page_texts[page_num]
                final_tables_data.append((table_path, table_info["name"], context_text, full_page_text, page_num))
            else:
                print(f"    ⚠️ Warning: expected saved table image not found: {table_path}")

        # Process images and match them with figure captions
        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                img_bbox = page.get_image_bbox(img)
            except ValueError:
                continue

            best_caption = None
            min_dist = float('inf')
            for anchor in caption_anchors:
                if anchor["y"] > img_bbox.y1:
                    dist = anchor["y"] - img_bbox.y1
                    if dist < min_dist:
                        min_dist = dist
                        best_caption = anchor

            if best_caption and min_dist < 150:
                figure_name = best_caption["name"]
                caption_text = best_caption["text"].replace('\n', ' ')
                start, end = max(0, page_num - 1), min(len(all_page_texts), page_num + 2)
                context_text = "\n\n---\n\n".join(all_page_texts[start:end])
                xref = img[0]
                base_image = doc.extract_image(xref)
                img_path = os.path.join(Config.IMAGE_CACHE_DIR, f"book_{os.path.basename(pdf_path).replace('.pdf', '')}_p{page_num}_i{img_index}.png")
                ensure_dir(os.path.dirname(img_path))
                with open(img_path, "wb") as f:
                    f.write(base_image["image"])
                img_path = os.path.abspath(img_path)
                final_images_data.append((img_path, figure_name, caption_text, context_text, all_page_texts[page_num], page_num, img_index))

    doc.close()
    print(f"✅ Parsed {pdf_path}: {len(text_chunks)} text chunks, {len(final_images_data)} matched images, {len(final_tables_data)} tables.")
    return text_chunks, final_images_data, final_tables_data


# -----------------------------------------------------------------------------
# 7. CSV -> JSONL helper
# -----------------------------------------------------------------------------

def convert_csv_to_jsonl(csv_path: str, jsonl_path: str):
    if not os.path.exists(csv_path):
        return
    print(f"Converting {csv_path} to {jsonl_path}...")
    df = pd.read_csv(csv_path)
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for record in df.to_dict(orient='records'):
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print("✅ Conversion complete.")


# -----------------------------------------------------------------------------
# 8. LangGraph Nodes (Synchronous) - parse current book node
# -----------------------------------------------------------------------------

def parse_current_book_node(state: PipelineState) -> dict:
    pdf_path = state["pdf_paths"][state["current_book_index"]]
    print(f"\n🚀 Starting processing for book {state['current_book_index'] + 1}: {pdf_path}")
    chunks, images, tables = parse_pdf(pdf_path)
    return {
        "text_chunks": chunks,
        "images_with_page_text": images,
        "table_data": tables,
        "qa_count_per_book": 0,
        "vqa_count_per_book": 0,
    }


def table_relevance_check_node(state: PipelineState) -> dict:
    if not Config.RUN_TABLE_QA:
        print("⏭️ Skipping Table Relevance Check as per configuration.")
        return {"table_data": []}

    if Config.LLM is None:
        print("⚠️ LLM is not available; skipping table relevance check.")
        return {"table_data": []}

    print(f"🔍 Checking relevance for {len(state['table_data'])} tables...")
    chain = Prompts.TABLE_RELEVANCE | Config.LLM | StrOutputParser()
    worthy_tables = []

    for item in state['table_data']:
        table_path, table_name, context_text, full_page_text, page_num = item
        try:
            result = chain.invoke({"full_page_text": full_page_text, "table_name": table_name})
            if result.strip().lower() == 'true':
                worthy_tables.append(item)
        except Exception as e:
            print(f"⚠️ Error checking relevance for table {table_path}: {e}")

    print(f"✅ Found {len(worthy_tables)} relevant tables.")
    return {"table_data": worthy_tables}


def image_relevance_check_node(state: PipelineState) -> dict:
    if not Config.RUN_VQA:
        print("⏭️ Skipping Image Relevance Check as per configuration.")
        return {"images_with_page_text": []}

    if Config.LLM is None:
        print("⚠️ LLM is not available; skipping image relevance check.")
        return {"images_with_page_text": []}

    print(f"🔍 Checking relevance for {len(state['images_with_page_text'])} images...")
    chain = Prompts.IMAGE_RELEVANCE | Config.LLM | StrOutputParser()
    worthy_images = []

    for item in state['images_with_page_text']:
        img_path, _, _, _, full_page_text, _, img_index = item
        try:
            result = chain.invoke({"full_page_text": full_page_text, "img_index": img_index})
            if result.strip().lower() == 'true':
                worthy_images.append(item)
        except Exception as e:
            print(f"⚠️ Error checking relevance for image {img_path}: {e}")

    print(f"✅ Found {len(worthy_images)} relevant images.")
    return {"images_with_page_text": worthy_images}


def generate_qa_node(state: PipelineState) -> dict:
    if not Config.RUN_TEXT_QA:
        print("⏭️ Skipping Text QA generation as per configuration.")
        return state

    if Config.LLM is None:
        print("⚠️ LLM is not available; skipping text QA generation.")
        return state

    print(f"🧠 Generating QA pairs from {len(state['text_chunks'])} text chunks...")
    new_qa = []
    chain = Prompts.TEXT_QA | Config.LLM | StrOutputParser()

    for chunk in state['text_chunks']:
        try:
            response_str = chain.invoke({"text_chunk": chunk["text"]})
            parsed_pairs = parse_qa_from_string(response_str)
            if parsed_pairs:
                for pair in parsed_pairs:
                    print(f"  - QA Pair Found:\n    Q: {pair['question']}\n    A: {pair['answer']}")
                new_qa.extend(parsed_pairs)
        except Exception as e:
            print(f"⚠️ Error during QA generation for page {chunk['page']}: {e}")

    updated_qa_count = state["qa_count"] + len(new_qa)
    updated_qa_per_book = state["qa_count_per_book"] + len(new_qa)
    print(f"📈 Generated {len(new_qa)} new QA pairs from text. Book total: {updated_qa_per_book}")

    return {
        "qa_pairs": state["qa_pairs"] + new_qa,
        "qa_count": updated_qa_count,
        "qa_count_per_book": updated_qa_per_book,
    }


def generate_vqa_node(state: PipelineState) -> dict:
    if not Config.RUN_VQA:
        print("⏭️ Skipping VQA generation as per configuration.")
        return state

    if Config.LLM is None:
        print("⚠️ LLM is not available; skipping VQA generation.")
        return state

    print(f"👁️ Generating VQA pairs for {len(state['images_with_page_text'])} relevant images...")
    new_vqa = []
    chain = Prompts.VQA_GENERATION | Config.LLM | StrOutputParser()

    for item in state['images_with_page_text']:
        img_path, figure_name, _, context_text, _, _, _ = item
        try:
            response_str = chain.invoke({"context_text": context_text})
            parsed_pairs = parse_qa_from_string(response_str)
            if parsed_pairs:
                pair_dict = parsed_pairs[0]
                print(f"  - VQA Pair Found for '{figure_name}':\n    Q: {pair_dict['question']}\n    A: {pair_dict['answer']}")
                pair_dict["image_path"] = img_path
                new_vqa.append(pair_dict)
        except Exception as e:
            print(f"⚠️ Error during VQA generation for image {img_path}: {e}")

    updated_vqa_count = state["vqa_count"] + len(new_vqa)
    updated_vqa_per_book = state["vqa_count_per_book"] + len(new_vqa)
    print(f"📈 Generated {len(new_vqa)} new VQA pairs. Book total: {updated_vqa_per_book}")

    return {
        "vqa_pairs": state["vqa_pairs"] + new_vqa,
        "vqa_count": updated_vqa_count,
        "vqa_count_per_book": updated_vqa_per_book,
    }


def qa_quality_check_node(state: PipelineState) -> dict:
    """Quality check for QA pairs to ensure they are medical-related."""
    if Config.LLM is None:
        print("⚠️ LLM is not available; skipping QA quality check.")
        return state

    print(f"🔍 Quality checking {len(state['qa_pairs'])} QA pairs...")
    chain = Prompts.QA_QUALITY_CHECK | Config.LLM | StrOutputParser()
    filtered_qa = []

    for pair in state['qa_pairs']:
        try:
            result = chain.invoke({"question": pair["question"]})
            if result.strip().lower() == 'true':
                filtered_qa.append(pair)
            else:
                print(f"  - Filtered out non-medical QA: {pair['question'][:100]}...")
        except Exception as e:
            print(f"⚠️ Error during QA quality check: {e}")
            # Keep the pair if check fails
            filtered_qa.append(pair)

    updated_qa_count = state["qa_count"] - (len(state['qa_pairs']) - len(filtered_qa))
    updated_qa_per_book = state["qa_count_per_book"] - (len(state['qa_pairs']) - len(filtered_qa))
    print(f"✅ Kept {len(filtered_qa)} high-quality QA pairs after filtering.")

    return {
        "qa_pairs": filtered_qa,
        "qa_count": updated_qa_count,
        "qa_count_per_book": updated_qa_per_book,
    }


def vqa_quality_check_node(state: PipelineState) -> dict:
    """Quality check for VQA pairs to ensure they are medical-related and properly phrased."""
    if Config.LLM is None:
        print("⚠️ LLM is not available; skipping VQA quality check.")
        return state

    print(f"🔍 Quality checking {len(state['vqa_pairs'])} VQA pairs...")
    chain = Prompts.VQA_QUALITY_CHECK | Config.LLM | StrOutputParser()
    filtered_vqa = []

    for pair in state['vqa_pairs']:
        try:
            result = chain.invoke({"question": pair["question"]})
            if result.strip().lower() == 'true':
                filtered_vqa.append(pair)
            else:
                print(f"  - Filtered out non-medical VQA: {pair['question'][:100]}...")
        except Exception as e:
            print(f"⚠️ Error during VQA quality check: {e}")
            # Keep the pair if check fails
            filtered_vqa.append(pair)

    updated_vqa_count = state["vqa_count"] - (len(state['vqa_pairs']) - len(filtered_vqa))
    updated_vqa_per_book = state["vqa_count_per_book"] - (len(state['vqa_pairs']) - len(filtered_vqa))
    print(f"✅ Kept {len(filtered_vqa)} high-quality VQA pairs after filtering.")

    return {
        "vqa_pairs": filtered_vqa,
        "vqa_count": updated_vqa_count,
        "vqa_count_per_book": updated_vqa_per_book,
    }


# -----------------------------------------------------------------------------
# 9. Robust table QA node (no table_path in QA pairs)
# -----------------------------------------------------------------------------

def generate_table_qa_node(state: PipelineState) -> dict:
    """Performs OCR on tables and generates QA pairs from them (robust file handling, with context)."""
    if not Config.RUN_TABLE_QA:
        print("⏭️ Skipping Table QA generation as per configuration.")
        return state

    if Config.VLM is None:
        print("⚠️ VLM is not available; skipping table QA generation.")
        return state

    print(f"📊 Generating QA pairs for {len(state['table_data'])} relevant tables...")
    new_qa = []
    qa_chain = Prompts.TABLE_QA_GENERATION | Config.LLM | StrOutputParser() if Config.LLM else None

    for item in state['table_data']:
        table_path, table_name, context_text, full_page_text, page_num = item
        try:
            abs_path = os.path.abspath(table_path)
            print(f"  - Processing table {table_name} at {abs_path}")

            if not os.path.exists(abs_path):
                print(f"    ⚠️ ERROR: file does not exist on disk: {abs_path}")
                continue

            # First attempt: call VLM with a plain filesystem path (NOT a file:// URI).
            markdown_table = None
            try:
                ocr_message = HumanMessage(
                    content=[
                        {"type": "text", "text": Prompts.TABLE_OCR},
                        # ADJUST HERE: many VLM clients expect a particular key for file paths.
                        # Try sending the absolute path under an "image_url" or "image_path" wrapper.
                        {"type": "image_url", "image_url": {"url": abs_path}}
                    ]
                )
                ocr_result = Config.VLM.invoke([ocr_message])
                markdown_table = getattr(ocr_result, 'content', None)
                print(f"    - VLM OCR result (path-mode) length: {len(markdown_table or '')}")
            except Exception as e_path:
                print(f"    ⚠️ VLM path-mode failed for {abs_path}: {e_path}")

                # Fallback: send base64 image bytes
                try:
                    with open(abs_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")

                    ocr_message = HumanMessage(
                        content=[
                            {"type": "text", "text": Prompts.TABLE_OCR},
                            # ADJUST HERE: some VLM clients expect base64 under a different field name
                            {"type": "image_base64", "image_base64": {"b64": b64}}
                        ]
                    )
                    ocr_result = Config.VLM.invoke([ocr_message])
                    markdown_table = getattr(ocr_result, 'content', None)
                    print(f"    - VLM OCR (base64) result length: {len(markdown_table or '')}")
                except Exception as e_b64:
                    print(f"    ⚠️ VLM base64-mode also failed: {e_b64}")

                    # Final fallback: local OCR with pytesseract (best-effort; not table-aware)
                    if HAS_PYTESSERACT:
                        try:
                            img = Image.open(abs_path)
                            local_text = pytesseract.image_to_string(img)
                            markdown_table = local_text
                            print("    - Using local pytesseract OCR fallback (plain text).")
                        except Exception as e_local:
                            print(f"    ⚠️ Local OCR fallback failed: {e_local}")
                    else:
                        print("    ⚠️ pytesseract not available as a fallback.")

            if markdown_table and "|" in str(markdown_table):
                if qa_chain is None:
                    print("    ⚠️ LLM is not available; skipping TABLE_QA generation for this table.")
                else:
                    response_str = qa_chain.invoke({"context_text": context_text, "markdown_table": markdown_table})
                    parsed_pairs = parse_qa_from_string(response_str)
                    if parsed_pairs:
                        for pair in parsed_pairs:
                            print(f"  - Table QA Found:\n    Q: {pair['question']}\n    A: {pair['answer']}")
                        new_qa.extend(parsed_pairs)  # No table_path added
            else:
                print(f"    ℹ️ No table-like markdown detected for {table_name} (skipping).")

        except Exception as e:
            print(f"⚠️ Error processing table {table_name}: {e}")

    updated_qa_count = state["qa_count"] + len(new_qa)
    updated_qa_per_book = state["qa_count_per_book"] + len(new_qa)
    print(f"📈 Generated {len(new_qa)} new QA pairs from tables. Book total: {updated_qa_per_book}")

    return {
        "qa_pairs": state["qa_pairs"] + new_qa,
        "qa_count": updated_qa_count,
        "qa_count_per_book": updated_qa_per_book,
    }


# -----------------------------------------------------------------------------
# 10. Workflow graph and execution (added quality check nodes)
# -----------------------------------------------------------------------------

workflow = StateGraph(PipelineState)
workflow.add_node("parse", RunnableLambda(parse_current_book_node))
workflow.add_node("table_relevance_check", RunnableLambda(table_relevance_check_node))
workflow.add_node("image_relevance_check", RunnableLambda(image_relevance_check_node))
workflow.add_node("generate_vqa", RunnableLambda(generate_vqa_node))
workflow.add_node("vqa_quality_check", RunnableLambda(vqa_quality_check_node))
workflow.add_node("generate_qa", RunnableLambda(generate_qa_node))
workflow.add_node("generate_table_qa", RunnableLambda(generate_table_qa_node))
workflow.add_node("qa_quality_check", RunnableLambda(qa_quality_check_node))


def advance_to_next_book(state: PipelineState) -> dict:
    return {"current_book_index": state["current_book_index"] + 1}

workflow.add_node("advance_book", RunnableLambda(advance_to_next_book))
workflow.set_entry_point("parse")
workflow.add_edge("parse", "table_relevance_check")
workflow.add_edge("table_relevance_check", "image_relevance_check")
workflow.add_edge("image_relevance_check", "generate_vqa")
workflow.add_edge("generate_vqa", "vqa_quality_check")
workflow.add_edge("vqa_quality_check", "generate_qa")
workflow.add_edge("generate_qa", "generate_table_qa")
workflow.add_edge("generate_table_qa", "qa_quality_check")


def book_router(state: PipelineState) -> str:
    is_last_book = state["current_book_index"] >= len(state["pdf_paths"]) - 1
    if is_last_book:
        print("✅ Last book processed. Finishing workflow.")
        return END
    print(f"🏁 Finished book {state['current_book_index'] + 1}. Moving to next...")
    return "next_book"

workflow.add_conditional_edges("qa_quality_check", book_router, {"next_book": "advance_book", END: END})
workflow.add_edge("advance_book", "parse")

app = workflow.compile()


# -----------------------------------------------------------------------------
# 11. Main execution block
# -----------------------------------------------------------------------------

def main():
    ensure_dir(Config.OUTPUT_DIR)
    ensure_dir(Config.IMAGE_CACHE_DIR)
    ensure_dir(Config.TABLE_CACHE_DIR)

    pdf_paths = [os.path.join(Config.BOOKS_DIR, f) for f in os.listdir(Config.BOOKS_DIR) if f.lower().endswith(".pdf")]
    if not pdf_paths:
        print(f"❌ No PDF files found in '{Config.BOOKS_DIR}'. Please add some books and try again.")
        return

    print(f"📚 Found {len(pdf_paths)} books to process.")

    initial_state = PipelineState(
        pdf_paths=pdf_paths,
        current_book_index=0,
        text_chunks=[],
        images_with_page_text=[],
        table_data=[],
        qa_pairs=[],
        vqa_pairs=[],
        qa_count=0,
        vqa_count=0,
        qa_count_per_book=0,
        vqa_count_per_book=0,
    )

    result = None
    print("\n🌟 Starting data generation workflow... Press Ctrl+C to interrupt and save progress.")
    try:
        final_state = app.invoke(initial_state, {"recursion_limit": 5000})
        result = final_state
    except KeyboardInterrupt:
        print("\n⏹️ Workflow interrupted by user. The script will attempt to save any completed data.")
        result = app.get_state(initial_state).values if 'app' in locals() else initial_state
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
    finally:
        if result:
            print("\n💾 Saving results...")

            qa_csv_path = os.path.join(Config.OUTPUT_DIR, "qa_pairs.csv")
            vqa_csv_path = os.path.join(Config.OUTPUT_DIR, "vqa_pairs.csv")

            if result.get("qa_pairs"):
                pd.DataFrame(result["qa_pairs"]).to_csv(qa_csv_path, index=False, encoding='utf-8')
                print(f"Saved {len(result['qa_pairs'])} QA pairs to {qa_csv_path}")

            if result.get("vqa_pairs"):
                pd.DataFrame(result["vqa_pairs"]).to_csv(vqa_csv_path, index=False, encoding='utf-8')
                print(f"Saved {len(result['vqa_pairs'])} VQA pairs to {vqa_csv_path}")

            convert_csv_to_jsonl(qa_csv_path, qa_csv_path.replace(".csv", ".jsonl"))
            convert_csv_to_jsonl(vqa_csv_path, vqa_csv_path.replace(".csv", ".jsonl"))

            print("\n📝 --- Final Report ---")
            print(f"Total QA pairs extracted:  {result.get('qa_count', 0)} 🚀")
            print(f"Total VQA pairs extracted: {result.get('vqa_count', 0)} 👁️")
            print(f"Grand Total:               {result.get('qa_count', 0) + result.get('vqa_count', 0)} 🎉")
            print("------------------------\n")
        else:
            print("No results were generated or an early error occurred.")


if __name__ == "__main__":
    main()