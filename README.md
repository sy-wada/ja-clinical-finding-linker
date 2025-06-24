# ja-clinical-finding-linker

This repository contains scripts used to build a LoRA adapter that extracts Japanese disease names and related abnormal findings from medical articles.

## Dataset
- **jp-disease-finding-dataset** – created using the scripts in this repository and available on [Hugging Face](https://huggingface.co/datasets/seiya/jp-disease-finding-dataset). It consists of pairs of medical text and JSON annotations describing diseases and their associated findings.

Sample data used in this repository is under `data/`.

## Model
A LoRA adapter trained with the scripts in this repository can be found at:
[Phi-4-mini-instruct-qlora-adapter-jp-disease-finding](https://huggingface.co/seiya/Phi-4-mini-instruct-qlora-adapter-jp-disease-finding)

## Workflow
### 1. Extract text from PDFs
OCR the source PDFs using [olmOCR](https://github.com/allenai/olm-ocr):
```bash
python -m olmocr.pipeline ./workspace \
  --pdfs data/pdfs/*.pdf \
  --model allenai/olmOCR-7B-0225-preview \
  --target_anchor_text_len 4000
```
The extracted text files become the input for the next step.

### 2. Generate annotations with an LLM
```
python llm_json_generator.py \
  --model <openai-model-name> \
  --input_dir <dir-with-text> \
  --output_dir <result-dir> \
  --prefix <file-prefix>
```
This script prompts the selected model using `json_generator_prompt.py` and saves the extracted results as JSON.

### 3. Fine-tune with LoRA
```
python trainer.py \
  --model_name <base-model> \
  --text_input_dir <text-dir> \
  --json_input_dir <annotation-dir> \
  --json_input_prefixes <comma-separated-prefixes> \
  --output_dir outputs
```
The script loads data created in step 1 and fine tunes the base model using unsloth. The resulting adapter is saved under the specified output directory.

## Notebook
`notebook/UsageExample_ExtractDiseaseFindings.ipynb` shows how to run inference with the trained adapter.
