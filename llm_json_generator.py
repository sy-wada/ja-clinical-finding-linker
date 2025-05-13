import argparse
import json
from pathlib import Path
from openai import OpenAI
from tqdm.auto import tqdm
from batch_templates import samples, prompts, ExtractionResult
import time
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, default="gpt-4.1-mini-2025-04-14", 
                       help="Model name to use for extraction (e.g., gpt-4.1-mini-2025-04-14)")
    parser.add_argument('--input_dir', required=True,
                       help="Directory containing input files to process")
    parser.add_argument('--output_dir', required=True,
                       help="Directory where processed results will be saved")
    parser.add_argument('--prefix', required=True,
                       help="Prefix to filter input files (e.g., '102_')")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI()

    def set_additional_properties_false(schema):
        if isinstance(schema, dict):
            if schema.get("type") == "object":
                schema.setdefault("properties", {})
                schema["additionalProperties"] = False
                for prop in schema["properties"].values():
                    set_additional_properties_false(prop)
            if schema.get("type") == "array" and "items" in schema:
                set_additional_properties_false(schema["items"])
            for defs_key in ("definitions", "$defs"):
                if defs_key in schema:
                    for def_schema in schema[defs_key].values():
                        set_additional_properties_false(def_schema)
        return schema

    def remove_titles_and_descriptions_on_ref(schema):
        if isinstance(schema, dict):
            # Remove title/description when $ref exists
            if "$ref" in schema:
                schema.pop("title", None)
                schema.pop("description", None)
            # Recursively process all keys
            for key, value in list(schema.items()):
                remove_titles_and_descriptions_on_ref(value)
        elif isinstance(schema, list):
            for item in schema:
                remove_titles_and_descriptions_on_ref(item)
        return schema

    # Process schema recursively
    schema = ExtractionResult.model_json_schema()
    schema = set_additional_properties_false(schema)
    schema = remove_titles_and_descriptions_on_ref(schema)

    files = [path for path in sorted(input_dir.glob('*')) if path.name.startswith(args.prefix) and path.is_file()]
    failed_dir = output_dir / 'failed'
    failed_dir.mkdir(parents=True, exist_ok=True)
    progress_bar = tqdm(files, desc="Initializing...", unit="file", dynamic_ncols=True)
    for file_path in progress_bar:
        progress_bar.set_description(f"Checking {file_path.name}")
        output_path = output_dir / (file_path.stem + '.json')
        if output_path.exists():
            progress_bar.write(f"Skip: {output_path} already exists.")
            print(f"Skip: {output_path} already exists.")
            continue

        # Display file being processed
        progress_bar.set_description(f"Processing {file_path.name}")

        try:
            if 'json' in file_path.suffix:
                with open(file_path, encoding='utf-8') as f:
                    obj = json.load(f)
                article_text = obj['text']
            elif file_path.suffix == '.md':
                with open(file_path, encoding='utf-8') as f:
                    article_text = f.read()
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        user_prompt = prompts['user'].substitute(
            ArticleText=article_text,
            example1_input=prompts.get('example1_input', samples[0]['input']),
            example1_output=prompts.get('example1_output', samples[0]['output']),
            example2_input=prompts.get('example2_input', samples[1]['input']),
            example2_output=prompts.get('example2_output', samples[1]['output']),
        )
        system_prompt = prompts['system']

        temperature = 0.0
        success = False
        last_failed_content = None
        last_failed_exception = None
        while temperature <= 1.0:
            try:
                response = client.responses.create(
                    model=args.model,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "extraction_result",
                            "schema": schema,
                            "strict": True
                        }
                    },
                    temperature=round(temperature, 2),
                )
                content = response.output_text
                data = json.loads(content)
                ExtractionResult.model_validate(data)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"Success: {output_path} (temperature={temperature})")
                success = True
                break
            except Exception as e:
                print(f"Retry {file_path.name} (temperature={temperature}): {e}")
                # Save failed response content
                failed_file = failed_dir / f"{file_path.stem}_temp{round(temperature,2):.2f}.txt"
                try:
                    # Save content if available
                    if 'content' in locals():
                        with open(failed_file, 'w', encoding='utf-8') as f:
                            f.write(content)
                    else:
                        with open(failed_file, 'w', encoding='utf-8') as f:
                            f.write(str(e))
                except Exception as save_e:
                    print(f"Failed to save failed response: {save_e}")
                temperature += 0.1
                time.sleep(1)

        if not success:
            print(f"Failed: {file_path.name} (all temperatures)")

if __name__ == '__main__':
    main()
