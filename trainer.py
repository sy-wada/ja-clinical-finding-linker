# -*- coding: utf-8 -*- 
import os
import gc
# os.environ["DISABLE_UNSLOTH_COMPILE"] = "1"
# # さらに他のコンパイル関連設定も無効化
# os.environ["TORCH_COMPILE"] = "0"
# os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
# os.environ["TORCH_LOGS"] = "+dynamo"
# os.environ["TORCHDYNAMO_VERBOSE"] = "1"
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import re
import json
from typing import Any
import argparse
from string import Template
from pathlib import Path

from unsloth import (
    FastLanguageModel,
    is_bfloat16_supported
)
import torch
from trl import SFTTrainer
from transformers import (
    TrainingArguments,
    PreTrainedTokenizer,
    DataCollatorForSeq2Seq,
    TrainerCallback
)
from pydantic import ValidationError
from json import JSONDecodeError

from datasets import Dataset

from templates_no_desc import samples, prompts, ExtractionResult

class EmptyCacheCallback(TrainerCallback):
    def __init__(self, every_n_steps: int = 10):
        self.every_n_steps = every_n_steps
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every_n_steps == 0:
            gc.collect(); torch.cuda.empty_cache()

# データセット作成
def format_prompt(
    data: dict[str, str],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    sys_prompt: str,
    user_prompt: Template,
    samples: list[dict[str, str]],
    ) -> dict[str, torch.Tensor]:
    output_text = data['output_text']
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt.substitute(ArticleText=data['article_text'],
                                                           example1_input=samples[0]['input'],
                                                           example1_output=samples[0]['output'],
                                                           example2_input=samples[1]['input'],
                                                           example2_output=samples[1]['output'])},
        {"role": "assistant", "content": output_text}
    ]
    
    # ① フルプロンプト（システム＋ユーザー＋アシスタント）を文字列として作成
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    tokenized_full = tokenizer(
        full_prompt,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # input_ids をコピーして labels を作成
    labels = tokenized_full["input_ids"].clone()
    
    # ② ユーザー部分（システム＋ユーザー）の文字列を作成
    prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False)
    # 同じ max_length, truncation を適用してトークナイズする（paddingは不要）
    tokenized_prompt = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    # ユーザー部分のトークン数を取得
    prompt_token_count = tokenized_prompt["input_ids"].shape[1]
    
    # ③ full prompt の先頭 prompt_token_count 個をマスク（loss計算対象外にする）
    # ※ 注意: full prompt が max_length に達している場合、
    #     ユーザー部分が max_length 以上になってしまうので、その場合は全体をマスクします。
    if prompt_token_count >= max_length:
        labels[:] = -100
    else:
        labels[0, :prompt_token_count] = -100

    # ④ loss計算対象部分（labelsが-100でない部分）のみをデコードしてExtractionResultでバリデーション
    loss_token_indices = (labels[0] != -100).nonzero(as_tuple=True)[0]
    if len(loss_token_indices) > 0:
        start = loss_token_indices[0].item()
        end = loss_token_indices[-1].item() + 1
        output_token_ids = tokenized_full["input_ids"][0][start:end]
        output_text_candidate = tokenizer.decode(output_token_ids, skip_special_tokens=True)
        try:
            json_data = json.loads(output_text_candidate)
            ExtractionResult.model_validate(json_data)
            eligible = True
        except Exception:
            eligible = False
    else:
        eligible = False

    return {
        "input_ids": tokenized_full["input_ids"][0],
        "attention_mask": tokenized_full["attention_mask"][0],
        "labels": labels[0],
        "eligible": eligible,
        # "output_text": output_text
    }

def filter_valid_examples(example):
    import numpy as np
    
    # Numpyの配列に変換して処理
    labels = np.array(example['labels'])
    
    # 全てマスクされているか
    all_masked = np.all(labels == -100)
    
    # 有効なトークン数をカウント
    valid_token_count = np.sum(labels != -100)
    
    # 最低限必要な有効トークン数
    min_valid_tokens = 5
    
    # 条件を満たすかどうか
    example['is_valid'] = not all_masked and valid_token_count >= min_valid_tokens and example.get('eligible', False)
    
    return example

def remove_finding_description(obj: dict[str, Any]) -> dict[str, Any]:
    for case in obj['results']:
        for item in case['abnormal_findings_caused_by_the_disease']:
            del item['finding_description']
    return obj

def main(
    model_name: str = None,
    text_input_dir: Path = None,
    json_input_dir: Path = None,
    json_input_prefixes: list[str] = None,
    output_dir: Path = None,
    adapter_dir: str = None,
    random_state: int = 3407,
    lora_rank: int = 16, # LoRA のランク: Suggested 8, 16, 32, 64, 128
    epoch: int = 1,
    learning_rate: float = 1e-4,
    max_seq_length: int = 8192,
    dtype: str = None,
    load_in_4bit: bool = True,
    fast_inference: bool = False
    ):
    # model, tokenizer = FastLanguageModel.from_pretrained(
    #     model_name = model_name,
    #     max_seq_length = max_seq_length,
    #     dtype = torch.bfloat16,
    #     load_in_4bit = load_in_4bit,
    #     device_map="auto",
    #     fix_tokenizer=True,
    #     trust_remote_code=False,
    #     use_gradient_checkpointing="unsloth",
    #     fast_inference=fast_inference,
    #     gpu_memory_utilization=0.9,
    #     float8_kv_cache=False,
    #     random_state=random_state,
    #     max_lora_rank=64,
    #     disable_log_stats=True,
    # )
    # 高速化狙い
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = torch.bfloat16,
        load_in_4bit = load_in_4bit,
        device_map="auto",
        fix_tokenizer=True,
        trust_remote_code=False,
        use_gradient_checkpointing="unsloth",   # Falseは速度優先。"unsloth"はメモリ効率優先。
        fast_inference=fast_inference,
        gpu_memory_utilization=0.9,
        float8_kv_cache=False,
        random_state=random_state,
        max_lora_rank=lora_rank,
        disable_log_stats=True,
    )
    # Phi-4には元々設定が存在していない。つけても効果はない？
    model.flash_attn = True
    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = lora_rank,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        # use_gradient_checkpointing = False, # True or "unsloth" for very long context
        random_state = random_state,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    print(f"{model.flash_attn=}")

    # jsonlファイルからデータを抽出し、Dataset形式に変換
    json_input_files = [path for path in sorted(json_input_dir.glob('*.json')) if path.name.split('_')[0] in json_input_prefixes]
    records = []
    for json_file_path in json_input_files:
        text_file_path = text_input_dir / (json_file_path.stem + '.jsonl')
        try:
            with open(text_file_path, encoding='utf-8') as f:
                obj = json.load(f)
            article_text = obj['text']
            obj = json.load(open(json_file_path, encoding='utf-8'))
            # objからfinding_descriptionを取り除く
            obj = remove_finding_description(obj)
            records.append({
                "file_stem": json_file_path.stem,
                "article_text": article_text,
                "output_text": json.dumps(obj, ensure_ascii=False)})
        except Exception as e:
            print(f"Error reading {json_file_path}: {e}")
            continue
    if not records:
        print("No valid input files found.")
        return
    
    dataset = Dataset.from_list(records)
    # 異常なデータがクラッシュの原因？調査のための挿入
    # import pandas as pd
    # df = pd.DataFrame(records)
    # df['len_article_text'] = df['article_text'].apply(lambda x: len(tokenizer.encode(x)))
    # df['len_output_text'] = df['output_text'].apply(lambda x: len(tokenizer.encode(x)))
    # df['total_len'] = df['len_article_text'] + df['len_output_text']
    # print(df[['len_article_text', 'len_output_text', 'total_len']].sort_values(by='total_len', ascending=False).head(20))
    # # raise ValueError("stop here")
    # dataset = Dataset.from_pandas(df[df['total_len'] >= 25000], preserve_index=False)

    # データセットの前処理
    preprocess_kwargs = {
        'max_length': max_seq_length,
        'tokenizer': tokenizer,
        'sys_prompt': prompts['system'],
        'user_prompt': prompts['user'],
        'samples': samples
    }
    train_dataset = dataset.map(
        format_prompt,
        fn_kwargs=preprocess_kwargs,
        batched=False,
    )

    # 有効性フラグを追加
    train_dataset = train_dataset.map(filter_valid_examples)

    # 無効なサンプルをフィルタリング
    filtered_dataset = train_dataset.filter(lambda example: example['is_valid'])

    # フィルタリング前後のデータ数を表示して確認
    print(f"元のデータセットサイズ: {len(train_dataset)}")
    print(f"フィルタリング後のデータセットサイズ: {len(filtered_dataset)}")
    
    # eligibleフラグの統計情報を表示
    eligible_count = sum(1 for example in train_dataset if example.get('eligible', False))
    print(f"ExtractionResultモデルでバリデーション可能なデータ数: {eligible_count}")
    print(f"最終的に学習に使用されるデータ数: {len(filtered_dataset)}")

    data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True, max_length=max_seq_length)

    # Trainer の初期化    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=filtered_dataset,  # フィルタリング済みデータセット
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=True,  # 短いシーケンスの場合は packing を有効にすると高速化可能
        data_collator=data_collator,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_ratio=0.05,
            # torch_compile=False,
            num_train_epochs=epoch,  # 必要に応じてエポック数を設定してください
            learning_rate=learning_rate, # We normally suggest 2e-4, 1e-4, 5e-5, 2e-5 as numbers to try
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=5,
            logging_first_step=True,
            optim="adamw_8bit",
            torch_empty_cache_steps=5,
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=random_state,
            output_dir=output_dir,
            save_strategy="no"
        ),
        # callbacks=[EmptyCacheCallback(every_n_steps=1)],
    )
    lora_dir = output_dir / adapter_dir

    trainer_stats = trainer.train()

    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    # `save_pretrained_merged`は潜在的にエラーが生じる。adapterのみを保存してvllmでよみこむ方針とする。
    # model.save_pretrained_merged(lora_dir, tokenizer, save_method="merged_4bit", safe_serialization=True)
    # model.save_pretrained_merged(lora_dir / 'lora', tokenizer, save_method="lora", safe_serialization=True)
 
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--text_input_dir", type=Path, required=True)
    parser.add_argument("--json_input_dir", type=Path, required=True)
    parser.add_argument("--json_input_prefixes", type=str, required=True)
    parser.add_argument("--output_dir", type=Path, default="./outputs")
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--random_state", type=int, default=3407)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    args = parser.parse_args()
    model_name = args.model_name
    train_vols = args.json_input_prefixes.split(',')

    adapter_dir = f"{model_name.replace('/', '_')}/{'_'.join(train_vols)}/e{args.epoch}.lr{args.learning_rate}.lora_rank{args.lora_rank}"
    if args.output_dir.joinpath(adapter_dir).exists():
        print(f"skip {adapter_dir}")
    else:
        main(
            model_name=model_name,
            text_input_dir=args.text_input_dir,
            json_input_dir=args.json_input_dir,
            json_input_prefixes=train_vols,
            max_seq_length=args.max_seq_length,
            epoch=args.epoch,
            learning_rate=args.learning_rate,
            lora_rank=args.lora_rank,
            output_dir=args.output_dir,
            adapter_dir=adapter_dir,
            fast_inference=False,
        )
