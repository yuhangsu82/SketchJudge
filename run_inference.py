from __future__ import annotations
import json, argparse
from pathlib import Path
from typing import Dict, Any, Iterator
from tqdm import tqdm
from models import make_model
from data_utils import build_query


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model-path', type=str, default='', help="local model path or huggingface model name")
    p.add_argument('--prompt-path', type=str, default='./prompts/prompt_baseline.yaml', help="custom prompt templates path")
    p.add_argument("--model-type", default="qwen", help="key in MODEL_REGISTRY, e.g. qwen | text")
    p.add_argument("--model-name", type=str, default='', help="for GPT model, the model name like gpt-4o")
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--output", required=True, help="destination JSONL")
    p.add_argument("--with-reference", action="store_true", default=True, help="Use reference answers (default)")
    p.add_argument("--no-reference", action="store_false", dest="with_reference", help="Disable reference answers")
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument('--api-key', type=str, default='')
    args = p.parse_args()

    if args.model_type == "gpt":
        model = make_model(
            args.model_type,
            model_name=args.model_name,
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
            api_key=args.api_key,
        )
    else:
        model = make_model(
            args.model_type,
            model_path=args.model_path,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    promt_path = args.prompt_path
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    master_file = args.dataset_path + "/master.json"
    taxonomy_file = args.dataset_path + "/taxonomy.json"

    with open(master_file, "r", encoding="utf-8") as data_file:
        dataset = json.load(data_file)
        print("Dataset loaded.")

    querys = build_query(
        dataset,
        dataset_path=dataset_path,
        prompt_path=promt_path,
        taxonomy_path=taxonomy_file,
        output_path=out_path,
        has_gt=args.with_reference
    )

    with open(out_path, "a", encoding="utf-8") as fout:
        for s in tqdm(querys, desc="Inference"):
            s["response"] = model.generate_from_sample(s)
            fout.write(json.dumps(s, ensure_ascii=False) + "\n")

    print("✓ Done – results saved to", out_path.resolve())


if __name__ == "__main__":
    main()
