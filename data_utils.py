import json
import yaml
from collections import defaultdict
from pathlib import Path


def load_taxonomy(taxonomy_path):
    with open(taxonomy_path, 'r', encoding='utf-8') as f:
        taxonomy = json.load(f)
    taxonomy_dict = {}
    for key, value in taxonomy.items():
        taxonomy_dict[key] = ""
        for error in value:
            taxonomy_dict[key] = taxonomy_dict[key] + "\n" + error['label'] + ": " + error['description']
            # print(taxonomy_dict[key])

    return taxonomy_dict


def build_query(dataset, dataset_path, prompt_path, taxonomy_path, output_path, has_gt=True):
    querys = []
    existing_ids = set()
    if Path(output_path).exists():
        existing_ids = {json.loads(line)["answer_id"] for line in open(output_path, "r", encoding="utf-8")}
    questions_dict  = {}
    for question in dataset['questions']:
        question_id = question['question_id']
        questions_dict[question_id] = question

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
    
    taxonomy = load_taxonomy(taxonomy_path)

    for annotation in dataset['annotations']:
        sample = {}
        answer_id = annotation['answer_id']
        if answer_id in existing_ids:
            continue
        question_id = annotation['question_id']
        has_image = questions_dict[question_id]['requires_input_image']
        question_text = questions_dict[question_id]['query_en']
        taxonomy_text = taxonomy[annotation['category']]

        if has_image:
            sample['image_0'] = str(dataset_path / questions_dict[question_id]['input_image_path'])
            if has_gt:
                sample['query'] = prompts['image_with_gt']
                sample['image_1'] = str(dataset_path / questions_dict[question_id]['gt_image_path'])
                sample['image_2'] = str(dataset_path / annotation['image_path'])
            else:
                sample['query'] = prompts['image_no_gt']
                sample['image_1'] = str(dataset_path / annotation['image_path'])
        else:
            if has_gt:
                sample['query'] = prompts['no_image_with_gt']
                sample['image_0'] = str(dataset_path / questions_dict[question_id]['gt_image_path'])
                sample['image_1'] = str(dataset_path / annotation['image_path'])
            else:
                sample['query'] = prompts['no_image_no_gt']
                sample['image_0'] = str(dataset_path / annotation['image_path'])

        sample['query'] = sample['query'].replace("[[QUERY_EN]]", question_text)
        sample['query'] = sample['query'].replace("[[TAXONOMY_BULLETED]]", taxonomy_text)
        sample['answer_id'] = answer_id
        querys.append(sample)

    return querys


def dataset_statistics(master_json_path):
    with open(master_json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    correct_ans_count = 0
    incorrect_ans_count = 0
    total_ans_count   = 0
    error_counts = defaultdict(int)
    ans_per_question = defaultdict(int)

    for annotation in dataset['annotations']:
        total_ans_count += 1
        ans_per_question[annotation['question_id']] += 1
        if annotation['is_correct']:
            correct_ans_count += 1
        else:
            incorrect_ans_count += 1
            # error statistics
            for error in annotation['error_list']:
                error_counts[error['error_type']] += 1

    print(f"Total Answers: {total_ans_count}")
    print(f"Correct Answers: {correct_ans_count}")
    print(f"Incorrect Answers: {incorrect_ans_count}")
    print("Error Counts:")
    for error_label, count in error_counts.items():
        print(f"  {error_label}: {count}")
    print(f"Max Answer Counts: {max(ans_per_question.values())} Min Answet Counts: {min(ans_per_question.values())}")


# if __name__ == "__main__":
#     dataset_statistics(master_json_path="./data/SketchJudge_v1/master.json")