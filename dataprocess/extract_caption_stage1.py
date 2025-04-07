import os
import json
from collections import defaultdict
from sklearn.model_selection import train_test_split
import random
from multiprocessing import Pool, cpu_count
from pdb import set_trace
import time
def extract_samples_from_json(file_path, directory):
    """Extract all unique sample types and their corresponding samples from a given JSON file."""
    samples = []
    
    if not os.path.isfile(file_path):
        print(f"File does not exist: {file_path}")
        return samples
    
    print(f"Reading file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data_list = json.load(f)
            
            for data in data_list:
                caption = data.get('caption')
                image = data.get('image')
                
                if caption and image:
                    relative_image_path = os.path.relpath(os.path.join(os.path.dirname(file_path), "images", image), start=directory)
                    abs_path = os.path.join(directory, relative_image_path)
                    if os.path.exists(abs_path):
                        samples.append({
                            "relative_image": relative_image_path,
                            "caption": caption,
                            "directory": os.path.dirname(file_path)
                        })
                    else:
                        print(f"Image path does not exist: {relative_image_path}")
                else:
                    print(f"Skipping invalid sample in {file_path}: image={image}, caption={caption}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    return samples


def generate_human_value(caption, index=1):
    """Generate a string with an image caption for human interaction in the specified format."""

    return f"Please describe this pathology image.\n<image>Caption#{index}:"


def create_combined_sample(sample, sample_id):
    """Create a combined sample object with associated captions and images for a single sample."""
    
    caption = sample["caption"]
    relative_image_path = sample["relative_image"]

    human_value = generate_human_value(caption, index=1)
    gpt_sample = caption

    conversations = [
        {"from": "human", "value": human_value},
        {"from": "gpt", "value": gpt_sample}
    ]

    combined_sample = {
        "sample_id": sample_id,
        "task_instruction_id": 1,
        "conversations": conversations,
        "image": [relative_image_path],
        "choice_list": None,
        "metadata": {
            "dataset": "Pathology",
            "task_instruction": ["Can you describe this pathological image?"],
            "question_type": "open-ended"
        },
    }

    return combined_sample


def process_file(file_path, root_directory):
    """Process a single file and return the extracted samples."""
    return extract_samples_from_json(file_path, root_directory)


# def process_all_samples_in_directory(root_directory, output_file):
#     """Process all samples from all data.json files in the root directory and its subdirectories, and save all valid samples."""
#     all_samples_by_dataset = defaultdict(list)

#     # Collect all file paths first
#     file_paths = []
#     for subdir, _, files in os.walk(root_directory):
#         if 'data.json' in files:
#             file_paths.append(os.path.join(subdir, 'data.json'))
    
#     # Process files sequentially (or use multiprocessing.Pool for parallel processing)
#     for file_path in file_paths:
#         result = process_file(file_path, root_directory)
#         if result:  # Ensure the result is not empty
#             all_samples_by_dataset[os.path.dirname(result[0]['directory'])].extend(result)

#     # Create combined samples ensuring they come from the same dataset
#     sample_counter = 0
#     all_combined_samples = []
#     for dataset_samples in all_samples_by_dataset.values():
#         for sample in dataset_samples:
#             key_dic = ["bone","cartilage", "Gleason grade 5","Gleason grade 1","Tubular Adenoma, High-Grade dysplasia","Hyperplastic Polyp",
#             "Non-viable tumor (intratumoral hemorrhage or necrosis or non-tumor tissue region)","Gleason grade 2","Areas of ulceration",
#             "Submucosal glands","Lamina muscularis mucosae","Lamina propria mucosae","Gleason 3, Stroma","Gleason 4, Stroma",
#             "The observed differentiation patterns include Gleason 5, Gleason 3, Gleason 4",
#             "The observed differentiation patterns include Gleason 3, Gleason 4",
#             "The observed differentiation patterns include Gleason 5, Gleason 4",
#             "The observed differentiation patterns include Gleason 5, Gleason 3"
#             ]
#             if sample["caption"]  in key_dic:
#                 continue
#             combined_sample = create_combined_sample(sample, sample_id=sample_counter + 1)
#             if combined_sample:
#                 all_combined_samples.append(combined_sample)
#                 sample_counter += 1

#     if not all_combined_samples:
#         print("No valid samples were found.")
#         with open(output_file, 'w', encoding='utf-8') as f:
#             json.dump([], f, indent=4, ensure_ascii=False)
#         return

#     # Save all samples to a JSON file
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(all_combined_samples, f, indent=4, ensure_ascii=False)

#     print(f"All valid samples saved to {output_file}.")

def process_all_samples_in_directory(root_directory, output_train_file, output_test_file,test_size=0.2, random_state=42):
    """Process all samples from all data.json files in the root directory and its subdirectories, and split them into training and testing sets."""
    all_samples_by_dataset = defaultdict(list)

    # Collect all file paths first
    file_paths = []
    for subdir, _, files in os.walk(root_directory):
        if 'data.json' in files:
            file_paths.append(os.path.join(subdir, 'data.json'))
    
    # Process files in parallel using multiprocessing.Pool
    with Pool(cpu_count()) as pool:
        processed_results = pool.starmap(process_file, [(file_path, root_directory) for file_path in file_paths])

    # Aggregate results from all processes by dataset
    for result in processed_results:
        all_samples_by_dataset[os.path.dirname(result[0]['directory'])].extend(result)

    # Create combined samples ensuring they come from the same dataset
    sample_counter = 0
    all_combined_samples = []
    for dataset_samples in all_samples_by_dataset.values():
        for sample in dataset_samples:
            key_dic = ["bone","cartilage", "Gleason grade 5","Gleason grade 1","Tubular Adenoma, High-Grade dysplasia","Hyperplastic Polyp",
            "Non-viable tumor (intratumoral hemorrhage or necrosis or non-tumor tissue region)","Gleason grade 2","Areas of ulceration",
            "Submucosal glands","Lamina muscularis mucosae","Lamina propria mucosae","Gleason 3, Stroma","Gleason 4, Stroma",
            "The observed differentiation patterns include Gleason 5, Gleason 3, Gleason 4",
            "The observed differentiation patterns include Gleason 3, Gleason 4",
            "The observed differentiation patterns include Gleason 5, Gleason 4",
            "The observed differentiation patterns include Gleason 5, Gleason 3"
            ]
            if sample["caption"]  in key_dic:
                continue
            combined_sample = create_combined_sample(sample, sample_id=sample_counter + 1)
            if combined_sample:
                all_combined_samples.append(combined_sample)
                sample_counter += 1

    if not all_combined_samples:
        print("No samples were processed because all datasets had insufficient samples.")
        # Save empty lists to the output files
        with open(output_train_file, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=4, ensure_ascii=False)
        with open(output_test_file, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=4, ensure_ascii=False)
        return
    print("^_^")

    grouped_samples = {}
    for sample in all_combined_samples:
        caption = sample["conversations"][0]["value"].split(":")[-1].strip()
        if caption not in grouped_samples:
            grouped_samples[caption] = []
        grouped_samples[caption].append(sample)
    # from pdb import set_trace
    # 
    train_samples = []
    test_samples = []
    for caption, samples in grouped_samples.items():
        train, test = train_test_split(samples, test_size=test_size, random_state=random_state, stratify=[s["metadata"]["dataset"] for s in samples])
        train_samples.extend(train)
        test_samples.extend(test)
    with open(output_train_file, 'w', encoding='utf-8') as f:
        json.dump(train_samples, f, indent=4, ensure_ascii=False)
    with open(output_test_file, 'w', encoding='utf-8') as f:
        json.dump(test_samples, f, indent=4, ensure_ascii=False)
    print(f"Training set saved to {output_train_file}.")
    print(f"Testing set saved to {output_test_file}.")
# Example usage
if __name__ == "__main__":
    root_directory = "/home/DATA2/cxh/Train_dataset/train_data"
    # output_file = os.path.join(root_directory, "val_zero.json")
    output_train_file = os.path.join(root_directory, "train.json")
    output_test_file = os.path.join(root_directory, "test.json")
    # output_val_file = os.path.join(root_directory, "val_few_shot5.json")
    test_size = 0.2  
    process_all_samples_in_directory(root_directory, output_train_file, output_test_file,test_size)
    # Process all samples from all subdirectories and save all valid samples to the output file
    # process_all_samples_in_directory(root_directory, output_file)