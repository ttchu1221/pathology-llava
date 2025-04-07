import os
import json
from collections import defaultdict
from sklearn.model_selection import train_test_split
import random
from multiprocessing import Pool, cpu_count
from pdb import set_trace
import time

def extract_samples_from_json(file_path, directory):
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

def select_n_samples_from_same_dataset(samples,nums,n_per_batch,caption_kinds, blacklist):
    # set_trace()
    category_samples = defaultdict(list)
    for sample in samples:
        if sample['relative_image'] not in blacklist:
            category_samples[sample['caption']].append(sample)
    selected_samples = []
    
    for category, samples_in_cat in category_samples.items():
        caption_groups = defaultdict(list)
        last_samples =[]
        for sample in samples_in_cat:
            caption_groups[sample['caption']].append(sample)

        found_pair = False
        for caption, group in caption_groups.items():
            if len(group) >= nums:
                selected_samples.extend(random.sample(group, nums))  
                found_pair = True
                break
        
        if not found_pair:
            selected_samples.append(None)  
        if len(selected_samples) == n_per_batch:
            break
    successful_picks = [s for s in selected_samples if s is not None]
    return successful_picks if len(successful_picks) == n_per_batch  else None
# def generate_human_value(captions,n):
#     caption_strings = [
#         f"<image>Caption#{i+1}:{caption}" for i, caption in enumerate(captions[:-1])
#     ]
#     if captions:
#         caption_strings.append(f"Based on the types and characteristics provided by the first {len(captions)-1} images, please try to determine which type this new image most closely resembles?\n<image>Caption#{len(captions)}:")
#     caption_strings += [f"{chr(65 + i)}:{caption}" for i, caption in enumerate(captions[:-1:n])]
#     return "\n".join(caption_strings)
def generate_human_value(captions,n):
    """Generate a string with image captions for human interaction."""
    #set_trace()

    caption_strings = [
        f"{'<image>'*n}Caption#{i+1}:{caption}" for i, caption in enumerate(captions[:-1:n])
    ]
    if captions:
        caption_strings.append(f"Based on the Caption and characteristics provided by the first {len(captions)-1} images, please try to determine which Caption this new image most closely resembles?\n<image>Caption#{len(captions)//n+1}:You must choose your answer from the Choice List.\n Choice List:")
    caption_strings += [f"{chr(65 + i)}:{caption}" for i, caption in enumerate(captions[:-1:n])]
    return "\n".join(caption_strings)
def create_combined_sample(samples, sample_id,n):
    """Create a combined sample object with associated captions and images."""
    if len(samples) < 2:
        print("Not enough samples to create a combined sample.")
        return None
    captions = [sample["caption"] for sample in samples]
    images = [sample["relative_image"] for sample in samples]
    human_value = generate_human_value(captions,n)
    for i ,caption in enumerate (captions[:-1:n]):
        if caption == captions[-1]:
            gpt_sample = f"{chr(65 + i)}"
    path_parts = images[0].split('/')
    dataset = path_parts[0]
    conversations = [
        {"from": "human", "value": human_value},
        {"from": "gpt", "value": gpt_sample}
    ]
    combined_sample = {
        "sample_id": sample_id,
        "task_instruction_id": 3,
        "conversations": conversations,
        "image": images,
        "choice_list": None,
        "metadata": {
            "dataset": dataset,
            "task_instruction": ["Given the image-caption pairs for the first several images, can you provide a description for the last image based on these pairs?",
                                 "Based on the preceding medical images-caption, can you provide a professional description for the last medical image?",
                                 "Please select a description for the last pathology image by choosing directly from the descriptions given for the first several pathology images.\n",
                                 "Having viewed the images, can you use the information presented to answer the following question?"],
            "question_type": "open-ended"
        },
    }

    return combined_sample
def process_file(file_path, root_directory):
    return extract_samples_from_json(file_path, root_directory)

def combine_samples_from_same_dataset(samples_dict, n,caption_kinds, sample_counter, blacklist,s_blacklist):
    combined_samples = []
    #key_dic = []
    # key_dic = ["bone","cartilage"]
    n_per_batch = n*len(caption_kinds)+1
    last__samples = []
    for dataset_samples in samples_dict.values():
        caption_samples = defaultdict(list)
        for sample in dataset_samples:
            caption_samples[sample['caption']].append(sample)
        for caption, samples in caption_samples.items():
            if n>=4:
                last__samples.extend(samples[:50])
                # if len(samples) >=4000 :
                #     cutoff_index = max(1, int(len(samples) * 0.995))
                #     last__samples.extend(samples[cutoff_index:])
                # elif len(samples) >=1000 and len(samples) <4000:
                #     cutoff_index = max(1, int(len(samples) * 0.99))
                #     last__samples.extend(samples[cutoff_index:])
                # elif len(samples) >=500 and len(samples) <1000:
                #     cutoff_index = max(1, int(len(samples) * 0.96))
                #     last__samples.extend(samples[cutoff_index:])
                # else:
                #     cutoff_index = max(1, int(len(samples) * 0.9))
                #     last__samples.extend(samples[cutoff_index:])
            else:
                seen_relative_images = set()  # 用于追踪已添加的 relative_image

                s_samples = []
                for sample in samples:
                    relative_image = sample['relative_image']
                    if relative_image in s_blacklist and relative_image not in seen_relative_images:
                        s_samples.append(sample)
                        seen_relative_images.add(relative_image)  # 标记为已见       
                s_samples.sort(key=lambda x: x['relative_image'])  # Sort to ensure consistent order
                last__samples.extend(s_samples)
            for last_samples in last__samples:
                blacklist.add(last_samples['relative_image'])
            last_samples1 = [dict(s) for s in set(frozenset(d.items()) for d in last__samples)]
    for dataset_samples in samples_dict.values():
        c = 0
        s = time.time()
        
        while True:
            c += 1
            selected_samples = select_n_samples_from_same_dataset(dataset_samples, n,n_per_batch-1,caption_kinds, blacklist)
            if not selected_samples or  not last_samples1 or len(selected_samples) < n_per_batch - 1:
                break
            last_sample = random.choice(last_samples1)

            selected_samples.append(last_sample)
            #set_trace()
            last_samples1.remove(last_sample)
            combined_sample = create_combined_sample(selected_samples, sample_id=sample_counter + 1,n=n)
            if combined_sample:
                combined_samples.append(combined_sample)
                sample_counter += 1
            
            # for sample in selected_samples[:-1]:
            #     if sample in dataset_samples:
            #         if sample['caption'] not in key_dic:
            #             dataset_samples.remove(sample)
            
            if c % 1000 == 0:
                e = time.time()
                print(e - s)
   
    return combined_samples, sample_counter
def process_all_samples_in_directory(root_directory, blacklist_file, output_test_file,output_val_file,n,test_size=0.2, random_state=42):
    all_samples_by_dataset = defaultdict(list)
    blacklist = set()  # Initialize the blacklist for tracking the last samples
    s_blacklist = set()
    if not os.path.isfile(blacklist_file):
        print(f"File does not exist: {blacklist_file}")

    print(f"Reading file: {blacklist_file}")
    try:
        with open(blacklist_file, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
            for entry in data_list:
                images = entry.get('image')
                if images and isinstance(images, list):  # Ensure 'image' is a list
                    # for image in images:
                        image = images[-1]
                        if image:  # Check if image path is not empty
                            s_blacklist.add(image)  # Directly add the image path to the set
                else:
                    print(f"Skipping invalid sample in {blacklist_file}: missing or incorrect format of image field")

    except Exception as e:
        print(f"Error reading {blacklist_file}: {e}")
    # Collect all file paths first
    file_paths = []
    for subdir, _, files in os.walk(root_directory):
        if 'data.json' in files:
            file_paths.append(os.path.join(subdir, 'data.json'))
    
    with Pool(cpu_count()) as pool:
        processed_results = pool.starmap(process_file, [(file_path, root_directory) for file_path in file_paths])

    for result in processed_results:
        all_samples_by_dataset[os.path.dirname(result[0]['directory'])].extend(result)

    sample_counter = 0
    all_combined_samples = []
    

    for dataset_samples in all_samples_by_dataset.values():
        
        caption_kinds = set()

        for item in dataset_samples:
            if 'caption' in item:
                caption_kinds.add(item['caption'])
        combined_samples, sample_counter = combine_samples_from_same_dataset({os.path.dirname(dataset_samples[0]['directory']): dataset_samples}, n,caption_kinds,sample_counter, blacklist,s_blacklist)
        all_combined_samples.extend(combined_samples)

    print("^_^")

    with open(output_val_file, 'w', encoding='utf-8') as f:
        json.dump(all_combined_samples, f, indent=4, ensure_ascii=False)

    print(f"valing_few_shot_{n} set saved to {output_val_file}.")
if __name__ == "__main__":
    # Define the root directory and output file paths
    import numpy as np


    np.random.seed(42)  # 设置随机种子
    root_directory = "/home/DATA2/cxh/Eval_data"
    blacklist_file = "/home/DATA2/cxh/Eval_data/val_4_shot_1.json"
    #blacklist_file = "/home/DATA2/cxh/Train_dataset/train_data/train.json"
    output_test_file = os.path.join(root_directory, "test_stage6.jsson")
    n=4
    output_val_file = os.path.join(root_directory, "val_4_shot_1.json")
    test_size = 0.2  
    process_all_samples_in_directory(root_directory, blacklist_file, output_test_file,output_val_file,n,test_size)