import os
import json
from collections import defaultdict
from sklearn.model_selection import train_test_split
import random
from multiprocessing import Pool, cpu_count
from pdb import set_trace
import time
import copy 
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



def select_n_samples_from_same_dataset(samples,nums,n_per_batch,caption_kinds, blacklist):
    """
    Select n samples from the same dataset ensuring they have unique captions.
    The last sample can have a caption that duplicates an earlier one within the batch but not in other batches.
    """

    category_samples = defaultdict(list)
    for sample in samples:
        if sample['relative_image'] not in blacklist:
            # if category_samples[sample['caption']] == "The observed differentiation patterns include Gleason 5, Gleason 3, Gleason 4":
            #     set_trace()
            category_samples[sample['caption']].append(sample)
    selected_samples = []
    
    for category, samples_in_cat in category_samples.items():
        # Group samples by their captions within the category
        caption_groups = defaultdict(list)
        # first_groups = defaultdict(list)
        last_samples =[]
        for sample in samples_in_cat:
            caption_groups[sample['caption']].append(sample)
        #set_trace()
        # Try to pick two samples with the same caption
        found_pair = False
        for caption, group in caption_groups.items():
            if len(group) >= nums:
                selected_samples.extend(random.sample(group, nums))  # Take the first two samples with this caption
                found_pair = True
                break
        
        if not found_pair:
            # Could not find two samples with the same caption in this category
            selected_samples.append(None)  # Indicate failure for this category
        if len(selected_samples) == n_per_batch:
            break
    # Check if we have successfully picked samples from n categories
    successful_picks = [s for s in selected_samples if s is not None]
    #set_trace()
    return successful_picks if len(successful_picks) == n_per_batch  else None


# def select_n_samples_from_same_dataset(samples, n, blacklist):
#     """
#     Select n samples from the same dataset ensuring they have unique captions.
#     The last sample can have a caption that duplicates an earlier one within the batch but not in other batches.
#     """
#     selected_samples = []
#     used_captions = set()
    
#     # Create a copy of the list to shuffle without modifying the original list
#     shuffled_samples = [sample for sample in samples if sample['relative_image'] not in blacklist]
#     random.shuffle(shuffled_samples)

#     for sample in shuffled_samples:
#         if len(selected_samples) < n - 1:  # Leave space for the last sample
#             if sample['caption'] not in used_captions:
#                 selected_samples.append(sample)
#                 used_captions.add(sample['caption'])
#         elif len(selected_samples) == n - 1:  # Add the last sample without checking for uniqueness within the batch
#             selected_samples.append(sample)
#             break

#     return selected_samples if len(selected_samples) == n else None
QUESTION_SET = [
"Can you figure out which caption best matches this new image?",
"Try to identify the caption that is most similar to this new picture.",
"Which caption do you think fits this image the closest?",
"Please analyze and decide which caption aligns best with this new image.",
"Could you determine the caption that resembles this picture the most?",
"Match this new image with the caption it closely relates to.",
"What caption would you say this image most closely corresponds to?",
"Help me find the caption that best suits this new image.",
"Assess which caption this image is most closely associated with.",
"Which caption comes closest to describing this new image?"]
padding_choice =["Gastric mucosa","Lamina propria mucosae","Adventitial tissue",
            "Lamina muscularis mucosae","Areas of ulceration","Submucosal glands",
            "Regression areas","Muscularis propria","Submucosa", "Oesophageal mucosa"]
def generate_human_value(captions,n):
    """Generate a string with image captions for human interaction."""
    #set_trace()
    caption_strings = [f"Please first understand the these image descriptions I provide examples.\nexamples:"]
    
    caption_strings += [
        f"<image>Caption#{i+1}:{caption}" for i, caption in enumerate(captions[:-1])
    ]
    if captions:
        caption_strings.append(f"\nplease try to determine which Caption this new image most closely resembles?<image>Caption#You must choose your answer from the Choice List.\n Choice List:")
    cap=copy.deepcopy(captions)
    while len(cap[:-1])<11:
        random_choice = random.choice(padding_choice)
        if random_choice not in cap:
            cap.insert(-1, random_choice)
    shuffle_samples =cap[:-1]
    random.shuffle(shuffle_samples)  
    caption_strings += [f"{chr(65 + i)}:{caption}" for i, caption in enumerate(shuffle_samples)]
    choice = [caption for i, caption in enumerate(shuffle_samples)]
    return "\n".join(caption_strings),choice
    #return ' '.join(caption_strings)
# def generate_human_value(captions,n):
#     """Generate a string with image captions for human interaction."""
#     #set_trace()
#     caption_strings = [f"Please first understand the these image descriptions I provide examples.\nexamples:"]
    
#     caption_strings += [
#         f"<image>Caption#{i+1}:{caption}" for i, caption in enumerate(captions[:-1])
#     ]
#     if captions:
#         caption_strings.append(f"\nplease try to determine which Caption this new image most closely resembles?<image>Caption#You must choose your answer from the Choice List.\n Choice List:")
#     caption_strings += [f"{chr(65 + i)}:{caption}" for i, caption in enumerate(captions[:-1])]
#     return "\n".join(caption_strings)
    # return ' '.join(caption_strings)
def create_combined_sample(samples, sample_id,n):
    """Create a combined sample object with associated captions and images."""
    if len(samples) < 2:
        print("Not enough samples to create a combined sample.")
        return None
    if len(samples) > 1:  # 确保列表中有多个元素
        last_sample = samples[-1]  # 提取最后一条数据
        remaining_samples = samples[:-1]  # 获取除最后一条外的数据
        random.shuffle(remaining_samples)  # 打乱剩余部分
        samples = remaining_samples + [last_sample]  # 重新组合
    
    captions = [sample["caption"] for sample in samples]
    
    images = [sample["relative_image"] for sample in samples]

    human_value,choice = generate_human_value(captions,n)
    for i ,caption in enumerate (choice):
        if caption == captions[-1]:
            gpt_sample = chr(65 + i)
    # set_trace()
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
    """Process a single file and return the extracted samples."""
    return extract_samples_from_json(file_path, root_directory)


def combine_samples_from_same_dataset(samples_dict, n,caption_kinds, sample_counter, blacklist,s_blacklist):
    combined_samples = []
    # key_dic = ["bone","cartilage", "Gleason grade 5","Gleason grade 1","Tubular Adenoma, High-Grade dysplasia","Hyperplastic Polyp",
    #         "Non-viable tumor (intratumoral hemorrhage or necrosis or non-tumor tissue region)","Gleason grade 2","Areas of ulceration",
    #         "Submucosal glands","Lamina muscularis mucosae","Lamina propria mucosae","Gleason 3, Stroma","Gleason 4, Stroma",
    #         "The observed differentiation patterns include Gleason 5, Gleason 3, Gleason 4",
    #         "The observed differentiation patterns include Gleason 3, Gleason 4",
    #         "The observed differentiation patterns include Gleason 5, Gleason 4",
    #         "The observed differentiation patterns include Gleason 5, Gleason 3"
    #         ]
    key_dic = ["Areas of ulceration","Gleason 3, Stroma","Gleason 4, Stroma","non-cancer",
            "The observed differentiation patterns include Gleason 5, Gleason 3, Gleason 4",
            "The observed differentiation patterns include Gleason 5, Gleason 4",
            "The observed differentiation patterns include Gleason 3, Gleason 4",
            "The observed differentiation patterns include Gleason 5, Gleason 3"
            ]
    # key_dic = []
    n_per_batch = n*len(caption_kinds)+1
    last__samples = []
    for dataset_samples in samples_dict.values():
        caption_samples = defaultdict(list)
        for sample in dataset_samples:
            caption_samples[sample['caption']].append(sample)
        
        for caption, samples in caption_samples.items():

            process_sample = any(sample['relative_image'] in s_blacklist for sample in samples)
            # from pdb import set_trace
            # if caption  in "Stroma":
            #     set_trace()
            # if process_sample:
            #     s_samples = [sample for sample in samples if sample['relative_image'] not in s_blacklist]
            #     s_samples.sort(key=lambda x: x['relative_image'])  # Sort to ensure consistent order
            #     if len(s_samples) >=2000:
            #         cutoff_index = max(1, int(len(s_samples) * 0.9))
            #         last__samples.extend(s_samples[cutoff_index:])
            #     elif len(s_samples) >=500 and len(s_samples) <2000:
            #         cutoff_index = max(1, int(len(s_samples) * 0.5))
            #         last__samples.extend(s_samples[cutoff_index:])
            #     else :
            #         last__samples.extend(s_samples)
            # else :
                
            #     if len(samples) >=10000:
            #         cutoff_index = max(1, int(len(samples) * 0.95))
            #         last__samples.extend(samples[cutoff_index:])
            #     elif len(samples) >=5000 and len(samples) <10000:
            #         cutoff_index = max(1, int(len(samples) * 0.9))
            #         last__samples.extend(samples[cutoff_index:])
            #     elif len(samples) >=1000 and len(samples) <5000:
            #         cutoff_index = max(1, int(len(samples) * 0.8))
            #         last__samples.extend(samples[cutoff_index:])
            #     elif len(samples) >=500 and len(samples) <1000:
            #         cutoff_index = max(1, int(len(samples) * 0.7))
            #         last__samples.extend(samples[cutoff_index:])
            #     else:
            #         cutoff_index = max(1, int(len(samples) * 0.5))
            #         last__samples.extend(samples[cutoff_index:])
            if process_sample:
                s_samples = [sample for sample in samples if sample['relative_image'] not in s_blacklist]
                s_samples.sort(key=lambda x: x['relative_image'])  # Sort to ensure consistent order
                #cutoff_index = max(1, int(len(s_samples) * 0.5))
                last__samples.extend(s_samples)
            else:
                cutoff_index = max(1, int(len(samples) * 0.7))
                last__samples.extend(samples[cutoff_index:])
            for last_samples in last__samples:
                blacklist.add(last_samples['relative_image'])
            last_samples1 = [dict(s) for s in set(frozenset(d.items()) for d in last__samples)]
            #caption_samples[caption] = s_samples[:cutoff_index]
    # set_trace()
    for dataset_samples in samples_dict.values():
        c = 0
        s = time.time()
        
        while True:
            c += 1
            #set_trace()
            selected_samples = select_n_samples_from_same_dataset(dataset_samples, n,n_per_batch-1,caption_kinds, blacklist)
            if not selected_samples or  not last_samples1 or len(selected_samples) < n_per_batch - 1:
                break
            last_sample = random.choice(last_samples1)

                # last__samples[last_sample_caption].remove(last_sample)
            # else:
            #     # If no more last 20% samples are available, choose from remaining samples
            #     last_sample = select_n_samples_from_same_dataset(dataset_samples, 1,1, caption_kinds,blacklist)[0]
            
            selected_samples.append(last_sample)
            #set_trace()
            last_samples1.remove(last_sample)
            combined_sample = create_combined_sample(selected_samples, sample_id=sample_counter + 1,n=n)
            if combined_sample:
                combined_samples.append(combined_sample)
                sample_counter += 1
                
                # Add the last sample to the blacklist to ensure it doesn't appear in the front of other batches
                # last_sample = selected_samples[-1]
                # blacklist.add(last_sample['relative_image'])
            
            # for sample in selected_samples[:-1]:
            #     if sample in dataset_samples:
            #         if sample['caption'] not in key_dic:
            #             dataset_samples.remove(sample)
            
            if c % 1000 == 0:
                e = time.time()
                print(e - s)
   
    # for dataset_samples in samples_dict.values():
    #     c= 0
    #     s =time.time()
    #     while True:
    #         c +=1
    #         selected_samples = select_n_samples_from_same_dataset(dataset_samples, n,n_per_batch,caption_kinds, blacklist)
            
    #         if not selected_samples or len(selected_samples) < n_per_batch:
    #             break
            
    #         combined_sample = create_combined_sample(selected_samples, sample_id=sample_counter + 1)
    #         if combined_sample:
    #             combined_samples.append(combined_sample)
    #             sample_counter += 1
    #             # Add the last sample to the blacklist to ensure it doesn't appear in the front of other batches
    #             last_sample = selected_samples[-1]
    #             blacklist.add(last_sample['relative_image'])
    #         for sample in selected_samples[:-1]:
    #             if sample in dataset_samples:
    #                 if sample['caption'] not in key_dic:
    #                     dataset_samples.remove(sample)
            
    #         if c%1000 ==0:
    #             e =time.time()
    #             print(e-s)
            
    return combined_samples, sample_counter



def process_all_samples_in_directory(root_directory, blacklist_file, output_test_file,output_val_file,n,test_size=0.2, random_state=42):
    """Process all samples from all data.json files in the root directory and its subdirectories, and split them into training and testing sets."""
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
                    for image in images:
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
        
        caption_kinds = set()

        # 假设data是一个列表，其中每个元素都有一个'caption'字段
        for item in dataset_samples:
            if 'caption' in item:
                caption_kinds.add(item['caption'])
        combined_samples, sample_counter = combine_samples_from_same_dataset({os.path.dirname(dataset_samples[0]['directory']): dataset_samples}, n,caption_kinds,sample_counter, blacklist,s_blacklist)
        all_combined_samples.extend(combined_samples)

    # if not all_combined_samples:
    #     print("No samples were processed because all datasets had insufficient samples.")
    #     # Save empty lists to the output files
    #     with open(output_train_file, 'w', encoding='utf-8') as f:
    #         json.dump([], f, indent=4, ensure_ascii=False)
    #     with open(output_test_file, 'w', encoding='utf-8') as f:
    #         json.dump([], f, indent=4, ensure_ascii=False)
    #     return
    print("^_^")

    with open(output_val_file, 'w', encoding='utf-8') as f:
        json.dump(all_combined_samples, f, indent=4, ensure_ascii=False)
    # grouped_samples = {}
    # for sample in all_combined_samples:
    #     caption = sample["conversations"][0]["value"].split(":")[-1].strip()
    #     if caption not in grouped_samples:
    #         grouped_samples[caption] = []
    #     grouped_samples[caption].append(sample)
    # val_samples = []
    # from pdb import set_trace
    # 
    # train_samples = []
    # test_samples = []
    # for caption, samples in grouped_samples.items():
    #     train, test = train_test_split(samples, test_size=test_size, random_state=random_state, stratify=[s["metadata"]["dataset"] for s in samples])
    #     train_samples.extend(train)
    #     test_samples.extend(test)
    # for caption, samples in grouped_samples.items():
    #     train, test = train_test_split(samples, test_size=test_size, random_state=random_state, stratify=[s["metadata"]["dataset"] for s in samples])
    #     val_samples.extend(train)
    #     val_samples.extend(test)
    # with open(output_val_file, 'w', encoding='utf-8') as f:
    #     json.dump(val_samples, f, indent=4, ensure_ascii=False)
    # Save the training and testing sets to separate JSON files
    # with open(output_train_file, 'w', encoding='utf-8') as f:
    #     json.dump(train_samples, f, indent=4, ensure_ascii=False)
    # with open(output_test_file, 'w', encoding='utf-8') as f:
    #     json.dump(test_samples, f, indent=4, ensure_ascii=False)

    # print(f"Training set saved to {output_train_file}.")
    # print(f"Testing set saved to {output_test_file}.")
    print(f"valing_few_shot_{n} set saved to {output_val_file}.")
if __name__ == "__main__":
    # Define the root directory and output file paths
    random.seed(42)  # 设置随机种子
    root_directory = "/home/DATA2/cxh/Pathology"
    blacklist_file = "/home/DATA2/cxh/Train_dataset/train_data/train.json"
    output_test_file = os.path.join(root_directory, "test_stage6.json")
    n=1
    output_val_file = os.path.join(root_directory, "train_quilt_v9.json")
    test_size = 0.2  
    process_all_samples_in_directory(root_directory, blacklist_file, output_test_file,output_val_file,n,test_size)




