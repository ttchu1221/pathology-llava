import numpy as np
from sklearn.neighbors import NearestNeighbors
from torchvision import models, transforms
from PIL import Image
import torch
import os
import json
from collections import defaultdict
from sklearn.model_selection import train_test_split
import random
from multiprocessing import Pool, cpu_count
from pdb import set_trace
import time
from tqdm import tqdm
from transformers import ViTFeatureExtractor, ViTModel
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from datetime import datetime
import faiss
# 1. 特征提取函数（使用预训练模型提取图像特征）
# def find_knn(query_embedding, feature_db,target_caption, k=5):
#     """
#     Find the k-nearest neighbors of a query image based on cosine similarity.
#     Args:
#         query_embedding: The embedding of the query image.
#         feature_db: A dictionary mapping image paths to their embeddings.
#         k: Number of nearest neighbors to find.
#     Returns:
#         top_k_paths: List of paths to the top-k most similar images.
#     """
#     similarities = []
#     for unique_id, img_data in feature_db.items():
#         if unique_id.startswith(target_caption):  # 检查是否以目标 caption 开头
#             for img_path, embedding in img_data.items():
#                 sim = cosine_similarity(query_embedding, np.array(embedding).reshape(1, -1))[0][0]
#                 similarities.append((img_path, sim))
#             break
    
#     # Sort by similarity in descending order
#     similarities.sort(key=lambda x: x[1], reverse=True)
    
#     # Extract the top-k image paths
#     top_k_paths = [path for path, _ in similarities[:k]]
#     return top_k_paths
def extract_features_batch(image_paths, feature_extractor, model):
    """
    批量提取图像特征。
    :param image_paths: 图像路径列表
    :param feature_extractor: ViT 的特征提取器
    :param model: 预训练的 ViT 模型
    :param batch_size: 每次处理的图像数量
    :return: 图像特征矩阵 (n_samples, feature_dim)
    """
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

    images = Image.open(os.path.join("Eval_data",image_paths)).convert('RGB') 
    inputs = feature_extractor(images=images, return_tensors="pt")  # 转换为模型输入格式
    
    # 将输入数据移动到 GPU
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)  # 提取特征
    
    # 取 [CLS] token 的特征作为图像表示
    features = outputs.last_hidden_state[:, 0, :]
    
    # 将特征移回 CPU 并转换为 NumPy 数组
    features = features.cpu().numpy()

    # 合并所有批次的特征
    return features

# 2. 构建图像数据库并执行KNN搜索
def build_feature_database(samples, feature_extractor, model,blacklist,batch_size=3072):
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    feature_db = defaultdict(dict)
    image_paths = [sample['relative_image']  for sample in samples if sample["relative_image"]not in blacklist]
    # images = [sample['image'] for sample in samples]
    captions = [sample['caption'] for sample in samples if sample["relative_image"] not in blacklist]
    # Process images in batches
    for i in range(0, len(image_paths), batch_size):
        # batch_images = images[i:i + batch_size]
        batch_paths = image_paths[i:i + batch_size]
        caption = captions[i:i + batch_size]
        # Preprocess images
        def load_image(img_path):
            full_path = os.path.join("Eval_data", img_path)
            return Image.open(full_path).convert('RGB')

        def load_images_parallel(batch_paths, num_workers=8):
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                images = list(executor.map(load_image, batch_paths))
            return images

        # 在批次处理时调用
        images = load_images_parallel(batch_paths)
        inputs = feature_extractor(images, return_tensors="pt").to(device)

        # Extract features using the model
        with torch.no_grad():
            outputs = model(**inputs)

        # Average pooling to get fixed-length embeddings
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

        # Store embeddings in the feature database
        for path, embedding,cap in zip(batch_paths, embeddings,caption):
            feature_db[cap][path] = embedding

    return feature_db

def build_faiss_index(feature_db, target_caption):
    embeddings = []
    img_paths = []

    # 筛选与目标 caption 匹配的嵌入向量和图像路径
    for unique_id, img_data in feature_db.items():
        if unique_id.startswith(target_caption):  # 检查是否以目标 caption 开头
            for img_path, embedding in img_data.items():
                embeddings.append(embedding)
                img_paths.append(img_path)
            break
    # 将嵌入向量转换为 NumPy 数组并标准化（FAISS 需要 float32 类型）
    embeddings = np.array(embeddings).astype('float32')

    # 归一化向量（对于余弦相似度计算）
    faiss.normalize_L2(embeddings)

    # 构建 FAISS 索引（使用内积等价于余弦相似度）
    dimension = embeddings.shape[1]
    index_flat = faiss.IndexFlatIP(dimension)  # 使用内积
    index_flat.add(embeddings)

    # 创建 GPU 资源并将索引移动到 GPU
    res = faiss.StandardGpuResources()  # 初始化 GPU 资源
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)  # 将索引移动到第一个 GPU

    return gpu_index, img_paths


def find_knn_faiss(query_embedding, feature_db, target_caption, k=5):
    # 构建 FAISS 索引
    gpu_index, img_paths = build_faiss_index(feature_db, target_caption)

    # 将查询向量归一化
    query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
    faiss.normalize_L2(query_embedding)

    # 在 GPU 索引中搜索
    distances, indices = gpu_index.search(query_embedding, k)

    # 提取对应的图像路径
    top_k_paths = [img_paths[i] for i in indices[0]]

    return top_k_paths




def select_n_samples_from_same_dataset(model,feature_extractor,feature_db,samples,nums,n_per_batch, blacklist,pred_img):
    """
    Select n samples from the same dataset ensuring they have unique captions.
    The last sample can have a caption that duplicates an earlier one within the batch but not in other batches.
    """
    
    
    # Build the feature database once
    #samples.sort(key=lambda x: x['caption'])
    query_image_path = pred_img['relative_image']
    query_embedding = extract_features_batch(query_image_path,feature_extractor,model)

    selected_samples = []
    category_samples = defaultdict(list)

    # Group samples by their captions
    for sample in samples:
        if sample['relative_image'] not in blacklist:
            category_samples[sample['caption']].append(sample)

    # Precompute the query embedding for pred_img
    # query_image_path = pred_img['relative_image']
    # query_embedding = feature_db.get(query_image_path)  # Get the embedding from the feature database
    if query_embedding is None:
        raise ValueError(f"Query image path {query_image_path} not found in the feature database.")

    def process_category(category, samples_in_cat):
            """
            Process a single category to select samples.
            """
            caption_groups = defaultdict(list)
            for sample in samples_in_cat:
                caption_groups[sample['caption']].append(sample)
            
            found_pair = False
            for caption, group in caption_groups.items():
                if len(group) >= nums:
                        # Use KNN to find the most similar images
                        similar_image_paths = find_knn_faiss(query_embedding, feature_db, caption, k=nums)
                        
                        # Filter out blacklisted images and add to selected samples
                        similar_samples = [
                            sample for sample in samples 
                            if sample['relative_image'] in similar_image_paths and sample['relative_image'] not in blacklist
                        ]
                        return similar_samples[:nums]
                    
                    # Randomly select samples if no special condition is met
                    # return random.sample(group, nums)
            
            return None  # Indicate failure for this category

        # Use ThreadPoolExecutor to process categories in parallel
    with ThreadPoolExecutor() as executor:
        futures = []
        for category, samples_in_cat in category_samples.items():
            futures.append(executor.submit(process_category, category, samples_in_cat))
        
        for future in as_completed(futures):
            result = future.result()
            if result:
                selected_samples.extend(result)
            
            if len(selected_samples) >= n_per_batch:
                break
    
    # Check if we have successfully picked samples from n categories
    successful_picks = [s for s in selected_samples if s is not None]
    successful_picks.sort(key=lambda x: x['caption'])
    return successful_picks if len(successful_picks) == n_per_batch else None
def generate_human_value(captions,n):
    """Generate a string with image captions for human interaction."""
    #

    caption_strings = [
        f"<image>Caption#{i+1}:{caption}" for i, caption in enumerate(captions[:-1])
    ]
    if captions:
        caption_strings.append(f"Based on the Caption and characteristics provided by the first {len(captions)-1} images, please try to determine which Caption this new image most closely resembles?\n<image>Caption#{len(captions)}:\nYou must choose your answer from the Choice List.\n Choice List:")
    caption_strings += [f"{chr(65 + i)}:{caption}" for i, caption in enumerate(captions[:-1:n])]
    # caption_strings += [f"<image>Caption#{len(captions)}:"]
    return "\n".join(caption_strings)
    #return ' '.join(caption_strings)

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
            gpt_sample = chr(65 + i)
    
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

def process_sample(dataset_samples, n,s_blacklist):
    last__samples = []
    caption_samples = defaultdict(list)

    # 按类别分组
    for sample in dataset_samples:
        caption_samples[sample['caption']].append(sample)

    # 处理每个类别的样本
    for caption, samples in caption_samples.items():
        #process_sample = any(sample['relative_image'] in s_blacklist for sample in samples)
        cutoff_index = max(1, int(len(samples) * 0.9))

        if n>4:
            # s_samples = [sample for sample in samples if sample['relative_image'] not in s_blacklist]
            # s_samples.sort(key=lambda x: x['relative_image'])  # 确保顺序一致
            last__samples.extend(samples[cutoff_index:])
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

    return last__samples

def mutiprocess_samples(samples_dict, n,blacklist,s_blacklist):
    last__samples = []

    # 使用线程池并行处理
    with ThreadPoolExecutor() as executor:
        futures = []

        # 提交任务到线程池
        for dataset_samples in samples_dict.values():
            future = executor.submit(process_sample, dataset_samples,n, s_blacklist)
            futures.append(future)

        # 收集结果
        for future in as_completed(futures):
            partial_last__samples = future.result()
            last__samples.extend(partial_last__samples)

    # 去重并更新黑名单
    last_samples1 = [dict(s) for s in set(frozenset(d.items()) for d in last__samples)]
    for sample in last_samples1:
        blacklist.add(sample['relative_image'])

    return last__samples, blacklist, last_samples1
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
def combine_samples_from_same_dataset(samples_dict, n,caption_kinds, sample_counter, blacklist,s_blacklist):
    combined_samples = []

    # key_dic = []
    n_per_batch = n*len(caption_kinds)+1
    last__samples, blacklist, last_samples1 = mutiprocess_samples(samples_dict, n,blacklist,s_blacklist)
    # last__samples = []
    # for dataset_samples in samples_dict.values():
    #     caption_samples = defaultdict(list)
    #     for sample in dataset_samples:
    #         caption_samples[sample['caption']].append(sample)
        
    #     for caption, samples in caption_samples.items():

    #         process_sample = any(sample['relative_image'] in s_blacklist for sample in samples)
    #         cutoff_index = max(1, int(len(samples) * 0.8))
    #         if process_sample:
    #             s_samples = [sample for sample in samples if sample['relative_image'] not in s_blacklist]
    #             s_samples.sort(key=lambda x: x['relative_image'])  # Sort to ensure consistent order
    #             last__samples.extend(s_samples[cutoff_index:])
    #         else:
    #             cutoff_index = max(1, int(len(samples) * 0.8))
    #             last__samples.extend(samples[cutoff_index:])
    #         for last_samples in last__samples:
    #             blacklist.add(last_samples['relative_image'])
    #         last_samples1 = [dict(s) for s in set(frozenset(d.items()) for d in last__samples)]
            #caption_samples[caption] = s_samples[:cutoff_index]
    # 
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    model_name = "google/vit-base-patch16-224-in21k"
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name).to(device)
    model.eval()

    for dataset_samples in samples_dict.values():
        s = time.time()
        readable_time = datetime.fromtimestamp(s).strftime("%Y-%m-%d %H:%M:%S")
        print("Readable time:", readable_time)
        feature_db = build_feature_database(dataset_samples, feature_extractor, model,blacklist)
        s1 = time.time()
        readable_time = datetime.fromtimestamp(s1).strftime("%Y-%m-%d %H:%M:%S")
        print("Readable time:", readable_time)
        batch_size = 1000
        #batch = random.sample(last_samples1, min(batch_size, len(last_samples1)))
        while last_samples1:
            batch = random.sample(last_samples1, min(batch_size, len(last_samples1)))
            for last_sample in batch:
                # 从同一数据集中选择 n_per_batch-1 个样本
                selected_samples = select_n_samples_from_same_dataset(
                    model, feature_extractor, feature_db, dataset_samples, n, n_per_batch - 1, blacklist, pred_img=last_sample
                )
                
                # 检查是否成功选择了足够的样本
                if not selected_samples or len(selected_samples) < n_per_batch - 1:
                    continue  # 跳过当前样本，继续处理下一个
                
                # 将当前样本加入到 selected_samples 中
                selected_samples.append(last_sample)
                
                # 创建组合样本
                combined_sample = create_combined_sample(selected_samples, sample_id=sample_counter + 1, n=n)
                
                # 如果组合样本创建成功，添加到结果列表中
                if combined_sample:
                    combined_samples.append(combined_sample)
                    sample_counter += 1
                #set_trace()
            # 批量移除已处理的样本
            last_samples1 = [x for x in last_samples1 if x not in batch]
                
                # Add the last sample to the blacklist to ensure it doesn't appear in the front of other batches
                # last_sample = selected_samples[-1]
                # blacklist.add(last_sample['relative_image'])
            
            # for sample in selected_samples[:-1]:
            #     if sample in dataset_samples:
            #         if sample['caption'] not in key_dic:
            #             dataset_samples.remove(sample)
   
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
    
    # Process files in parallel using multiprocessing.Pool
    with Pool(cpu_count()) as pool:
        processed_results = pool.starmap(process_file, [(file_path, root_directory) for file_path in file_paths])

    # Aggregate results from all processes by dataset
    for result in processed_results:
        all_samples_by_dataset[os.path.dirname(result[0]['directory'])].extend(result)

    # Create combined samples ensuring they come from the same dataset
    sample_counter = 0
    all_combined_samples = []
    

    for dataset_samples in tqdm(all_samples_by_dataset.values()):
        
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
    np.random.seed(42)  # 设置随机种子
    root_directory = "/home/DATA2/cxh/Eval_data"
    blacklist_file = "/home/DATA2/cxh/Eval_data/val_4_shot_1.json"
    #blacklist_file = "/home/DATA2/cxh/Train_dataset/train_data/train.json"
    output_test_file = os.path.join(root_directory, "test_stage6.jsson")
    n=1
    output_val_file = os.path.join(root_directory, "val_1_shot_knn.json")
    test_size = 0.2  
    process_all_samples_in_directory(root_directory, blacklist_file, output_test_file,output_val_file,n,test_size)

