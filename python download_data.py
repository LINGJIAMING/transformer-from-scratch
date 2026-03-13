import os
import requests
from datasets import Dataset

# 1. 设置镜像 (可选，因为 raw.githubusercontent.com 通常国内也能访问，如果慢也可以找镜像)
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" 

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
local_file_path = "data/shakespeare.txt"

# 创建 data 目录
os.makedirs("data", exist_ok=True)

print("正在下载莎士比亚原始数据...")
if not os.path.exists(local_file_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(local_file_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"✅ 下载成功，保存至: {local_file_path}")
    else:
        raise Exception("下载失败，请检查网络")
else:
    print("✅ 文件已存在，跳过下载")

# 2. 读取本地文本
with open(local_file_path, "r", encoding="utf-8") as f:
    text_data = f.read()

# 3. 转换为 HuggingFace Dataset 格式
# 我们手动构建一个字典列表，这是最通用的格式
data_dict = {"text": [text_data]} 
# 注意：这里我们把整本书作为一个样本。
# 如果你想按行分割，可以用 text_data.split('\n')，但那样会切断句子上下文。
# 对于字符级 LM，通常把整个长文本作为一个样本，然后在 DataLoader 里做滑动窗口切割。

dataset = Dataset.from_dict(data_dict)

print("-" * 30)
print(f"✅ 数据集构建成功！")
print(f"样本数: {len(dataset)}")
print(f"总字符数: {len(dataset[0]['text']):,}")
print(f"前 100 字符预览:\n{dataset[0]['text'][:100]}")