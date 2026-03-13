import torch
from torch.utils.data import Dataset, DataLoader
import os

class ShakespeareCharDataset(Dataset):
    def __init__(self, text, block_size, train=True, val_fraction=0.05, test_fraction=0.05):
        """
        text: 完整的莎士比亚文本字符串
        block_size: 上下文窗口长度 (例如 64 或 128)
        train: 是否处于训练模式 (决定使用哪部分数据)
        """
        self.block_size = block_size
        
        # 1. 构建词汇表 (Character Level)
        # sorted 保证每次运行顺序一致，确保复现性
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = { chi:i for i,chi in enumerate(chars) }  # char to index
        self.itos = { i:chi for i,chi in enumerate(chars) }  # index to char  

        # 2. 数字化整个文本
        self.data_ids = [self.stoi[c] for c in tex]
        total_len = len(self.data_ids)

        # 3. 按顺序划分数据集 (关键！防止数据泄露)
        # 假设: Train 90%, Val 5%, Test 5%
        train_end = int(total_len * (1 - val_fraction - test_fraction))
        val_end = int(total_len * (1 - test_fraction))  

        if train:
            # 训练集：取前 90%
            self.data_ids = self.data_ids[:train_end]
            self.mode = "Train"
        else:
            # 这里为了简单，我们把 Val 和 Test 合并为 "Eval" 模式
            # 实际工程中通常分开实例化两个 Dataset
            # 这里我们取最后 10% 作为评估集
            self.data_ids = self.data_ids[train_end:]
            self.mode = "Eval"
            
        self.length = len(self.data_ids)
        print(f"[{self.mode}] 数据加载完成: 字符数 {self.length:,}, 词汇表大小 {self.vocab_size}")

    def __len__(self):
        # 滑动窗口能生成的样本数 = 总长度 - block_size
        # 如果数据太短连一个 block 都凑不够，返回 0
        return max(0, self.length - self.block_size)
    
    def __getitem__(self, idx):
        # 截取输入 x 和 目标 y (y 是 x 向后移一位)
        # x: [idx, idx+1, ..., idx+block_size-1]
        # y: [idx+1, idx+2, ..., idx+block_size]
        chunk = self.data_ids[idx : idx + self.block_size + 1]
        
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        return x, y

    def decode(self, ids):
        return ''.join([self.itos[i] for i in ids])

    def encode(self, text):
        return [self.stoi[c] for c in text]


# --- 使用示例 ---
if __name__ == "__main__":
    # 模拟读取数据 (实际请替换为你之前的下载代码)
    with open("data/shakespeare.txt", "r", encoding="utf-8") as f:
        full_text = f.read()
    
    BLOCK_SIZE = 64
    BATCH_SIZE = 32
    
    # 实例化训练集
    train_dataset = ShakespeareCharDataset(full_text, BLOCK_SIZE, train=True)
    
    # 实例化验证/测试集
    eval_dataset = ShakespeareCharDataset(full_text, BLOCK_SIZE, train=False)
    
    # 构建 DataLoader
    # num_workers=0 在 Windows 上最稳定，Linux/Mac 可以设为 CPU 核心数加速
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 验证一下
    print("\n--- 采样检查 ---")
    xb, yb = next(iter(train_loader))
    print(f"Input Shape: {xb.shape}, Target Shape: {yb.shape}")
    print(f"第一句输入 (解码): {train_dataset.decode(xb[0].tolist())}")
    print(f"第一句目标 (解码): {train_dataset.decode(yb[0].tolist())}")
    print("-" * 30)
    print("✅ 数据集准备就绪，可以开始训练模型了！")