import torch
import torch.nn as nn
import math

# ==========================================
# 1. 配置类 (Config)
# ==========================================
class Config:
    def __init__(self):
        # 模型维度
        self.d_model = 512
        self.num_heads = 8
        self.dff = 2048  # Feed Forward 维度
        
        # 层数
        self.num_encoder_layers = 6
        self.num_decoder_layers = 6
        
        # 数据相关
        self.vocab_size = 65  # 假设莎士比亚字符级词汇表大小 (需根据实际数据处理调整)
        self.max_seq_len = 100
        self.dropout_rate = 0.1
        
        # 训练相关
        self.learning_rate = 0.0001
        self.batch_size = 64
        
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def print_config(self):
        print(f"--- Model Configuration ---")
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")
        print("---------------------------")

# ==========================================
# 2. 位置编码 (Positional Encoding)
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# ==========================================
# 3. 引入你之前的核心模块
# ==========================================
# (为了代码完整性，这里简略引用，实际使用时请确保 MultiHeadAttention, EncoderBlock, DecoderBlock 已定义)
# 假设它们在当前文件或从 transformer_layers.py 导入
# from transformer_layers import MultiHeadAttention, EncoderBlock, DecoderBlock 

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, v, k, q, mask):
        batch_size = q.size(0)
        q = self.wq(q); k = self.wk(k); v = self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention, _ = self.scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()
        scaled_attention = scaled_attention.view(batch_size, -1, self.d_model)
        return self.dense(scaled_attention)

    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        dk = k.size(-1)
        scaled_logits = matmul_qk / torch.sqrt(torch.tensor(dk, dtype=torch.float32))
        if mask is not None:
            scaled_logits += (mask * -1e9)
        attention_weights = nn.Softmax(dim=-1)(scaled_logits)
        output = torch.matmul(attention_weights, v)
        return output, attention_weights

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(nn.Linear(d_model, dff), nn.ReLU(), nn.Linear(dff, d_model))
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        attn_output = self.mha(x, x, x, mask)
        out1 = self.layernorm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + self.dropout2(ffn_output))
        return out2

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(nn.Linear(d_model, dff), nn.ReLU(), nn.Linear(dff, d_model))
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        attn1 = self.mha1(x, x, x, look_ahead_mask)
        out1 = self.layernorm1(x + self.dropout1(attn1))
        
        attn2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        out2 = self.layernorm2(out1 + self.dropout2(attn2))
        
        ffn_output = self.ffn(out2)
        out3 = self.layernorm3(out2 + self.dropout3(ffn_output))
        return out3

# ==========================================
# 4. 完整的 ShakespeareTransformer
# ==========================================
class ShakespeareTransformer(nn.Module):
    def __init__(self, config):
        super(ShakespeareTransformer, self).__init__()
        self.config = config
        
        # 输入嵌入
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_len, config.dropout_rate)
        
        # Encoder 堆叠
        self.encoders = nn.ModuleList([
            EncoderBlock(config.d_model, config.num_heads, config.dff, config.dropout_rate)
            for _ in range(config.num_encoder_layers)
        ])
        
        # Decoder 堆叠
        self.decoders = nn.ModuleList([
            DecoderBlock(config.d_model, config.num_heads, config.dff, config.dropout_rate)
            for _ in range(config.num_decoder_layers)
        ])
        
        # 输出层
        self.final_layer = nn.Linear(config.d_model, config.vocab_size)
        
        # 初始化权重 (可选，有助于收敛)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_padding_mask(self, seq):
        # (batch, seq_len) -> (batch, 1, 1, seq_len)
        # 假设 0 是 padding token
        return (seq == 0).unsqueeze(1).unsqueeze(2).float()

    def create_look_ahead_mask(self, size):
        # (seq_len, seq_len)
        mask = 1 - torch.triu(torch.ones(size, size), diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0) # (1, 1, seq_len, seq_len)

    def forward(self, enc_input, dec_input):
        # 1. 创建 Mask
        enc_padding_mask = self.create_padding_mask(enc_input)
        dec_padding_mask = self.create_padding_mask(enc_input) # 用于 Encoder-Decoder 注意力
        look_ahead_mask = self.create_look_ahead_mask(dec_input.size(1)).to(enc_input.device)
        combined_mask = torch.max(dec_padding_mask, look_ahead_mask) # Decoder 需要两种 mask

        # 2. Encoder 输入处理
        x = self.embedding(enc_input) * math.sqrt(self.config.d_model)
        x = self.pos_encoding(x)
        enc_output = x
        
        for encoder_layer in self.encoders:
            enc_output = encoder_layer(enc_output, enc_padding_mask)
            
        # 3. Decoder 输入处理
        x = self.embedding(dec_input) * math.sqrt(self.config.d_model)
        x = self.pos_encoding(x)
        dec_output = x
        
        for decoder_layer in self.decoders:
            dec_output = decoder_layer(dec_output, enc_output, look_ahead_mask, dec_padding_mask)
            
        # 4. 最终输出
        final_output = self.final_layer(dec_output) # (batch, seq_len, vocab_size)
        
        return final_output

# ==========================================
# 5. 测试运行
# ==========================================
if __name__ == "__main__":
    # 初始化配置
    config = Config()
    config.print_config()
    
    # 实例化模型
    model = ShakespeareTransformer(config)
    model.to(config.device)
    
    print(f"\nModel Device: {config.device}")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 构造假数据 (Batch=4, SeqLen=20)
    # 假设 0 是 PAD, 其他是有效 token
    enc_input = torch.randint(1, config.vocab_size, (4, 20)).to(config.device)
    dec_input = torch.randint(1, config.vocab_size, (4, 15)).to(config.device)
    
    # 前向传播
    with torch.no_grad():
        output = model(enc_input, dec_input)
    
    print(f"\nInput Enc Shape: {enc_input.shape}")
    print(f"Input Dec Shape: {dec_input.shape}")
    print(f"Output Shape: {output.shape}") 
    # 期望输出: (batch, dec_seq_len, vocab_size) -> (4, 15, 65)
    
    if output.shape == (4, 15, config.vocab_size):
        print("\n✅ Success! ShakespeareTransformer is working correctly.")
    else:
        print("\n❌ Error: Output shape mismatch.")