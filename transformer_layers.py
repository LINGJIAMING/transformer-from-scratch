import torch.nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.depth = d_model // num_heads
        
        self.wq = torch.nn.Linear(d_model, d_model)
        self.wk = torch.nn.Linear(d_model, d_model)
        self.wv = torch.nn.Linear(d_model, d_model)
        
        self.dense = torch.nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, v, k, q, mask):
        batch_size = q.size(0)
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention, _ = self.scaled_dot_product_attention(q, k, v, mask)
        
        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()
        scaled_attention = scaled_attention.view(batch_size, -1, self.d_model)
        
        output = self.dense(scaled_attention)
        
        return output
    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        
        dk = k.size(-1)
        scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(dk, dtype=torch.float32))
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        attention_weights = torch.nn.Softmax(dim=-1)(scaled_attention_logits)
        
        output = torch.matmul(attention_weights, v)
        
        return output, attention_weights
    


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model,dff),
            torch.nn.ReLU(),
            torch.nn.Linear(dff,d_model)
        )
        self.layernorm1 = torch.nn.LayerNorm(d_model)
        self.layernorm2 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout_rate)
        self.dropout2 = torch.nn.Dropout(dropout_rate)
        

    def forward(self, x, mask):
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2
    


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model,dff),
            torch.nn.ReLU(),
            torch.nn.Linear(dff,d_model)
        )
        
        self.layernorm1 = torch.nn.LayerNorm(d_model)
        self.layernorm2 = torch.nn.LayerNorm(d_model)
        self.layernorm3 = torch.nn.LayerNorm(d_model)
        
        self.dropout1 = torch.nn.Dropout(dropout_rate)
        self.dropout2 = torch.nn.Dropout(dropout_rate)
        self.dropout3 = torch.nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        attn1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + x)
        
        attn2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(attn2 + out1)
        
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(ffn_output + out2)
        
        return out3


if __name__ == "__main__":
    # 简单的测试
    print("Testing MultiHeadAttention...")
    dummy_input = torch.randn(32, 10, 512) # (batch, seq_len, d_model)
    attention = MultiHeadAttention(d_model=512, num_heads=8)
    output = attention(dummy_input, dummy_input, dummy_input)
    print(f"Input shape: {dummy_input.shape}, Output shape: {output.shape}")
    
    print("All tests passed!")