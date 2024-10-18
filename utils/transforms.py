import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)
        
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))
        
        if mask is not None:
            scaled_attention += (mask * -1e9)
        
        attention_weights = torch.softmax(scaled_attention, dim=-1)
        output = torch.matmul(attention_weights, v)
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, -1, self.d_model)
        output = self.dense(output)
        
        return output, attention_weights
    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention
        _x = self.norm1(x)
        x = x + self.dropout(self.self_attn(_x, _x, _x, mask)[0])
        
        # Cross-attention
        _x = self.norm2(x)
        x = x + self.dropout(self.cross_attn(_x, context, context, mask)[0])
        
        # Feed-forward
        _x = self.norm3(x)
        x = x + self.dropout(self.feed_forward(_x))
        
        return x
    
class ObjectDetectionHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int, num_queries: int = 100):
        super().__init__()
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.transformer = nn.Transformer(d_model=d_model, num_encoder_layers=6, num_decoder_layers=6)
        self.bbox_embed = nn.Linear(d_model, 4)  # Bounding box coordinates
        self.class_embed = nn.Linear(d_model, num_classes + 1)  # +1 for no object

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        # memory: [batch, seq_len, d_model]
        query = self.query_embed.weight.unsqueeze(1).repeat(1, x.size(0), 1)  # [num_queries, batch, d_model]
        tgt = torch.zeros_like(query)
        output = self.transformer(tgt, memory.transpose(0, 1))  # [num_queries, batch, d_model]
        output = output.transpose(0, 1)  # [batch, num_queries, d_model]

        # Bounding box predictions
        bboxes = self.bbox_embed(output)  # [batch, num_queries, 4]

        # Class predictions
        classes = self.class_embed(output)  # [batch, num_queries, num_classes + 1]
        
        return bboxes, classes

class MultimodalTransformer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_length: int,
        dropout: float = 0.1,
        img_channels: int = 3,
        embed_dim: int = 512,
        num_classes: int = 600
    ):
        super().__init__()
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder_text = PositionalEncoding(d_model, max_len=max_seq_length)
        
        self.image_encoder = WorldModel(embed_dim=embed_dim, img_channels=img_channels)
        self.pos_encoder_image = PositionalEncoding(embed_dim, max_len=64)
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, vocab_size)
        
        self.object_detection_head = ObjectDetectionHead(d_model, num_classes)

    def forward(self, text: torch.Tensor, images: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        text_emb = self.text_embedding(text) * math.sqrt(self.text_embedding.embedding_dim)
        text_emb = self.pos_encoder_text(text_emb)
        
        img_emb = self.image_encoder(images)
        img_emb = self.pos_encoder_image(img_emb.unsqueeze(1))
        
        context = img_emb.repeat(1, text_emb.size(1), 1)
        
        for layer in self.layers:
            text_emb = layer(text_emb, context, mask)
        
        output = self.classifier(text_emb)
        
        bboxes, classes = self.object_detection_head(text_emb, context)
        
        return output, bboxes, classes
