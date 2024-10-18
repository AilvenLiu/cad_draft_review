import torch
import torch.nn as nn
import math
from .world_model import WorldModel
from .attention import MultiHeadAttention
from .feed_forward import FeedForward
from .positional_encoding import PositionalEncoding
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
        x = x + self.dropout(self.self_attn(_x, _x, _x, mask))
        
        # Cross-attention
        _x = self.norm2(x)
        x = x + self.dropout(self.cross_attn(_x, context, context, mask))
        
        # Feed-forward
        _x = self.norm3(x)
        x = x + self.dropout(self.feed_forward(_x))
        
        return x

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
        self.pos_encoder_image = PositionalEncoding(embed_dim, max_len=64)  # Adjust max_len based on image patches
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, vocab_size)  # Example classifier
        
        # Object Detection Head
        self.object_detection_head = ObjectDetectionHead(d_model, num_classes)

    def forward(self, text: torch.Tensor, images: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Encode text
        text_emb = self.text_embedding(text) * math.sqrt(self.text_embedding.embedding_dim)
        text_emb = self.pos_encoder_text(text_emb)
        
        # Encode images
        img_emb = self.image_encoder(images)  # [batch, embed_dim]
        img_emb = self.pos_encoder_image(img_emb.unsqueeze(1))  # [batch, 1, embed_dim]
        
        # Prepare context for cross-attention (combine image features)
        context = img_emb.repeat(1, text_emb.size(1), 1)  # [batch, text_len, embed_dim]
        
        # Pass through Transformer layers with cross-attention
        for layer in self.layers:
            text_emb = layer(text_emb, context, mask)
        
        # Example classification (e.g., next word prediction)
        output = self.classifier(text_emb)  # [batch, seq_len, vocab_size]
        
        # Object Detection
        bboxes, classes = self.object_detection_head(text_emb, context)  # [batch, num_queries, 4], [batch, num_queries, num_classes +1]
        
        return output, bboxes, classes
