import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PerformerAttention(nn.Module):
    def __init__(self, input_dim, feature_dim=16):
        super(PerformerAttention, self).__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.random_features = nn.Parameter(torch.randn(feature_dim, 1))

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        random_features = self.random_features.unsqueeze(0)  # add batch dimension
        attention_scores = torch.einsum('bqd,fd->bqf', [x, random_features])
        attention_scores = attention_scores / (self.feature_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=1)
        output = torch.einsum('bqf,bqv->bvf', [attention_weights, x])
        return output

class PatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):        
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Embeddings(nn.Module):        
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embeddings = PatchEmbeddings(config)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))
        self.position_embeddings = nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches + 1, config["hidden_size"]))
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size, _, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = PerformerAttention(config["hidden_size"])
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = nn.Sequential(
            nn.Linear(config["hidden_size"], config["intermediate_size"]),
            nn.GELU(),
            nn.Linear(config["intermediate_size"], config["hidden_size"]),
            nn.Dropout(config["hidden_dropout_prob"])
        )
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])

    def forward(self, x):
        attention_output = self.attention(self.layernorm_1(x))
        x = x + attention_output
        mlp_output = self.mlp(self.layernorm_2(x))
        x = x + mlp_output
        return x

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(config["num_hidden_layers"]):
            block = Block(config)
            self.blocks.append(block)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class ViTForClassfication(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config["image_size"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]
        self.embedding = Embeddings(config)
        self.encoder = Encoder(config)
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        self.apply(self._init_weights)

    def forward(self, x):
        embedding_output = self.embedding(x)
        encoder_output = self.encoder(embedding_output)
        logits = self.classifier(encoder_output[:, 0, :])
        return logits
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.cls_token.dtype)
