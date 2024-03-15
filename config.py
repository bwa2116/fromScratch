patch_size = 4
hidden_size = 48
num_hidden_layers = 4
num_attention_heads = 4
hidden_dropout_prob = 0.0
attention_probs_dropout_prob = 0.0
initializer_range = 0.02

config_CIFAR10 = {
    "patch_size": patch_size4,
    "hidden_size": hidden_size,
    "num_hidden_layers": num_hidden_layers,
    "num_attention_heads": num_attention_heads,
    "intermediate_size": 4 * hidden_size,
    "hidden_dropout_prob": hidden_dropout_prob,
    "attention_probs_dropout_prob": attention_probs_dropout_prob,
    "initializer_range": initializer_range,
    "image_size": 32, # num_classes of CIFAR10
    "num_classes": 10,
    "num_channels": 3,
    "qkv_bias": True,
}
