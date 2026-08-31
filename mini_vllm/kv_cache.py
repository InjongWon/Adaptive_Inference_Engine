import torch

class KVCacheManager:
    def __init__(self,num_layers,num_blocks,block_size, num_kv_heads,head_dim,dtype,device:str="cpu"):
        
        if any(v <= 0 for v in (num_layers, num_blocks, block_size, num_kv_heads, head_dim)):
            raise ValueError(f"must be greater than 0")
        
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size =block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        
        self.key_cache = torch.empty(
            (
                num_layers,
                num_blocks,
                block_size,
                num_kv_heads,
                head_dim,
            ),
            dtype=dtype,
            device=device,
        )

        self.value_cache = torch.empty(
            (
                num_layers,
                num_blocks,
                block_size,
                num_kv_heads,
                head_dim,
            ),
            dtype=dtype,
            device=device,
        )