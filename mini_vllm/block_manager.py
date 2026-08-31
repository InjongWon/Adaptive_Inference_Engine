from collections import deque   
from mini_vllm.request import Request

class BlockManager:
    def __init__(self, num_blocks, block_size):
        if num_blocks<=0:
            raise ValueError("should be greater than 0")

        if block_size <=0:
            raise ValueError("should be greater than 0")
        
        self.num_blocks = num_blocks
        self.block_size = block_size 
        
        self.free_blocks = deque(range(num_blocks))
        self.request_blocks: dict[int,list[int]] = {}
    
    def _calculate_blocks(self,num_tokens)->int:
        if num_tokens <0:
            raise ValueError("has to be positive")
        if num_tokens ==0:
            return 0

        return (num_tokens +self.block_size -1) // self.block_size #(A+b-1)//b
    
    def append_slots(self,request:Request):
        if request.request_id not in self.request_blocks:
            raise ValueError(f"requst{request.request_id} has no allocated blocks")
        
        new_token_count = len(request.token_ids) + 1
        
        total_blocks = self._calculate_blocks(new_token_count)
        current_blocks = len(self.request_blocks[request.request_id])
        
        new_blocks_needed = total_blocks - current_blocks
        
        if new_blocks_needed <=0:
            return []
        
        if new_blocks_needed > len(self.free_blocks):
            raise RuntimeError("not enough blocks available")
        
        new_blocks = [
            self.free_blocks.popleft() for _ in range(new_blocks_needed)
        ]
        self.request_blocks[request.request_id].extend(new_blocks) # increase nested arrays
        
        return new_blocks
    
    def allocate(self,request:Request):
        """
        allocate enough blocks for new request coming in
        """
        if request.request_id in self.request_blocks:
            raise ValueError(f"this request{request.request_id} have already been served")
        need_blocks = self._calculate_blocks(len(request.token_ids))
        
        if need_blocks > len(self.free_blocks):
            raise RuntimeError("not enough blocks")
        
        blocks = [
            self.free_blocks.popleft() for _ in range(need_blocks)
        ]
        self.request_blocks[request.request_id] = blocks
        return blocks
    
    def free(self,request:Request):
        blocks = self.request_blocks.pop(request.request_id,None,)
        if blocks is None:
            return
        self.free_blocks.extend(blocks)
        
    def get_block_table(self,request:Request):
        if request.request_id not in self.request_blocks:
            raise ValueError(f"request {request.request_id} has no allocated blocks")
        return self.request_blocks[request.request_id]
    
    def translate_token_position(self,request:Request,token_position,):
        """similar to virtual memory paging """
        if token_position <0:
            raise ValueError("token positions cannot be neg")
        block_table = self.get_block_table(request)
        
        virtual_block = token_position // self.block_size
        block_offset = token_position % self.block_size 
        
        if virtual_block >= len(block_table):
            raise ValueError(f"token position {token_position} has no allocated block")

        physical_block = block_table[virtual_block]
        return physical_block, block_offset
    
    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - len(self.free_blocks)