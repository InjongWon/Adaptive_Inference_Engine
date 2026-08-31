from dataclasses import dataclass

import torch
import heapq


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    
class Sampler:
    def sample(
        self,
        logits: torch.Tensor,
        sampling:SamplingParams,
    ) -> torch.Tensor:

        if logits.ndim != 2:
            raise ValueError("passing 1D")
        #greedy
        if sampling.temperature == 0: 
            return logits.argmax(dim=-1)
        if sampling.temperature <0:
            raise ValueError("cannot be negative")

        #temp > 0: logits/temperature
        scale = logits / sampling.temperature 
        k = sampling.top_k
        if sampling.top_k is not None:

            batch, tokens = logits.shape
            if sampling.top_k <= 0:
                raise ValueError("top_k must be greater than 0")

            if sampling.top_k > tokens:
                raise ValueError("cannot choose k_highest exceeding vocab size")

            for b in range(batch): # each requests in one batch 
                min_heap = []   
                
                for idx in range(tokens):
                    
                    val = scale[b, idx].item() #going through each row and col
                    if len(min_heap) < sampling.top_k:
                        heapq.heappush(min_heap, (val,idx))
                    elif val > min_heap[0][0]:
                        heapq.heapreplace(min_heap,(val,idx))
                
                
                keep_indices = {idx for _, idx in min_heap}
    
                # # mask every token not selected by top-k
                for idx in range(tokens):
                    if idx not in keep_indices:
                        scale[b,idx] = float('-inf')
                        
        probs = torch.softmax(scale, dim=-1)
        if sampling.top_p is not None:
            if not 0.0<sampling.top_p <=1.0:
                raise ValueError("has to within the range")
            
            sorted_probs, sorted_index = torch.sort(
                probs,
                dim =-1,
                descending = True
            )
            cumulative_probs = torch.cumsum(
                sorted_probs,
                dim=-1,
            )
            
            
            for b in range(probs.shape[0]):
                cumulative = 0
                for i in range(probs.shape[1]):
                    cumulative += sorted_probs[b,i].item()
                    if cumulative >= sampling.top_p:
                        for j in range(i+1, probs.shape[1]):
                            original_idx = sorted_index[b,j].item()
                            probs[b,original_idx] = 0.0
                        break
            
            #normalize 
            probs = probs / probs.sum(dim=-1, keepdim=True)
        
        #actually decode one token for each request
        sampled = torch.multinomial(
            probs,
            num_samples=1,
        )

        return sampled.squeeze(-1)
    
        
                        
                    
                
            
             
