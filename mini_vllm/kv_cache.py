class KVCacheManager:
    def __init__(self):
        self.cache: dict[int, object] = {}
    
    def set(self,request_id, past_key_vals):
        self.cache[request_id] = past_key_vals
    
    def get(self,request_id):
        return self.cache.get(request_id)
    
    def has(self,request_id):
        return request_id in self.cache

    def free(self, request_id):
        self.cache.pop(request_id, None)