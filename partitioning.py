def get_all_partitions(k: list[int], K: list[int]):
    # Pair with original indices and sort descending
    nums = sorted([(val, idx) for idx, val in enumerate(k)], key=lambda x: x[0], reverse=True)
    targets = sorted(enumerate(K), key=lambda x: x[1], reverse=True)
    
    used = [False] * len(nums)

    def backtrack(t_idx, current_partition):
        # Optimization 1 is still mathematically sound: 
        # The final target simply takes whatever indices are left over.
        if t_idx == len(targets) - 1:
            orig_t_idx, _ = targets[t_idx]
            remaining = [nums[i][1] for i in range(len(nums)) if not used[i]]
            
            # Reconstruct the output to match K's original order
            final_partition = current_partition + [(orig_t_idx, remaining)]
            final_partition.sort(key=lambda x: x[0])
            yield [group for _, group in final_partition]
            return
        
        orig_t_idx, target_val = targets[t_idx]
        
        def find_subsets(start_i, left, current_group):
            if left == 0:
                # Target satisfied, move to the next one
                yield from backtrack(t_idx + 1, current_partition + [(orig_t_idx, current_group)])
                return
            
            for i in range(start_i, len(nums)):
                if used[i]:
                    continue
                
                val, orig_k_idx = nums[i]
                
                if val > left:
                    continue  # Skip this number for THIS target, but keep searching smaller numbers
                
                used[i] = True
                yield from find_subsets(i + 1, left - val, current_group + [orig_k_idx])
                used[i] = False  # Backtrack
                
        yield from find_subsets(0, target_val, [])

    yield from backtrack(0, [])