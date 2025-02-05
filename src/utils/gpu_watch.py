import torch

def get_gpu_memory_usage_message():
    """
    Returns a formatted message with the current and total GPU memory usage for the active CUDA device.
    
    Returns:
        str: A message showing current and total GPU memory usage or an error if CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return "CUDA is not available on this system."
    
    device = torch.cuda.current_device()
    current_usage = torch.cuda.memory_allocated(device) / (1024 * 1024)  # Convert bytes to MB
    total_memory = torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)  # Convert bytes to MB
    
    return f" GPU  {current_usage:.2f} : {total_memory:.2f}MB\n"


# Example usage
if __name__ == "__main__":
    print(get_gpu_memory_usage_message())
