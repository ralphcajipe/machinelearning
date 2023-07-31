import tensorflow as tf

# Check if CUDA is available
print(f"Is CUDA supported by this system? {tf.test.is_gpu_available()}")

# Get the CUDA version
print(f"CUDA version: {tf.version.VERSION}")

# Get the ID of the current CUDA device
cuda_id = tf.config.experimental.list_physical_devices('GPU')[0]
print(f"ID of current CUDA device: {cuda_id}")

# Get the name of the current CUDA device
print(f"Name of current CUDA device: {tf.config.experimental.get_device_details(cuda_id)['name']}")

