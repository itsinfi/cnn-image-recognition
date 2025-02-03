import tensorflow as tf
gpu_acc_is_enabled = len(tf.config.experimental.list_physical_devices('GPU')) != 0
print('Is GPU acceleration enabled?', 'Yes.' if gpu_acc_is_enabled else 'No.')