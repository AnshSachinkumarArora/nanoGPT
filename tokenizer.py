import tiktoken
import numpy as np

#import dataset
with open('./dataset/input.txt', 'r', encoding='utf-8') as file:
    text = file.read()
file_path = './dataset/dataset.bin'

#encode dataset
encoder = tiktoken.get_encoding('cl100k_base')
encoded_dataset = encoder.encode_ordinary(text)
encoded_dataset = np.array(encoded_dataset, dtype=np.uint32)

#export dataset
encoded_dataset.tofile(file_path)
print(f'Total tokens: {len(encoded_dataset)}')
print(f'Dataset saved to {file_path}')