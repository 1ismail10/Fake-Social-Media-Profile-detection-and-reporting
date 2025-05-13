import pickle

# Just change this to your actual file path
file_path = 'decision_fake.pkl'

with open(file_path, 'rb') as f:
    content = pickle.load(f)

print("✅ File Loaded!")
print("Type of content:", type(content))
print(content)
