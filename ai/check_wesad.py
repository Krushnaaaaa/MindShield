import pickle

file_path = "data/raw/WESAD/S2/S2_E4_Data.pkl"

with open(file_path, "rb") as file:
    data = pickle.load(file, encoding="latin1")

print(type(data))
print(data.keys())
print("MindShield test")