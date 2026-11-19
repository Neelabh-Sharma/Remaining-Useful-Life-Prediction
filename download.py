from datasets import load_dataset

ds = load_dataset("penikmatrumput/nasa-cmapss-rul")
print(ds)

train = ds["train"].to_pandas()
test = ds["test"].to_pandas()

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print(train.head())

train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)
print("Saved train.csv and test.csv")