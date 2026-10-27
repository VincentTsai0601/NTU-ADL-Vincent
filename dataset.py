from datasets import load_dataset

ds = load_dataset("theblackcat102/ADL_Final_25W_part1_with_cost")

# Save all dataset splits into this local folder
ds.save_to_disk("./data/ADL_Final_25W_part1_with_cost")

print(ds)