from metatrain.utils.data import DiskDataset, collate_fn
import torch
import sys
import json
import tqdm


dataset_name = sys.argv[1]

disk_dataset = DiskDataset(dataset_name)
print(len(disk_dataset))
dataloader = torch.utils.data.DataLoader(disk_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)


sample_batch = next(iter(dataloader))
targets = sample_batch[1]
sum_of_squares = {key: 0.0 for key in targets.keys()}
num_elements = {key: 0 for key in targets.keys()}

for batch in tqdm.tqdm(dataloader):
    targets = batch[1]
    for key in targets.keys():
        sum_of_squares[key] += (targets[key].block().values ** 2).sum().item()
        num_elements[key] += targets[key].block().values.numel()

standard_deviations = {key: (sum_of_squares[key] / num_elements[key]) ** 0.5 for key in targets.keys()}
print("Standard Deviations:")
for key, std in standard_deviations.items():
    print(f"{key}: {std:.4f}")

with open(f'{dataset_name[:-4]}.json', 'w') as f:
    json.dump(standard_deviations, f, indent=4)
