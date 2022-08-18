import torch
import random

if __name__ == '__main__':
    saved_samples = []
    saved_labels = []
    for i in range(81):
        saved_samples.append(torch.Tensor(3, 32, 32))
        rand_int = random.randint(0, 9)
        saved_labels.append(rand_int)
        print(rand_int)
    print(len(saved_samples))
    out_saved_samples_tensor = torch.Tensor(3, 32, 32)
    torch.cat(saved_samples, dim=0, out=out_saved_samples_tensor)
    print(torch.stack(saved_samples).size())

    saved_labels_tensor = torch.zeros(81, 1)
    for i in range(81):
        saved_labels_tensor[i][0] = saved_labels[i]
    print(saved_labels_tensor)

    exit(0)
    out_saved_samples_labels_tensor = torch.Tensor(len(saved_samples), 1)
    saved_samples_tensor = [entry[0] for entry in saved_samples]
    for i in saved_samples:
        print(i[0].size())
        print(torch.Tensor(i[1], dtype=torch.int8).size())
    saved_samples_labels_tensor = [torch.Tensor(entry[1]) for entry in saved_samples]
    torch.cat(saved_samples_tensor, out=out_saved_samples_tensor)
    torch.cat(saved_samples_labels_tensor, out=out_saved_samples_labels_tensor)
    print(len(saved_samples))
    print(out_saved_samples_tensor.size())
    print(out_saved_samples_tensor)
    print(out_saved_samples_labels_tensor.size())
    print(out_saved_samples_labels_tensor)