import torch
import random

if __name__ == '__main__':
    dict_example = [{0: {(0, 0): '(0, 0)'}, "INs": {(0, "INs"): '(0, "INs")', (0, "INs", 0): '(0, "INs", 0)'}, 1: {(0, 1, "w"): '(0, 1, "w")', (0, 1): '(0, 1)'}, 2: {(0, 2, "w"): '(0, 2, "w")', (0, 2): '(0, 2)'}, 3: {(0, 3, "w"): '(0, 3, "w")', (0, 3): '(0, 3)'}, 4: {(0, 4, "w"): '(0, 4, "w")', (0, 4): '(0, 4)'}, 5: {(0, 5, "w"): '(0, 5, "w")', (0, 5): '(0, 5)'}, 6: {(0, 6, "w"): '(0, 6, "w")', (0, 6): '(0, 6)'}, "OUT": {(0, "OUT"): '(0, "OUT")', (0, "OUT", 0): '(0, "OUT", 0)'}},
                    {0: {}, "INs": {(1, "INs"): '(1, "INs")', (1, "INs", 0): '(1, "INs", 0)'}, 1: {}, 2: {}, 3: {}, 4: {(1, 4): '(1, 4)', (1, 4, 0, "f"): '(1, 4, 0, "f")'}, 5: {(1, 5, "w"): '(1, 5, "w")', (1, 5): '(1, 5)'}, 6: {(1, 6, "w"): '(1, 6, "w")', (1, 6): '(1, 6)'}, "OUT": {(1, "OUT"): '(1, "OUT")', (1, "OUT", 1): '(1, "OUT", 1)'}},
                    {0: {}, "INs": {(2, "INs"): '(2, "INs")', (2, "INs", 0): '(2, "INs", 0)'}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {(2, 5): '(2, 5)', (2, 5, 1, "f"): '(2, 5, 1, "f")'}, 6: {(2, 6, "w"): '(2, 6, "w")', (2, 6): '(2, 6)'}, "OUT": {(2, "OUT"): '(2, "OUT")', (2, "OUT", 2): '(2, "OUT", 2)'}},
                    {0: {(3, 0): '(3, 0)'}, "INs": {(3, "INs"): '(3, "INs")', (3, "INs", 3): '(3, "INs", 3)'}, 1: {(3, 1, "w"): '(3, 1, "w")', (3, 1): '(3, 1)'}, 2: {(3, 2, "w"): '(3, 2, "w")', (3, 2): '(3, 2)'}, 3: {(3, 3, "w"): '(3, 3, "w")', (3, 3): '(3, 3)'}, 4: {(3, 4, "w"): '(3, 4, "w")', (3, 4): '(3, 4)'}, 5: {(3, 5, "w"): '(3, 5, "w")', (3, 5): '(3, 5)'}, 6: {(3, 6, "w"): '(3, 6, "w")', (3, 6): '(3, 6)'}, "OUT": {(3, "OUT"): '(3, "OUT")', (3, "OUT", 3): '(3, "OUT", 3)'}}]

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