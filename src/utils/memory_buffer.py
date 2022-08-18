import random

class MemoryBuffer:
    # Class constructor
    def __init__(self, memory_size):
        self.nr_of_observed_data_samples = 0  # The number of encountered distinct data samples (x, u)
        self.nr_of_observed_task_label_pairs = 0  # The nr of encountered pairs (u, y)
        self.memory_size = memory_size  # The nr of data samples that can fit in the memory
        self.memory = {}  # The current memory buffer and its contents. A dict of lists

    # Call this function if you observe a data sample, which may, depending on the memory buffer settings, be added to
    # the memory
    def observe_sample(self, data_sample, task_id, label):
        self.nr_of_observed_data_samples += 1  # Update this at the start to include it in the probability calculation
        # TODO: should be unique data samples

        if self.__get_memory_filled_size() < self.memory_size:
            # Surely add the data sample since we have space
            self.__add(data_sample, task_id, label, is_memory_full=False)
        else:
            # Roll a dice to see if we add the observed data sample to the memory or not
            probability_to_hit = self.memory_size / self.nr_of_observed_data_samples
            if probability_to_hit >= random.uniform(0, 1):
                self.__add(data_sample, task_id, label, is_memory_full=True)

    # Call this function if you want the data samples of a specific task (and possibly label)
    def get_samples(self, task_id=None, label=None):
        if task_id is None:
            return self.memory
        else:
            if label is None:
                # Loop over all keys to find the keys where task_id=task_id
                # Return as a flattened list
                entries = []
                for key in self.memory:
                    # TODO: test whether removing a label leads to an error
                    if key[0] == task_id and key[1] != 9:
                        for entry in self.memory[key]:
                            x_and_y = (entry, key[1])
                            entries.append(x_and_y)
                return entries
            else:
                return self.memory[(task_id, label)]

    # Internal function to add a data sample to the memory
    def __add(self, data_sample, task_id, label, is_memory_full=False):
        entry_to_add = data_sample
        entry_key = (task_id, label)
        if not is_memory_full:
            self.__add_to_memory(entry_to_add, entry_key)
        else:
            # Find the most represented (task_id, label) pair
            # This can possibly be multiple in case multiple pairs have the same number of saved data samples
            keys_and_sizes_in_memory = [{'key': key, 'size': len(self.memory[key])} for key in self.memory]
            keys_and_sizes_in_memory.sort(reverse=True, key=lambda x: x['size'])
            largest_representation = max([key_and_size['size'] for key_and_size in keys_and_sizes_in_memory])
            most_represented_pairs = [key_and_size['key'] for key_and_size in keys_and_sizes_in_memory if key_and_size['size'] == largest_representation]

            # Pick a random pair, from the most_represented_pairs, and index to replace
            chosen_most_represented_pair = random.choice(most_represented_pairs)
            chosen_index = random.randint(0, len(self.memory[chosen_most_represented_pair]) - 1)

            # Remove
            self.__remove(chosen_most_represented_pair, chosen_index)

            # Add the new data sample
            self.__add_to_memory(entry_to_add, entry_key)

    # Internal function check if a (task_id, label) was already added to the memory or not, to easily add samples
    def __add_to_memory(self, entry_to_add, entry_key):
        if entry_key not in self.memory:
            self.memory[entry_key] = [entry_to_add]
        else:
            self.memory[entry_key].append(entry_to_add)

    # Internal function to remove a data sample from the memory
    def __remove(self, entry_to_remove_key, entry_to_remove_index):
        self.memory[entry_to_remove_key].pop(entry_to_remove_index)

    # Gets the number of data samples added to the memory
    def __get_memory_filled_size(self):
        return sum([1 for key in self.memory for entry in self.memory[key]])
