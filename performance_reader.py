import zipfile
import json
from dateutil import parser

# Change this name if you want to evaluate values for a different run
# Note that the name of this .zip file should be <some_string>_<some_other_string>.zip and that the according .out file
# should have a matching name: slurm-<some_string>.out
ZIP_FILE_NAME = "1_traces.zip"

FLOPS_PER_FW_PASS = 409419160
BATCH_SIZE = 256
INDEPENDENT_MODEL_LAST_TASK_ACCURACIES = {"minus": 0.388,
                                          "plus": 0.711,
                                          "out": 0.365,
                                          "in": 0.627,
                                          "pl": 0.711,
                                          "long": 0.498}

def get_slurm_file():
    run_number = ZIP_FILE_NAME.split("_")[0]
    return "slurm-" + str(run_number) + ".out"

def get_sec(time_str):
    """Get seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

def main():
    print("Calculating the performance given an output .zip file '" + ZIP_FILE_NAME + "' of Independent, MNTDP or"
                                                                                      " devised algorithms:")

    zip_file = zipfile.ZipFile(ZIP_FILE_NAME, "r")
    slurm_file_name = get_slurm_file()
    total_iterations = 0
    tasks_count = 0
    performance_avg_acc = 0
    avg_acc_when_seen = 0
    total_nr_of_params = 0
    total_iterations = 0
    start_time = None
    end_time = None
    performance_transfer_last_task = 0
    is_double_resources = False
    memory_size_mb = 0
    manually_calculated_accuracies = []

    is_first_line = True
    with zip_file.open("1_main") as f:
        for line in f:
            line_str = line.decode("utf-8")
            if "update" in line_str:
                # Accumulate the total nr of iterations
                if "n iterations to find the best model" in line_str:
                    iterations = int(line_str.split("\"y\": [")[1].split("], ")[0])
                    total_iterations += iterations
                    tasks_count += 1

                # Continually update the performance values, as the 1_main file is temporally ordered, we always get the last
                # value (which is the correct one)
                if "Average Accuracies now" in line_str:
                    performance_avg_acc = float(line_str.split("\"y\": [")[1].split("], ")[0])
                if "Average Accuracies when seen" in line_str:
                    avg_acc_when_seen = float(line_str.split("\"y\": [")[1].split("], ")[0])
                if "Total # of params used by the learner" in line_str:
                    total_nr_of_params = float(line_str.split("\"y\": [")[1].split("], ")[0])
                if "Training Durations" in line_str:
                    # This is only measured per task, so hence the +
                    total_iterations += float(line_str.split("\"y\": [")[1].split("], ")[0])

                if "Learning_accuracies" in line_str and "test" not in ZIP_FILE_NAME:
                    test_performance_current_task = float(line_str.split("\"y\": [")[1].split("], ")[0])
                    stream_name = None
                    if "minus" in ZIP_FILE_NAME:
                        stream_name = "minus"
                    elif "plus" in ZIP_FILE_NAME:
                        stream_name = "plus"
                    elif "out" in ZIP_FILE_NAME:
                        stream_name = "out"
                    elif "in" in ZIP_FILE_NAME:
                        stream_name = "in"
                    elif "pl" in ZIP_FILE_NAME:
                        stream_name = "pl"
                    elif "long" in ZIP_FILE_NAME:
                        stream_name = "long"
                    performance_transfer_last_task = test_performance_current_task - INDEPENDENT_MODEL_LAST_TASK_ACCURACIES[stream_name]

            elif "events" in line_str and is_first_line:
                content_str = str(json.loads(line_str)[1]['data'][0]['content'])
                start_time_str = content_str.split("<")[0].replace('Started at ', '')
                start_time = parser.parse(start_time_str)
            is_first_line = False

    try:
        total_time = 0
        with open(slurm_file_name) as f:
            for line in f:
                if "INFO:src.utils.log_observer:completed_event " in line:
                    end_time_str = line.replace('INFO:src.utils.log_observer:completed_event ', '').replace('\n', '')
                    end_time = parser.parse(end_time_str)
                    break
                if "INFO:src.experiments.stream_tuning:Args" in line and "double" in line:
                    is_double_resources = True
                if "[RESULT] Evaluation accuracy for task" in line:
                    line_split = line.split(" ")
                    manually_calculated_accuracies.append(float(line_split[len(line_split)-1].strip()))
                if "[RESULT] Updated task" in line and "parameters" in line:
                    line_split = line.split(" ")
                    total_nr_of_params -= int(line_split[len(line_split)-2])
                if "[RESULT] Used memory with" in line:
                    line_split = line.split(" ")
                    memory_size_mb = float(line_split[len(line_split)-4])
                if "[RESULT] Updated task" in line and "to update the evaluation accuracy from" in line:
                    line_split = line.split(" ")
                    updated_task = int(line_split[3].strip())
                    new_acc = float(line_split[len(line_split)-1].strip())

                    # Replace the saved average accuracy. At this point all accuracies per task should have been
                    # calculated already
                    manually_calculated_accuracies[updated_task] = new_acc
    except:
        print("[WARNING]: NO SLURM FILE FOUND FOR INPUT ZIP FILE")

    print("#tasks:", tasks_count)
    print("A(S):")
    if manually_calculated_accuracies:
        performance_avg_acc = sum(manually_calculated_accuracies) / len(manually_calculated_accuracies)
    print("%.3f" % performance_avg_acc)
    print("F(S):")
    if "%.3f" % (performance_avg_acc - avg_acc_when_seen) == "-0.000":
        print("0.000")
    else:
        print("%.3f" % (performance_avg_acc - avg_acc_when_seen))
    print("T(S):")
    print("%.3f" % performance_transfer_last_task)
    print("M(S):")
    print("%.3f" % ((total_nr_of_params * 32 * 0.000000125) + memory_size_mb))
    print("C(S):")
    print("%.3f" % ((FLOPS_PER_FW_PASS * 3 * total_iterations * BATCH_SIZE) / 1000000000000000))
    print("W(S):")
    total_time = abs((start_time - end_time).total_seconds())
    total_time = total_time if not is_double_resources else total_time * 2
    print("%.3f" % total_time)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
