# resolution values
# 320 x 180
# 640 x 360
# 960 x 540
# 1280 x 720

import csv
import math
import matplotlib.pyplot as plt
import itertools
import statistics
import copy

start_time_analysis = 115

resolution_set = [320 * 180, 640 * 360, 960 * 540, 1280 * 720]

throughput_list = []
resolution_list = []
qpsum_per_sec_list = []
actual_qoe_list = []
predicted_qoe_list = []
qoe_sum_list = []
actual_frame_jitter_list = []

frame_rate_constant = 100000

file_name = "20mbps_Test_19_features.csv"
file_path = "/Users/shubhamtalbar/Desktop/jsonParser/Data/" + file_name

# def calculate_qoe(alpha, beta, gamma, curr_res, min_res, qp_sum, frame_jitter):

csv_file = None
with open(file_path, 'r') as file:
    csv_file = csv.DictReader(file)
    for row in csv_file:
        throughput = int(dict(row)['outbound_packetsSent/s'])
        resolution = int(dict(row)['outbound_frameWidth']) * int(dict(row)['outbound_frameHeight'])
        frames_per_sec = int(dict(row)['outbound_framesPerSecond'])
        qpsum_per_sec = float(dict(row)['outbound_qpSum/framesEncoded']) * frames_per_sec

        if throughput_list and throughput == 0:
            throughput = throughput_list[-1]

        throughput_list.append(throughput)
        resolution_list.append(resolution)
        qpsum_per_sec_list.append(qpsum_per_sec)

future_window_size = 5
past_window_size = 10

current_time = 11
final_analysis_time = len(throughput_list)

while current_time < final_analysis_time:
    frame_jitter = 0

    index = current_time
    while index > current_time - past_window_size:
        frame_jitter += abs(resolution_list[index] / throughput_list[index] - \
                        resolution_list[index - 1] / throughput_list[index - 1])
        index -= 1

    actual_frame_jitter_list.append(frame_jitter)
    current_time += 1

alpha = 1/(math.log(statistics.mean(resolution_list)) - math.log(min(resolution_list)))
beta = 1/statistics.mean(qpsum_per_sec_list)
gamma = 1/statistics.mean(actual_frame_jitter_list)
print(alpha, beta,gamma)
current_time = start_time_analysis
final_analysis_time = start_time_analysis + future_window_size

while current_time < final_analysis_time:
    frame_jitter = 0

    index = current_time
    while index > current_time - past_window_size:
        frame_jitter += abs(resolution_list[index] / throughput_list[index] - \
                        resolution_list[index - 1] / throughput_list[index - 1])
        index -= 1

    minimum_resolution = min(resolution_list[start_time_analysis-past_window_size : start_time_analysis])
    qoe = alpha * (math.log(resolution_list[current_time]) - math.log(minimum_resolution)) - \
          beta * qpsum_per_sec_list[current_time] - gamma * frame_jitter
    actual_qoe_list.append(qoe)
    current_time += 1

resolution_list_permutations = list(itertools.product(resolution_set, repeat=future_window_size))

print("Total permutations to try : " + str(len(resolution_list_permutations)))

perm_num = 1

for resolution_permutation in resolution_list_permutations:
    # print("Executing permutation# " + str(perm_num))

    current_time = start_time_analysis
    copy_of_resolution_list = copy.deepcopy(resolution_list)
    itr = current_time
    while itr < final_analysis_time:
        copy_of_resolution_list[itr] = resolution_permutation[itr - current_time]
        itr += 1

    qoe_value_list = []
    current_time = start_time_analysis
    final_analysis_time = start_time_analysis + future_window_size

    while current_time < final_analysis_time:
        frame_jitter = 0

        index = current_time
        while index > current_time - past_window_size:
            frame_jitter += abs(copy_of_resolution_list[index] / throughput_list[index] - \
                            copy_of_resolution_list[index - 1] / throughput_list[index - 1])
            index -= 1

        minimum_resolution = min(copy_of_resolution_list[start_time_analysis - past_window_size: start_time_analysis])
        qoe = alpha * (math.log(copy_of_resolution_list[current_time]) - math.log(minimum_resolution)) - \
              beta * qpsum_per_sec_list[current_time] - gamma * frame_jitter
        qoe_value_list.append(qoe)
        current_time += 1

    predicted_qoe_list.append(qoe_value_list)
    qoe_sum_list.append(sum(qoe_value_list))
    perm_num += 1

plt.figure()
plt.subplot(121)
plt.grid()

y_ticks_labels = ['320 x 180','640 x 360','960 x 540','1280 x 720']

xs = range(future_window_size)
ys = resolution_list[start_time_analysis:start_time_analysis+future_window_size]
plt.plot(xs, ys, 'o:r')

max_qoe_sum_index = qoe_sum_list.index(max(qoe_sum_list))
ys = resolution_list_permutations[max_qoe_sum_index]
plt.plot(xs, ys, 'o:g')

plt.title('Actual Vs Suggested Resolution')
plt.xlabel('Time')
plt.ylabel('Resolution')
plt.legend(["Actual Resolution", "Suggested Resolution"])

plt.subplot(122)
xs = range(future_window_size)
ys = actual_qoe_list
plt.plot(xs, ys, 'o-r')

plt.subplot(122)
max_qoe_sum_index = qoe_sum_list.index(max(qoe_sum_list))
ys = predicted_qoe_list[max_qoe_sum_index]
plt.plot(xs, ys, 'o-g')

qoe_diff = round(sum(predicted_qoe_list[max_qoe_sum_index]) - sum(actual_qoe_list), 0)
plt.title('Actual Vs Predicted QoE' + ' (Gain in QoE : ' + str(qoe_diff) + ')')
plt.xlabel('Time')
plt.ylabel('QoE')
plt.legend(["Actual QoE", "Predicted QoE"])

plt.show()