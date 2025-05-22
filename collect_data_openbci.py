import argparse
import time
from pprint import pprint

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

import json

marker_dict = {
    0: "Nothing",
    1: "Left Hand", 
    2: "Right Hand", 
    3: "Rest"
}

prompt_order = [0, 3, 3, 2, 3, 1, 3, 2, 3, 2, 3, 1, 3, 1, 3, 2, 3, 1, 3, 3, 0]

def add_nothing_prompts(lst):
    result = []
    for i, val in enumerate(lst):
        result.append(val)
        if i != len(lst) - 1:
            result.append(0)
    return result

print(prompt_order)
prompt_order = add_nothing_prompts(prompt_order)
print(prompt_order)

def main():

    board_id = BoardIds.CYTON_BOARD
    
    BoardShim.enable_dev_board_logger()

    params = BrainFlowInputParams()
    params.serial_port = "/dev/ttyUSB1"

    pprint(BoardShim.get_board_descr(board_id))

    board = BoardShim(BoardIds.CYTON_BOARD, params)
    board.prepare_session()
    board.start_stream()

    iter = 0
    prompt_iter = 0

    done = False

    eeg_samples = []

    eeg_markers = []

    prev_time = time.time()

    while not done:
        iter = iter + 1
        time.sleep(0.1)

        cur_time = time.time()

        if cur_time - prev_time > 2: 
            print(marker_dict[prompt_order[prompt_iter]])
            prompt_iter += 1
            prev_time = cur_time

            if prompt_iter >= len(prompt_order):
                done = True
                continue

        # data = board.get_current_board_data (256) # get latest 256 packages or less, doesnt remove them from internal buffer
        data = board.get_board_data()  # get all data and remove it from internal buffer

        channels = board.get_eeg_channels(board_id)
        # print(channels)


        if iter == 1:
            print("Number of Channels:", len(data))

        # print(time.time())

        eeg_sample = [data[i].tolist() for i in channels]

        # print(eeg_sample)

        eeg_samples.append(eeg_sample)
        eeg_markers.append(prompt_order[prompt_iter])

        # print(data[2])
        
        # break

        # print(f"---------{iter}---------")

        # print("Channel 1:", eeg_sample[0])
        # print("Channel 2:", eeg_sample[1])
        # print("Channel 3:", eeg_sample[2])
        # print("Channel 4:", eeg_sample[3])
        # print("Channel 5:", eeg_sample[4])
        # print("Channel 6:", eeg_sample[5])
        # print("Channel 7:", eeg_sample[6])
        # print("Channel 8:", eeg_sample[7])
       
        # print("Channel 0:", data[0])
        # print("Channel 1:", data[1])
        # print("Channel 2:", data[2])
        # print("Channel 3:", data[3])
        # print("Channel 4:", data[4])
        # print("Channel 5:", data[5])
        # print("Channel 6:", data[6])
        # print("Channel 7:", data[7])
        # print("Channel 8:", data[8])
        # print("Channel 9:", data[9])
        # print("Channel 10:", data[10])
        # print("Channel 11:", data[11])
        # print("Channel 12:", data[12])
        # print("Channel 13:", data[13])
        # print("Channel 14:", data[14])
        # print("Channel 15:", data[15])
        # print("Channel 16:", data[16])
        # print("Channel 17:", data[17])
        # print("Channel 18:", data[18])
        # print("Channel 19:", data[19])
        # print("Channel 20:", data[20])
        # print("Channel 21:", data[21])
        # print("Channel 22:", data[22])
        # print("Channel 23:", data[23])

    # return

    board.stop_stream()
    board.release_session()

    samples_path = "eeg_samples.json"
    markers_path = "eeg_markers.json"

    with open(samples_path, 'w') as json_file1:
        json.dump(eeg_samples, json_file1)

    with open(markers_path, 'w') as json_file2:
        json.dump(eeg_markers, json_file2)


if __name__ == "__main__":
    main()