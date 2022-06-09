import json as js
import csv
import glob

# set of all the features
header = ['inbound_firCount', 'inbound_pliCount', 'inbound_nackCount', 'inbound_jitter', 'inbound_framesDropped',
          'inbound_framesReceivedPerSec',
          'outbound_firCount', 'outbound_pliCount', 'outbound_nackCount', 'outbound_qpSum', 'outbound_framesSentPerSec',
          'outbound_frameWidth', 'outbound_frameHeight']


def preprocess(input_key, data_type=0):
    temp_string = input_key[1]['values']
    temp_string = temp_string.split(']')
    temp_string = temp_string[0].split('[')
    temp_string = temp_string[1]

    if data_type == 0:
        feature_vector = [int(s) for s in temp_string.split(',')]
    else:
        feature_vector = [float(s) for s in temp_string.split(',')]
    return feature_vector


def parse_directory(ip_directory_path, op_directory_path):
    file_names = glob.glob(ip_directory_path + "*/*.txt")
    counter = 1

    for file_name in file_names:
        print("processing file_name = " + file_name)
        parse_file(file_name, op_directory_path + str(counter) + "_features" + ".csv")
        # write_features(op_directory_path + counter + "_features" + ".csv")
        counter += 1


def parse_file(file_path, output_file_path):
    f = open(file_path, "r")
    json_string = f.read()
    f.close()

    json = js.loads(json_string)

    inbound_firCount = None
    inbound_pliCount = None
    inbound_nackCount = None
    inbound_jitter = None
    inbound_framesDropped = None
    inbound_framesReceivedPerSec = None

    outbound_firCount = None
    outbound_pliCount = None
    outbound_nackCount = None
    outbound_qpSum = None
    outbound_frameWidth = None
    outbound_frameHeight = None
    outbound_framesSentPerSec = None

    for peer_connection in json["PeerConnections"]:
        peer_connection_value = json["PeerConnections"][peer_connection]
        peer_stats = peer_connection_value["stats"]

        for key in peer_stats.items():
            key_name = key[0]
            if key_name.find("RTCInboundRTPVideoStream") != -1:
                # print("Inbound parameter name -" + key_name)
                if key_name.find("firCount") != -1:
                    # print("Inbound parameter name -" + key_name)
                    # nonlocal inbound_firCount
                    inbound_firCount = preprocess(key)
                elif key_name.find("pliCount") != -1:
                    # print("Inbound parameter name -" + key_name)
                    # nonlocal inbound_pliCount
                    inbound_pliCount = preprocess(key)
                elif key_name.find("nackCount") != -1:
                    # print("Inbound parameter name -" + key_name)
                    # nonlocal inbound_nackCount
                    inbound_nackCount = preprocess(key)
                elif key_name.find("jitter") != -1:
                    # print("Inbound parameter name -" + key_name)
                    # nonlocal inbound_jitter
                    inbound_jitter = preprocess(key)
                elif key_name.find("framesDropped") != -1:
                    # print("Inbound parameter name -" + key_name)
                    # nonlocal inbound_framesDropped
                    inbound_framesDropped = preprocess(key)
                elif key_name.find("framesReceived/s") != -1:
                    # print("Inbound parameter name -" + key_name)
                    # nonlocal inbound_framesReceivedPerSec
                    inbound_framesReceivedPerSec = preprocess(key)

            elif "RTCOutboundRTPVideoStream" in key_name:
                # print("Outbound parameter -" + key_name)
                if key_name.find("firCount") != -1:
                    # print("Outbound parameter -" + key_name)
                    # nonlocal outbound_firCount
                    outbound_firCount = preprocess(key)
                elif key_name.find("pliCount") != -1:
                    # print("Outbound parameter -" + key_name)
                    # nonlocal outbound_pliCount
                    outbound_pliCount = preprocess(key)
                elif key_name.find("nackCount") != -1:
                    # print("Outbound parameter -" + key_name)
                    # nonlocal outbound_nackCount
                    outbound_nackCount = preprocess(key)
                elif key_name.find("qpSum") != -1 and key_name.find("qpSum/framesEncoded") == -1:
                    # print("Outbound parameter -" + key_name)
                    # nonlocal outbound_qpSum
                    outbound_qpSum = preprocess(key)
                elif key_name.find("frameWidth") != -1:
                    # print("Outbound parameter -" + key_name)
                    # nonlocal outbound_frameWidth
                    outbound_frameWidth = preprocess(key)
                elif key_name.find("frameHeight") != -1:
                    # print("Outbound parameter -" + key_name)
                    # nonlocal outbound_frameHeight
                    outbound_frameHeight = preprocess(key)
                elif key_name.find("framesSent/s") != -1:
                    # print("Outbound parameter -" + key_name)
                    # nonlocal outbound_framesSentPerSec
                    outbound_framesSentPerSec = preprocess(key, 1)

    with open(output_file_path, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for index in range(len(inbound_firCount)):
            data = []

            if inbound_firCount != None and index < len(inbound_firCount):
                data.append(inbound_firCount[index])
            else:
                data.append("None")

            if inbound_pliCount != None and index < len(inbound_pliCount):
                data.append(inbound_pliCount[index])
            else:
                data.append("None")

            if inbound_nackCount != None and index < len(inbound_nackCount):
                data.append(inbound_nackCount[index])
            else:
                data.append("None")

            if inbound_pliCount != None and index < len(inbound_pliCount):
                data.append(inbound_pliCount[index])
            else:
                data.append("None")

            if inbound_jitter != None and index < len(inbound_jitter):
                data.append(inbound_jitter[index])
            else:
                data.append("None")

            if inbound_framesDropped != None and index < len(inbound_framesDropped):
                data.append(inbound_framesDropped[index])
            else:
                data.append("None")

            if inbound_framesReceivedPerSec != None and index < len(inbound_framesReceivedPerSec):
                data.append(inbound_framesReceivedPerSec[index])
            else:
                data.append("None")

            if outbound_firCount != None and index < len(outbound_firCount):
                data.append(outbound_firCount[index])
            else:
                data.append("None")

            if outbound_pliCount != None and index < len(outbound_pliCount):
                data.append(outbound_pliCount[index])
            else:
                data.append("None")

            if outbound_firCount != None and index < len(outbound_firCount):
                data.append(outbound_firCount[index])
            else:
                data.append("None")

            if outbound_nackCount != None and index < len(outbound_nackCount):
                data.append(outbound_nackCount[index])
            else:
                data.append("None")

            if outbound_qpSum != None and index < len(outbound_qpSum):
                data.append(outbound_qpSum[index])
            else:
                data.append("None")

            if outbound_framesSentPerSec != None and index < len(outbound_framesSentPerSec):
                data.append(outbound_framesSentPerSec[index])
            else:
                data.append("None")

            if outbound_frameWidth != None and index < len(outbound_frameWidth):
                data.append(outbound_frameWidth[index])
            else:
                data.append("None")

            if outbound_frameHeight != None and index < len(outbound_frameHeight):
                data.append(outbound_frameHeight[index])
            else:
                data.append("None")

            writer.writerow(data)

    inbound_firCount = None
    inbound_pliCount = None
    inbound_nackCount = None
    inbound_jitter = None
    inbound_framesDropped = None
    inbound_framesReceivedPerSec = None

    outbound_firCount = None
    outbound_pliCount = None
    outbound_nackCount = None
    outbound_qpSum = None
    outbound_frameWidth = None
    outbound_frameHeight = None
    outbound_framesSentPerSec = None

# def write_features(file_path):
#     with open(file_path, 'w', encoding='UTF8') as f:
#         writer = csv.writer(f)
#         writer.writerow(header)
#
#         for index in len(inbound_firCount):
#             data = []
#
#             data.append(inbound_firCount[index])
#             data.append(inbound_pliCount[index])
#             data.append(inbound_nackCount[index])
#             data.append(inbound_jitter[index])
#             data.append(inbound_framesDropped[index])
#             data.append(inbound_framesReceivedPerSec[index])
#
#             data.append(outbound_firCount[index])
#             data.append(outbound_pliCount[index])
#             data.append(outbound_nackCount[index])
#             data.append(outbound_qpSum[index])
#             data.append(outbound_frameWidth[index])
#             data.append(outbound_frameHeight[index])
#             data.append(outbound_framesSentPerSec[index])
#
#             writer.writerow(data)
