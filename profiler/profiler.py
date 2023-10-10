import json
import argparse


def get_python_function_event(data):
    trace_events = data["traceEvents"]
    python_function_event = []
    for trace_event in trace_events:
        if ("cat" in trace_event) and (trace_event["cat"] == "python_function"):
            python_function_event.append(trace_event)
            trace_event["son"] = []
            if trace_event["args"]["Python parent id"] != None:
                python_function_event[trace_event["args"]["Python parent id"] - 1][
                    "son"
                ].append(trace_event)
    return python_function_event


# module -> vit_0 -> forward -> transformer_0 -> forward
# 规定模型样式后，修改为寻找特定 flag 的版本
def find_layers(root):
    layers = []
    for son in root["son"]:
        if "name" in son and son["name"].split(" ")[-1] == "ViT_0":
            root = son
            break

    for son in root["son"]:
        if "name" in son and son["name"].split(" ")[-1] == "forward":
            root = son
            break

    for son in root["son"]:
        if "name" in son and son["name"].split(" ")[-1] == "Transformer_0":
            root = son
            break

    for son in root["son"]:
        if "name" in son and son["name"].split(" ")[-1] == "forward":
            root = son
            break

    for son in root["son"]:
        if "name" in son and son["name"].split(" ")[0] == "nn.Module:":
            layers.append(son)

    return layers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        dest="file",
        type=str,
        default="nico3_964181.1694573568162.pt.trace.json",
        help="torch profiler log path",
    )
    args = parser.parse_args()

    with open(args.file, "r") as file:
        data = json.load(file)

    python_function_event = get_python_function_event(data)

    layers = find_layers(python_function_event[0])

    for layer in layers:
        print(layer["name"] + ": " + str(layer["dur"]) + "us")
