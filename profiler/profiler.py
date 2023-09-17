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


def find_root(events):
    for event in events:
        if event["name"].split(" ")[-1] == "layers_forward_with_loss":
            return event


def find_layers(root):
    layers = []
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

    root = find_root(python_function_event)

    layers = find_layers(root)

    for layer in layers:
        print(layer["name"] + ": " + str(layer["dur"]) + "us")
