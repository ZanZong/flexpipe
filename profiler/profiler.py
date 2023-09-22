import json
import argparse
import bisect


def get_python_function_event(data):
    trace_events = data["traceEvents"]
    python_function_events = []
    for trace_event in trace_events:
        if ("cat" in trace_event) and (trace_event["cat"] == "python_function"):
            python_function_events.append(trace_event)
            trace_event["son"] = []
            if trace_event["args"]["Python parent id"] != None:
                python_function_events[trace_event["args"]["Python parent id"] - 1][
                    "son"
                ].append(trace_event)
    return python_function_events


def get_python_memory_event(data):
    trace_events = data["traceEvents"]
    python_memory_events = []
    for trace_event in trace_events:
        if ("name" in trace_event) and (trace_event["name"] == "[memory]"):
            python_memory_events.append(trace_event)
    return python_memory_events


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

    python_function_events = get_python_function_event(data)

    python_momory_events = get_python_memory_event(data)

    python_momory_events_ts = [
        python_momory_event["ts"] for python_momory_event in python_momory_events
    ]

    root = find_root(python_function_events)

    layers = find_layers(root)

    for layer in layers:
        # 认为时间区间为 [layer["ts"], layer["ts"] + layer["dur"]]
        # 按理说右侧应该为开区间，但是只有闭区间才能在 no_grad 下保证 activation = 0
        layer["alloc_before"] = python_momory_events[
            bisect.bisect_left(python_momory_events_ts, layer["ts"]) - 1
        ]["args"]["Total Allocated"]
        layer["alloc_after"] = python_momory_events[
            bisect.bisect_right(python_momory_events_ts, layer["ts"] + layer["dur"])
        ]["args"]["Total Allocated"]
        layer["activation"] = layer["alloc_after"] - layer["alloc_before"]

        print(
            layer["name"]
            + ": "
            + str(layer["dur"])
            + "us"
            + ", "
            + str(layer["alloc_before"] / 1024 / 1024)
            + "MB"
            + " "
            + str(layer["alloc_after"] / 1024 / 1024)
            + "MB"
            + " "
            + str(layer["activation"] / 1024 / 1024)
            + "MB"
        )
