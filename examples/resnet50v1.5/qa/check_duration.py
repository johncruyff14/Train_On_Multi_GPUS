import argparse
import json
from math import isfinite

def str_list(x):
    return x.split(',')

parser = argparse.ArgumentParser(description='Performace Tests')
parser.add_argument('baseline', metavar='DIR', help='path to baseline')
parser.add_argument('report', metavar='DIR', help='path to report')
parser.add_argument('--tolerance', default=0.15, type=float)
args = parser.parse_args()

def check(baseline, report):
    maxv = baseline['total_duration'] * (1 + args.tolerance)
    r = float(report[-1]['elapsedtime'])
    if not isfinite(r) or r > maxv:
        print("Result value doesn't match baseline: total_duration: allowed max: {}, result: {}".format(
                maxv, r))
        return False
    return True


with open(args.report, 'r') as f:
    lines = f.read().splitlines()
    log_data = [json.loads(line[5:]) for line in lines]
    epochs_report = list(filter(lambda x: len(x['step']) == 1, log_data))

with open(args.baseline, 'r') as f:
    baseline_json = json.load(f)

if check(baseline_json, epochs_report):
    print("&&&& DURATION PASSED")
    exit(0)
else:
    print("&&&& DURATION FAILED")
    exit(1)
