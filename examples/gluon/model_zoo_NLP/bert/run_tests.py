
import sys, os
import subprocess
from utils import remove_data

worse_tests = []
better_tests = []
file_name = 'run_times.txt'
res_file = 'res_all.txt'

def compare(test_name, a, b, tolerance=5):
    """Comparing current (a) and base (b) run times an report when the difference is bigger than tolerance (defined in %'s)"""
    if b > 0:
        diff = a - b
        abs_diff = abs(diff)
        tol = 100 * abs_diff / a
        if tol <= tolerance:
            return
        global worse_tests, better_tests, res_file
        if diff < 0:
            better_tests.append([test_name, a, b])
            adverb = 'better'
        else:
            worse_tests.append([test_name, a, b])
            adverb = 'worse'
        str = " For test '{}' the run time ({}) is {} than the base run time ({})".format(test_name, a, adverb, b)
    else:
        str = " For test '{}' the run time ({}) was defined for the first time".format(test_name, a)
    print(str)
    f = open(res_file, "a")
    f.write(str)
    f.close()

def report_results(tests, adverb):
    l = len(tests)
    if l == 0:
        return
    if l == 1:
        print("\nFollowing test shows {} run time:".format(adverb))
    else:
        print("\nFollowing {} tests show {} run times:".format(l, adverb))

    for test_res in tests:
        print("   {}  run time={}  base_run_time={}".format(test_res[0], test_res[1], test_res[2]))


if __name__ == "__main__":
    l = len(sys.argv)

    test_number = int(sys.argv[1])
    num_tests = int(sys.argv[2])
    tolerance = float(sys.argv[3])
    if test_number <= 0:
        print("Will run {} tests".format(num_tests))
    else:
        test_number -= 1
        print("Will run only '{}'".format(sys.argv[test_number+4]))

    if os.path.exists(res_file):
        os.remove(res_file)

    remove_data()

    base_line = -1
    for i in range(num_tests):
        # Check, if specific test should be launched
        if test_number >= 0 and i != test_number:
            continue

        test_name = sys.argv[i+4]
        base_line = float(sys.argv[i+4+num_tests]) if l > i+4+num_tests else -1
        f = open(file_name, 'a')
        f.write('{} '.format(test_name))
        f.close()
        task_name_sh = "./" + test_name + ".sh"
        ret_val = subprocess.call(task_name_sh, shell=True)
        if ret_val != 0:
            print("Test '% s' failed..." % test_name)
            continue

        if base_line >= 0:
            f = open(file_name)
            # Reading all lines, we take the last one
            lines = f.readlines()
            f.close()
            line = lines[-1].strip()
            # Skip test_name and convert remaining part of the string into floating point number
            curr_time = float(line[len(test_name)+1:])
            compare(test_name, curr_time, base_line, tolerance)

    if base_line >= 0:
        if len(better_tests) or len(worse_tests):
            report_results(better_tests, "better")
            report_results(worse_tests, "worse")
        else:
            print("\nThe run times of all tests are inside the tolerance of {}% from the base line".format(tolerance))
    else:
        print("\nRun times were NOT compared.")
        print("Please, add base run times from '{}' to 'BaseLines.txt' or your BaseLineCust file for computer you just used".format(file_name))

