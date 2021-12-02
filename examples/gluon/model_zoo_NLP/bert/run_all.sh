#!/bin/bash

function get_params() {
  if [ $# -ge 1 ]; then
    comp_name=$1
    if [ $comp_name == 'dgx1v' ]; then
      task_times=("${task_times_dgx1v[@]}")
    fi

    if [ $comp_name == 'dgx' ]; then
      task_times=("${task_times_dgx[@]}")
    fi

    if [ $comp_name == 't4' ]; then
      task_times=("${task_times_t4[@]}")
    fi

    if [ $comp_name == 'xpl' ]; then
      task_times=("${task_times_xpl[@]}")
    fi

    if [ $comp_name == 'v100sPCIe' ]; then
      task_times=("${task_times_v100sPCIe[@]}")
    fi

    if [ $# -ge 2 ]; then
      test_number=$2
    fi

    if [ $# -ge 3 ]; then
      tolerance=$3
    fi
  fi

    if [ $# -ge 2 ]; then
    test_number=$2
  fi

  if [ $# -ge 3 ]; then
    tolerance=$3
  fi
}

comp_name='unknown_computer'
test_number=-1   # launching all tests from task_names list
tolerance=5      # default tolerance 5%

if [[ $# -ge 1 ]] && [[ -f "$1" ]]; then
  source $1
  get_params $2 $3 $4
else
  source BaseLines.txt
  get_params $1 $2 $3
fi

if [ $# -ge 1 ]; then
  comp_name=$1
  if [ $comp_name == '-h' ] || [ $comp_name == '-H' ]; then
    echo "Launching: $0 [comp_name [test_number [tolerance]]]"
    echo "Following tests could be launched:"
    j=1
    for i in "${task_names[@]}"
    do
      printf "%3d: %s\n" $j $i
      ((j=j+1))
    done
    echo "When test_number <= 0,  all these tests will be launched"

    echo "Following set of computer names could be used:"
    j=1
    for i in "${comp_names[@]}"
    do
      printf "%3d: %s\n" $j $i
    ((j=j+1))
    done
    echo "Default value for tolerance is 5 (%)"
    exit 0
  fi
fi

# Saving computer name
file='run_times.txt'
echo $comp_name >$file

if [ $test_number -le 0 ]; then
  echo "Run time tests are launched on '$comp_name' with tolerance $tolerance"
else
  echo "Run time test# $test_number is launched on '$comp_name' with tolerance $tolerance"
fi

pip install gluonnlp
python run_tests.py $test_number ${#task_names[@]} $tolerance "${task_names[@]}" "${task_times[@]}"
