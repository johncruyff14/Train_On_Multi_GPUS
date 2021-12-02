pip install nose

function set_env_vars() {
  if [ -z "$NUM_EPOCHS" ]; then
      if [ $# -ge 1 ]; then
          export NUM_EPOCHS=$1
      else
          export NUM_EPOCHS=200
      fi
  fi

  if [ -z "$WARM_EPOCHS" ]; then
      if [ $# -ge 2 ]; then
          export WARM_EPOCHS=$2
      else
          export WARM_EPOCHS=5
      fi
  fi

  if [ "$WARM_EPOCHS" -ge "$NUM_EPOCHS" ]; then
      export WARM_EPOCHS=$NUM_EPOCHS
  fi

  if [ -z "$DTYPE" ]; then
      if [ $# -ge 3 ]; then
          export DTYPE=$3
      else
          export DTYPE=float16
      fi
  fi

  if [ -z "$USE_AMP" ]; then
      if [ $# -ge 4 ] && [ $4 != '0' ]; then
          export USE_AMP="--amp"
      fi
  else
      if [ $USE_AMP != '0' ]; then
          export USE_AMP="--amp"
      else
          unset USE_AMP    # by default no AMP
      fi
  fi

  if [ ! -z "$_NUM_GPUS" ]; then
      export NUM_GPUS=$_NUM_GPUS
  fi

  gpus=0
  counter=0
  while [ $counter -lt "$NUM_GPUS" ]
  do
    if [ $counter -gt 0 ]; then
      gpus="${gpus},${counter}"
    fi
    ((counter++))
  done
  export GPUS="${gpus}"

  if [ ! -z "$_TRAIN_DATA_DIR" ]; then
      export TRAIN_DATA_DIR=$_TRAIN_DATA_DIR
  fi

  if [ ! -z "$_MODEL" ]; then
      export MODEL=$_MODEL
  fi
}