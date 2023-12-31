set -ex
trap "exit" INT TERM
trap "kill 0" EXIT

# cargo build --example sha256
# BIN=../target/debug/examples/sha256

cargo build --release --example sha256 --features parallel
BIN=../target/release/examples/sha256

l=2
t=1
m=32768
n=8

for n_parties in $n
do
  PROCS=()
  for i in $(seq 0 $(($n_parties - 1)))
  do
    #$BIN $i ./network-address/4 &
    if [ $i == 0 ]
    then
      RUST_BACKTRACE=0 RUST_LOG=sha256 $BIN $i ../network-address/$n_parties $l $t $m &
      pid=$!
      PROCS[$i]=$pid
    else
      RUST_LOG=sha256 $BIN $i ../network-address/$n_parties $l $t $m > /dev/null &
      pid=$!
      PROCS[$i]=$pid
    fi
  done
  
  for pid in ${PROCS[@]}
  do
<<<<<<< HEAD
    wait $pid || { echo "Process $pid exited with an error status"; exit 1; }
=======
    wait $pid
>>>>>>> 6c6268a (Rebase)
  done
done

echo done

