for i in {1..20}
do
  for iter in {50..300..50}
  do
    gpuFileName="${PWD}/images/julia/gpuOut_${i}_${iter}.png"
    cpuFileName="${PWD}/images/julia/cpuOut_${i}_${iter}.png"
    echo "python3 main.py --julia $i --width 300 --gpu_output \
$gpuFileName --cpu_output $cpuFileName --timing $PWD/julia_times.csv \
--block 8 --iterations ${iter}"
    python3 main.py --julia $i --size 300 --gpu_output $gpuFileName \
    --cpu_output $cpuFileName --block 8 \
    --iterations $iter
  done
done
