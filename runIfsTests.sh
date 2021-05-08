for iter in 10000 50000 100000 150000 2000000 300000
do
  for ifs in "fern" "dragon" "pentadentrite" "koch" "spiral" "tree" "twig"
  do
    gpuFileName="${PWD}/images/ifs/${ifs}_${iter}.png"
    cpuFileName="${PWD}/images/ifs/${ifs}_${iter}.png"
    echo "python3 main.py --ifs $ifs --width 300 --gpu_output \
      $gpuFileName --cpu_output $cpuFileName --timing $PWD/ifs_times.csv \
      --block 8 --iterations ${iter}"
    python3 main.py --ifs ${ifs} --size 300 --gpu_output ${gpuFileName} \
    --cpu_output ${cpuFileName} --timing $PWD/ifs_times.csv --block 8 \
    --iterations ${iter}
  done
done
