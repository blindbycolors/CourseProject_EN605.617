for ifs in "leaf" "fern" "dragon" "pentadentrite" "koch" "spiral" "tree" "twig"
do
  for numPoint in 10000 50000 100000 150000 2000000 300000
  do
    gpuFileName="${PWD}/images/ifs/gpu_${ifs}_${numPoint}.png"
    cpuFileName="${PWD}/images/ifs/cpu_${ifs}_${numPoint}.png"
    echo "python3 main.py --ifs $ifs --width 300 --gpu_output \
$gpuFileName --cpu_output $cpuFileName --timing $PWD/ifs_times.csv \
--block 8 --points ${numPoint}"
    python3 main.py --ifs ${ifs} --size 300 --gpu_output ${gpuFileName} \
    --cpu_output ${cpuFileName} --timing $PWD/ifs_times.csv --block 8 \
    --points ${numPoint}
  done
done
