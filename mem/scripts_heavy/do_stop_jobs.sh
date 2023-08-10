NUM_GPUS=`nvidia-smi --list-gpus | wc -l`

echo "NUM_GPUS"
echo $NUM_GPUS

for ((i=0; i<$NUM_GPUS; i++)); do
    docker stop sspy_$i
done