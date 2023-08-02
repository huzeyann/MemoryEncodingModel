# /bin/bash

if [[ "$#" -lt 3 ]]; then
    echo "no argument, using default values..."
    repeat_check=3
    sm_th=10
    mem_th=10
else
    repeat_check=$1
    sm_th=$2
    mem_th=$3
fi

echo "Finding Available GPUs:"


NUM_GPUS=`nvidia-smi --list-gpus | wc -l`
echo "Total $NUM_GPUS GPUs found"

for (( i=$NUM_GPUS-1; i>=0; i-- )); do
    total_sm=0
    max_sm=0
    for (( j=0; j<$repeat_check; j++)); do
        sm=`nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $i`
        echo "$(date) GPU $i is $sm% busy"
        total_sm=$(expr "$total_sm" + "$sm")
        if [[ "$sm" -gt "$max_sm" ]]; then
            max_sm=$sm
        fi
        sleep 1
    done
    sm=$total_sm
    avg_sm=$(expr "$sm" / "$repeat_check")
    echo "GPU $i is average $avg_sm% busy, max $max_sm% busy"
    
    mem_used=`nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i`
    mem_total=`nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $i`
    mem_percent=$(expr "$mem_used" \* 100 / "$mem_total")
    echo "GPU $i has $mem_percent% memory used"

    if [[ "$max_sm" -lt "$sm_th" ]] && [[ "$mem_percent" -lt "$mem_th" ]]; then
        echo "available, stopping job..."
        docker stop sspy_$i
    else
        echo "busy, skipping..."
    fi
done

echo "Done"