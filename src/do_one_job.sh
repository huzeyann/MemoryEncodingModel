# python file as arg
x=$1
echo "LOG START" > /tmp/log
while true; do
    echo "======================" >> /tmp/log
    echo $(date) >> /tmp/log
    echo "Running ${x}" >> /tmp/log
    starttime=$(date +%s)
    python -u ${x} >> /tmp/log 2>&1
    wait
    endtime=$(date +%s)
    echo "Finished ${x}" >> /tmp/log
    timeelapsed=$((endtime - starttime))
    echo "Time elapsed: ${timeelapsed}" >> /tmp/log
    echo "======================" >> /tmp/log
    timethreshold=300
    if [ $timeelapsed -gt $timethreshold ]; then
        echo "Time elapsed ${timeelapsed} is greater than threshold ${timethreshold}" >> /tmp/log
        echo "Sleeping for 10 seconds"
        sleep 10
    else
        echo "Time elapsed ${timeelapsed} is less than threshold ${timethreshold}" >> /tmp/log
        echo "go to vacation" >> /tmp/log
        break
    fi
done
echo "LOG END" >> /tmp/log

sleep infinity