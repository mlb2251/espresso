FILE=$1
echo $$ $1 >> ~/.espresso/backend_process_log.txt
while true; do
    eval `cat $FILE` &> $FILE
done
