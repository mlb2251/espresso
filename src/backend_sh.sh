# maybe migrate to using stdin for input? Not sure if that would be less intensive than reading from a pipe bc a pipe is kinda like a file so sorta like a disk object? Tho I think stdin exists somewhere as a pipe disk object anyways so who knows
IN=$1       # first arg is pipe to get input from
OUT=$2      # second arg is pipe to write output to

cleanup(){
        #echo "[bash backend shutting down...]"
        rm $IN
        rm $OUT
        exit 0
}
trap 'cleanup' EXIT #cleanup on any exit
trap '' INT TERM HUP QUIT # ignore signals. Important bc child processes recv all signals sent to their parents.

LINE=''
exec 3<> $IN
while true; do
    read -t 2 -u 3 -s LINE # -t = timeout -u = from file descriptor *could use -d to change the line terminator from \n
    if [ $? -eq 0 ]; then #if read succeeded
        #echo "eval $LINE &> $FILE"
        eval $LINE &> $OUT
    fi
    PARENT=`ps -o ppid= $$` #because $PPID doesn't update
    #echo child: $$ parent: $PARENT
    if [ $PARENT -eq 1 ]; then # shut down if parent dies
        exit 0 #this will naturally call cleanup()
    fi
done
