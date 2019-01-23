# note plain $* doesn't work you need the 'eval' or 'echo hi; echo bye' for example would not work
ORIGINAL=$PWD
eval $*
echo $PWD > $ORIGINAL/.$PPID.PWD

#if [ "$1" = "capture" ]; then
#    shift
#    eval $* | tee .$PPID.stdout
#elif [ "$1" = "nocapture" ]; then
#    shift
#    eval $*
#else
#    echo "incorrect usage of sh backend .sh script"
#fi
