#!/usr/local/bin/python3

## code that goes here is stuff can can only ever be run a single time.

import sys
if sys.version_info[0] < 3 or sys.version_info[1] < 7:
    raise Exception("Must be using Python 3.7 or higher")

while True:
    try:
        import util as u
    except Exception as e:
        print(e)
        input("[Reload with ENTER]")
        continue

    ## we REALLY only want this to run once with loading lots of text from files, hence we put it in a module that never gets reloaded
    import readline
    import os

    try:
        readline.clear_history()
        readline.read_history_file(u.histfile)
        # default history len is -1 (infinite), which may grow unruly
        readline.set_history_length(1000)
    except IOError:
        pass

## `atexit` doesn't seem to work when a process exits with nonzero code like when it gets killed by a SIGHUP from the terminal, however my signal.signal() efforts havent achieved anything. By testing with SIGALRM clearly the signals are being set correctly (for alarm at least) but then they never seem to fire. Note that it could have to do with ctrl-z in iterm2 allowing for revivals -- this should be tested with Terminal.app as well. Note as well that ctrl-d is handled in repl.py and calls exit(0) after saving the history. However bash seems to save history even when the terminal is killed (at least in Terminal.app), so it'd be nice to do that.


#    import atexit
#    import signal
#
#    def cleanup(*args):
#        print("clean")
#        readline.write_history_file(u.histfile)
#        sys.exit(0)
#
#    signal.signal(signal.SIGQUIT,cleanup)
#    signal.signal(signal.SIGHUP,cleanup)
#    signal.signal(signal.SIGINT,cleanup)
#    signal.signal(signal.SIGPIPE,cleanup)
#    signal.signal(signal.SIGALRM,cleanup)
#    signal.signal(signal.SIGTERM,cleanup)

    #signal.alarm(1)

    # write history on exiting
    #atexit.register(cleanup)

    try:
        import main
        main.main()
        break  # the end!
    except Exception as e:
        print(u.format_exception(e, u.src_path, verbose=True))
        input(u.mk_g("[Reload with ENTER]"))
        continue
