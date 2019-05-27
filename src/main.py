#!/usr/local/bin/python3
## NOTE THAT THIS IS NOT THE PATH TO PYTHON# ON EVERYONES COMPUTER SO IT PROB WONT WORK FOR MOST PPL SO FIX THAT
# MAIN

import time
_start_time = time.time()

from importlib import reload
import sys
import os
from copy import deepcopy
import repl

import codegen
import util as u

import argparse

import readline
HISTMAX=1000

# clear oldest history if needed
#with open(histfile,'r') as f:
#    text = f.read()
#    if text.count('\n') > HISTMAX-10: # clear it if it's near full. seems to stop at 998 inst of 1000 otherwise
#        listform = text.split('\n')
#        open(histfile,'w').write('\n'.join(listform[:-HISTMAX//2]))
#        del listform
#del text

#try:
#    readline.clear_history()
#    print("cleared")
#    readline.read_history_file(histfile)
#    # default history len is -1 (infinite), which may grow unruly
#    readline.set_history_length(1000)
#except IOError:
#    pass
#import atexit
#
#def cleanup():
#    readline.write_history_file(histfile)


# write history on exiting
#atexit.register(cleanup)
#readline.set_auto_history(True)

## note atexit.register() is a general cleanup fn for when your program randomly exits at any time -- worth looking at more!


def get_arguments():
    parser = argparse.ArgumentParser(description='Espresso programming language')
    parser.add_argument('infile', nargs='?', default=None, type=str, help='The optional input file')
    parser.add_argument('--debug', action='store_true', default=False,
            help='Enables full debugging from startup (equivalent to running !debug as first command)')
    parser.add_argument('--repl', action='store_true', default=False,
            help='Only useful when infile is provided, in which case this forces the parser to drop into a repl after executing the file')
#    parser.add_argument('--stable', action='store_true', default=False,
#            help='Switch global system to use stable espresso and quit')
#    parser.add_argument('--unstable', action='store_true', default=False,
#            help='Switch global system to use unstable espresso and quit')
    return parser.parse_args()


prgm_args = sys.argv[2:]    # often this is []

# initial code to be executed
prelude = [
    "import sys,os",
    "sys.path.append(\""+u.src_path+"\")",
    "import backend",
    "os.chdir(\""+os.getcwd()+"\")",
    "backend.setup_displayhook()",
    "argv = sys.argv",
    "argc = len(argv)",
    ]

# initialize any directories needed
u.init_dirs()


### I havent tested this lately and it should be fixed up at some point.
### For example rn it doesnt pass arguments to the program it compiles
### and it doesnt write it to disk (not sure if that is desirable or not)
# Wouldn't be too hard to fix this
# (you could pass args by inserting them into the prelude somehow)
def do_compile(config):
    #outfile = "a_out.py"

    code = prelude

    with open(infile,'r') as f:
        for line in f.readlines():
            code.append(codegen.parse(line))
    result = '\n'.join(code)
    #print(mk_yellow(result))

    # TODO would be good to just compile to pure python for transferrability, 
    # also you need a way to pass arguments...
    exec(result)

    #green("successfully written to:"+outfile)
    #green("running...")
    #print(backend.sh('python3 a_out.py '+' '.join(prgm_args)))


# This is the important function
def do_repl(config):
    #print(mk_green("Welcome to the ")+mk_bold(mk_yellow("Espresso"))+mk_green(" Language!"))

    # alt. could replace this with a 'communicate' code that tells repl to run its full self.code block
    # initialize the repl
    if config.infile is not None:
        if not os.path.isfile(config.infile):
            os.r(f'file {config.infile} not found')
            exit(1)
    the_repl = repl.Repl(config)

    for line in prelude:
        the_repl.run_code([line])
    the_repl.update_banner()
    print("Boot time: {:.3f}".format(time.time()-_start_time))
    # This is the important core loop!
    u.y(f'PID: {os.getpid()}')
    while True:

        #readline.write_history_file(histfile)
        #print("hist len: {} last item: {}".format(readline.get_current_history_length(), readline.get_history_item(readline.get_current_history_length())))
        try:
            line = the_repl.get_input()
            u.reload_modules(verbose=the_repl.verbose_exceptions)
            #print(the_repl.line_idx)
            #the_repl = repl.Repl(repl=the_repl) # updates repl to a new version. This must be done outside of the Repl itself bc otherwise changes to methods won't be included.
            #print(the_repl.line_idx)
            the_repl.next(line)
        except u.VerbatimExc as e:
            print(e)
            if the_repl.context.compile:
                sys.exit(1)
        except Exception as e:
            #the program will never crash!!! It catches and prints exceptions and then continues in the while loop!
            print(u.format_exception(e,u.src_path,verbose=the_repl.verbose_exceptions))
            if the_repl.context.compile:
                sys.exit(1)


def main():
    try:
        config = get_arguments()
        do_repl(config)
        #if config.infile is None:
        #    do_repl(config)
        #else:
        #    do_compile(config)
    except Exception as e:
        print(u.format_exception(e,u.src_path,verbose=True))

#if __name__ == '__main__':
#    main()

