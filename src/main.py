#!/usr/local/bin/python3
## NOTE THAT THIS IS NOT THE PATH TO PYTHON# ON EVERYONES COMPUTER SO IT PROB WONT WORK FOR MOST PPL SO FIX THAT SHIT
# MAIN

from importlib import reload
import sys
import os
from copy import deepcopy

import codegen
import util as u
import repl

import readline
histfile = os.path.join(u.error_path+'eshist')
HISTMAX=1000

# clear oldest history if needed
with open(histfile,'r') as f:
    text = f.read()
    if text.count('\n') > HISTMAX:
        listform = text.split('\n')
        open(histfile,'w').write('\n'.join(listform[-HISTMAX//2:]))
        del listform
del text

try:
    readline.read_history_file(histfile)
    # default history len is -1 (infinite), which may grow unruly
    readline.set_history_length(-1)
except IOError:
    pass
import atexit

# write history on exiting
atexit.register(readline.write_history_file, histfile)

## note atexit.register() is a general cleanup fn for when your program randomly exits at any time -- worth looking at more!

del histfile


prgm_args = sys.argv[2:]    # often this is []

# initial code to be executed
prelude = [
    "import sys,os",
    "sys.path.append(\""+u.src_path+"\")",
    "import backend",
    "os.chdir(\""+os.getcwd()+"\")",
    "backend.setup_displayhook()",
    ]
# initial state of the REPL
init_state = {
        'globs':dict(), #TODO this should prob actually be set to whatever pythons initial globals() list is
        #'locs':dict(),
        'code':prelude,
        'mode':'speedy',
        'banner':'>>> ',
        'banner_uncoloredlen':4,
        'banner_cwd':'',
        'debug':False,
        'communicate': [],
        'verbose_exceptions':False,
}

# initialize any directories needed
u.init_dirs()

# updates the state based on communications sent through the list ReplState.communication
def handle_communication(state):
    old_communicate = deepcopy(state.communicate)
    state.communicate = []
    for msg in old_communicate:
        if msg == 'reset state':
            state = ReplState(deepcopy(init_state))
        if msg == 'drop out of repl to reload from main':
            while u.reload_modules(sys.modules,verbose=state.verbose_exceptions):
                line = input(u.mk_g("looping reload from main... \nhit enter to retry.\n only metacommand is '!v' to print unformatted exception\n>>> "))
                if line.strip() == '!v':
                    state.verbose_exceptions = not state.verbose_exceptions
    return state

### I havent tested this lately and it should be fixed up at some point.
### For example rn it doesnt pass arguments to the program it compiles
### and it doesnt write it to disk (not sure if that is desirable or not)
# Wouldn't be too hard to fix this
# (you could pass args by inserting them into the prelude somehow)
def do_compile():
    infile = sys.argv[1]
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
def start_repl():
    #print(mk_green("Welcome to the ")+mk_bold(mk_yellow("Espresso"))+mk_green(" Language!"))

    state = ReplState(deepcopy(init_state))


    # alt. could replace this with a 'communicate' code that tells repl to run its full self.code block
    # initialize the repl
    the_repl = repl.Repl(state)
    for line in prelude:
        the_repl.run_code([line])
    the_repl.update_banner()
    state = the_repl.get_state()
    # This is the important core loop!
    while True:
        try:
            the_repl = repl.Repl(state) #initialize Repl (new version)
            the_repl.next() # run repl, this will update Repl internal state
            state = the_repl.get_state() #extract state to be fed back in
            state = handle_communication(state)
        except u.VerbatimExc as e:
            print(e)
        except Exception as e:
            #the program will never crash!!! It catches and prints exceptions and then continues in the while loop!
            print(u.format_exception(e,u.src_path,verbose=state.verbose_exceptions))


# ReplState keeps track of the important state variables of the REPL
# Each time a repl.Repl is run on a new line, it returns a main.ReplState to main.py
# And main.py uses this state to generate a new repl.Repl. The reason for this silliness
# is so that we generate a new instance of repl.Repl on every step which is important
# because repl.py gets reload()ed all the time to account for source code changes.
# This lives in Main since it might cause probs to have it in Repl because it might prevent being able to reload or something. Not certain, could try some time.
class ReplState:
    def __init__(self,value_dict):
        self.globs=value_dict["globs"]  # globals dict used by exec()
        #self.locs=value_dict["locs"]    # locals dict used by exec()
        self.code=value_dict["code"]    # a list containing all the generated python code so far. Each successful line the REPL runs is added to this
        self.mode=value_dict["mode"]    # 'normal' or 'speedy'
        self.banner=value_dict["banner"] #the banner is that thing that looks like '>>> ' in the python interpreter for example
        self.banner_uncoloredlen=value_dict["banner_uncoloredlen"] # the length of the banner, ignoring the meta chars used to color it
        self.banner_cwd=value_dict["banner_cwd"] # the length of the banner, ignoring the meta chars used to color it
        self.debug=value_dict["debug"]  #if this is True then parser output is generated. You can toggle it with '!debug'
        self.communicate=value_dict["communicate"] # for passing messages between main.py and repl.py, for things like hard resets and stuff
        self.verbose_exceptions=value_dict['verbose_exceptions'] #if this is true then full raw exceptions are printed in addition to the formatted ones


try:
    if len(sys.argv) > 1:
        do_compile()
    else:
        start_repl()
except Exception as e:
    print(u.format_exception(e,u.src_path))
