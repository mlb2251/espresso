# MAIN

from importlib import reload
import sys
import os
from copy import deepcopy

import codegen
import util as u
from util import die,warn,mk_blue,mk_red,mk_yellow,mk_cyan,mk_bold,mk_gray,mk_green,mk_purple,mk_underline,red,blue,green,yellow,purple,pretty_path
import repl

import readline
import rlcompleter
readline.parse_and_bind("tab: complete")
histfile = os.path.join(u.error_path+'eshist')
try:
    readline.read_history_file(histfile)
    # default history len is -1 (infinite), which may grow unruly
    readline.set_history_length(1000)
except IOError:
    pass
import atexit
atexit.register(readline.write_history_file, histfile)
del histfile, rlcompleter

## apparently theres a way to make your own completer!
## could have a function that defaults to the python 'rlcompleter' 


prgm_args = sys.argv[2:]    # often this is []

prelude = [
    "import sys,os",
    "sys.path.append(\""+u.src_path+"\")",
    "import backend",
    "os.chdir(\""+os.getcwd()+"\")",
    #"BACKEND_PIPE = backend.init_sh_backend()",
    ]
init_state = {
        'globs':dict(),
        'locs':dict(),
        'code':prelude,
        'mode':'normal',
        'banner':'>>> ',
        'banner_uncoloredlen':4,
        'debug':False,
        'communicate': [],
        'verbose_exceptions':False,
}

# initialize any directories needed
u.init_dirs()

def handle_communication(state):
    old_communicate = deepcopy(state.communicate)
    state.communicate = []
    for msg in old_communicate:
        if msg == 'reset state':
            state = ReplState(deepcopy(init_state))
        if msg == 'drop out of repl to reload from main':
            while u.reload_modules(sys.modules,verbose=state.verbose_exceptions):
                line = input(mk_green("looping reload from main... \nhit enter to retry.\n only metacommand is '!v' to print unformatted exception\n>>> "))
                if line.strip() == '!v':
                    state.verbose_exceptions = not state.verbose_exceptions
    return state

def do_compile():
    infile = sys.argv[1]
    outfile = "a_out.py"

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


def start_repl():
    print(mk_green("Welcome to the ")+mk_bold(mk_yellow("Espresso"))+mk_green(" Language!"))

    state = ReplState(deepcopy(init_state))


    # alt. could replace this with a 'communicate' code that tells repl to run its full self.code block
    the_repl = repl.Repl(state)
    the_repl.run_code(prelude)
    the_repl.update_banner()
    state = the_repl.get_state()
    # the main outermost loop
    while True:
        try:
            the_repl = repl.Repl(state) #initialize Repl (new version)
            the_repl.next() # run repl, this will update Repl internal state
            state = the_repl.get_state() #extract state to be fed back in
            state = handle_communication(state)
        except Exception as e:
            print(u.format_exception(e,u.src_path,verbose=state.verbose_exceptions))


# this lives in Main since it might cause probs to have it in Repl because it might prevent being able to reload or something. Not certain, could try some time.
class ReplState:
    def __init__(self,value_dict):
        #self.master_dir=value_dict["master_dir"]
        self.globs=value_dict["globs"]
        self.locs=value_dict["locs"]
        #self.tmpfile=value_dict["tmpfile"]
        self.code=value_dict["code"]
        self.mode=value_dict["mode"]
        self.banner=value_dict["banner"]
        self.banner_uncoloredlen=value_dict["banner_uncoloredlen"]
        self.debug=value_dict["debug"]
        self.communicate=value_dict["communicate"] # messages to/from main.py
        self.verbose_exceptions=value_dict['verbose_exceptions']


try:
    if len(sys.argv) > 1:
        do_compile
    else:
        start_repl()
except Exception as e:
    print(u.format_exception(e,u.src_path))
