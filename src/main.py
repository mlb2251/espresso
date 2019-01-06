# MAIN

from importlib import reload
import sys
import os
from copy import deepcopy

sys.path.append(os.environ['HOME']+'/espresso/src')
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
    "sys.path.append(os.environ['HOME']+'/espresso/src/')",
    "import backend",
    "os.chdir(\""+os.getcwd()+"\")",
    #"BACKEND_PIPE = backend.init_sh_backend()",
    ]

# initialize any directories needed
u.init_dirs()

if len(sys.argv) > 1:
    infile = sys.argv[1]
    outfile = "a_out.py"

    code = prelude

    with open(infile,'r') as f:
        for line in f.readlines():
            code.append(codegen.parse(line))
    result = '\n'.join(code)
    print(mk_yellow(result))

    with open(outfile,'w') as f:
        f.write(result)

    green("successfully written to:"+outfile)
    green("running...")
    #print(backend.sh('python3 a_out.py '+' '.join(prgm_args)))

else: # REPL mode
    print(mk_green("Welcome to the ")+mk_bold(mk_yellow("Espresso"))+mk_green(" Language!"))

    master_dir = u.repl_path

    # create a unique temp file to store code.
    # Uses a file named a_out.py in a directory named with a number so that importing works well
    i=0
    tmpdir = master_dir+str(i)
    while os.path.isdir(tmpdir):
        i = i+1
        tmpdir = master_dir+str(i)+'/'
    os.mkdir(tmpdir)
    sys.path.append(tmpdir) # this way "import a_out" will work later
    tmpfile = tmpdir+'a_out.py'

    # Prelude
    code = prelude
    with open(tmpfile,'w') as f:
        f.write('\n'.join(code))
    code.append("backend.disablePrint() # REMOVE THIS FOR SCRIPT") # this MUST go after the f.write() or you wont get any output from REPL


    prelude_plus = [s for s in code]

    import a_out
    initial_vars_a_out = dir(a_out)
    init_state = {
            'master_dir':master_dir,
            'tmpfile':tmpfile,
            'code':code,
            'mode':'normal',
            'debug':False,
            'communicate': [],
            'verbose_exceptions':False,
    }
    state = deepcopy(init_state)
    state['a_out'] = a_out # passing in a module! (deep copy fails on it)

    def handle_communication(state):
        old_communicate = deepcopy(state['communicate'])
        state['communicate'] = []
        for msg in old_communicate:
            if msg == 'delete all tmpfiles for a_out':
                state = deepcopy(init_state)
                state['a_out'] = a_out
                # here's the jank way to delete variables from an imported/reloaded mod
                # note that 'del a_out' would indeed delete the module but upon
                # reloading it all of the variables would return magically
                for v in dir(a_out):
                    if v not in initial_vars_a_out:
                        exec('del a_out.{}'.format(v))
            if msg == 'drop out of repl to reload from main':
                while u.reload_modules(sys.modules,verbose=state['verbose_exceptions']):
                    line = input(mk_green("looping reload from main... \nhit enter to retry.\n only metacommand is '!v' to print unformatted exception\n>>> "))
                    if line.strip() == '!v':
                        state['verbose_exceptions'] = not state['verbose_exceptions']
        return state

    # the main outermost loop
    while True:
        try:
            the_repl = repl.Repl(state) #initialize Repl (new version/
            the_repl.next() # run repl, this will update Repl internal state
            state = the_repl.get_state() #extract state to be fed back in
            state = handle_communication(state)
        except Exception as e:
            print(u.format_exception(e,u.src_path,verbose=state['verbose_exceptions']))
            continue











