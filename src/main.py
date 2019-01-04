# MAIN

from importlib import reload
import sys
import os

sys.path.append(os.environ['HOME']+'/espresso/src')
import codegen
import util as u
from util import die,warn,mk_blue,mk_red,mk_yellow,mk_cyan,mk_bold,mk_gray,mk_green,mk_purple,mk_underline,red,blue,green,yellow,purple,pretty_path
import repl




import readline
import rlcompleter
readline.parse_and_bind("tab: complete")
histfile = os.path.join(os.path.expanduser("~"), ".eshist")
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
    "import sys", "import os", "sys.path.append(os.environ['HOME']+'/espresso/src/')", "import backend", "os.chdir(\""+os.getcwd()+"\")",
    ]

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
    print(mk_red("Welcome to the ")+mk_bold(mk_yellow("Espresso"))+mk_red(" Language!"))

    #initialize directory for holding generated code
    master_dir = u.homedir+'/.espresso/repl-tmpfiles/'
    if not os.path.isdir(master_dir):
        os.makedirs(master_dir)

    # create a unique temp file to store code.
    # Uses a file named a_out.py in a directory named with a number so that importing works well
    i=0
    tmpdir = master_dir+str(i)
    while os.path.isdir(tmpdir):
        i = i+1
        tmpdir = master_dir+str(i)
    os.mkdir(tmpdir)
    sys.path.append(tmpdir) # this way "import a_out" will work later
    tmpfile = tmpdir+'/a_out.py'

    # Prelude
    code = prelude
    with open(tmpfile,'w') as f:
        f.write('\n'.join(code))
    code.append("backend.disablePrint() # REMOVE THIS FOR SCRIPT") # this MUST go after the f.write() or you wont get any output from REPL

    debug=False
    mode='normal'
    import a_out
    state = [master_dir,tmpfile,code,mode,debug,a_out] # even passes the module a_out in!
    retry=False
    while True:
        if retry:
            input(green(">>> Hit enter to try again..."))
            retry = False
        try:
            reload(repl)
            reload(u)
            reload(codegen)
            the_repl = repl.Repl(state) #initialize Repl (new version)
            the_repl.next() # run repl, this will update Repl internal state
            state = the_repl.get_state() #extract state to be fed back in at new Repl init
        except Exception as e:
            print(u.format_exception(e,u.src_path)[0])
            retry=True
            continue












