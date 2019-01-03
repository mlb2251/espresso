# MAIN
import sys
import os
homedir = os.environ['HOME']
sys.path.append(homedir+'/espresso/src/')

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
del os, histfile, rlcompleter

## apparently theres a way to make your own completer!
## could have a function that defaults to the python 'rlcompleter' 


import sys
import os
from importlib import reload

import codegen
import util as u

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
    print(u.mk_yellow(result))

    with open(outfile,'w') as f:
        f.write(result)

    u.green("successfully written to:"+outfile)
    u.green("running...")
    #print(backend.sh('python3 a_out.py '+' '.join(prgm_args)))

else: # REPL mode
    print(u.mk_red("Welcome to the ")+u.mk_bold(u.mk_yellow("Espresso"))+u.mk_red(" Language!"))

    # create a unique temp file to store code.
    # Uses a file named a_out.py in a directory named with a number so that importing works well
    i=0
    master_dir = homedir+'/espresso/repl-tmpfiles/'
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

    import a_out    #initial import so that refresh() can be used in loop
    code.append("backend.disablePrint() # REMOVE THIS FOR SCRIPT") # this MUST go after 'import' or you wont get any output!
    debug=False
    mode='normal'
    # REPL loop
    while True:
        prettycwd = os.getcwd().replace(homedir,'~')
        if mode == 'normal':
            banner = u.mk_green("es:"+prettycwd+" $ ")
        if mode == 'speedy':
            banner = u.mk_purple("es:"+prettycwd+" $ %")
        line = input(banner)
        reload(codegen) # constantly reloads codegen to update with your changes!
        reload(u)
        if len(line.strip()) == 0: continue

        if line.strip() == '%':
            if mode != 'speedy':
                mode = 'speedy'
            else:
                mode = 'normal'
            continue
        # handle metacommands
        if line[0] == '!':
            if line.strip() == '!print':
                print(u.mk_yellow('\n'.join(code)))
            if line.strip() == '!debug':
                debug = not debug
            if line.strip() == '!help':
                u.blue('Currently implemented macros listing:')
                print(u.mk_purple('\n'.join(codegen.macro_argc.keys())))
            if line.strip() == '!%':
                pass
            if line.strip() == '!sh':
                pass
            continue

        # update codeblock
        code.append("backend.enablePrint()")
        if line.strip()[-1] == ':': # start of an indent block
            if mode == 'speedy': u.warn('dropping into normal mode for multiline')
            lines = [line]
            while True:
                line = input(u.mk_green('.'*len(banner)))
                if line.strip() == '': break    # ultra simple logic! No need to keep track of dedents/indents
                lines.append(line)
            code += [codegen.parse(line) for line in lines]
            to_undo = len(lines)
        else:
            if mode == 'speedy': #prepend '%' to every line
                line = line.strip() #strips line + prepends '%'

                line = '%' + line
                toks = line.split(' ')
                # deal with special case transformations
                if toks[0][1:] not in codegen.macro_argc:
                    line = 'sh{'+line[1:]+'}' # the 1: just kills '%'
                    u.warn('macro {} not recognized. Trying sh:\n{}'.format(toks[0][1:],line))
                elif toks[0] in ['%cd','%cat']: # speedy cd autoquotes the $* it's given
                    line = toks[0]+' "'+' '.join(toks[1:])+'"'

                # finally, print the result
                line = '%p_ignoreNone ('+line+')'


            code.append(codegen.parse(line))
            to_undo = 1

        # write to tmpfile
        with open(tmpfile,'w') as f:
            f.write('\n'.join(code))

        # execute tmpfile
        try:
            reload(a_out)
            latest = code[-to_undo:]
            code = code[:-(to_undo+1)] #cutting out the print disabler
            code += latest
            sys.stdout.flush()
        except Exception as e:
            u.red("Error:{}".format(e))
            sys.stdout.flush()
            code = code[:-(to_undo+1)] #erase added code including print disabler



