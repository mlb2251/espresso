import sys
import os
from importlib import reload

import codegen
import util as u

prgm_args = sys.argv[2:]    # often this is []

if len(sys.argv) > 1:
    infile = sys.argv[1]
    outfile = "a_out.py"

    code = "import sys\n"
    code += "import os\n"
    code += "sys.path.append(os.environ['HOME']+'/espresso/src/')\n"
    code += "import backend\n"

    with open(infile,'r') as f:
        for line in f.readlines():
            code += codegen.parse(line)
    print(u.mk_yellow(code))

    with open(outfile,'w') as f:
        f.write(code)

    u.green("successfully written to:"+outfile)
    u.green("running...")
    #print(backend.sh('python3 a_out.py '+' '.join(prgm_args)))

else: # REPL mode
    print(u.mk_red("Welcome to the ")+u.mk_bold(u.mk_yellow("Espresso"))+u.mk_red(" Language!"))

    # create a unique temp file to store code.
    # Uses a file named a_out.py in a directory named with a number so that importing works well
    i=0
    master_dir = os.environ['HOME']+'/espresso/repl-tmpfiles/'
    tmpdir = master_dir+str(i)
    while os.path.isdir(tmpdir):
        i = i+1
        tmpdir = master_dir+str(i)
    os.mkdir(tmpdir)
    sys.path.append(tmpdir) # this way "import a_out" will work later
    tmpfile = tmpdir+'/a_out.py'

    # Prelude
    code = ["import sys"]
    code.append("import os")
    code.append("sys.path.append(os.environ['HOME']+'/espresso/src/')")
    code.append("import backend")
    with open(tmpfile,'w') as f:
        f.write('\n'.join(code))

    debug=False
    import a_out
    code.append("backend.disablePrint() # REMOVE THIS FOR SCRIPT") # disable print() attempts up until next enablePrint()
    # REPL loop
    while True:
        line = input(u.mk_green(">>> "))
        reload(codegen) # constantly reloads codegen to update with your changes!
        reload(u)
        if len(line.strip()) == 0: continue

        # handle metacommands
        if line[0] == '!':
            if line.strip() == '!print':
                print(u.mk_yellow('\n'.join(code)))
            if line.strip() == '!debug':
                debug = not debug
            continue

        # update codeblock
        code.append("backend.enablePrint()")
        if line.strip()[-1] == ':': # start of an indent block
            lines = [line]
            while True:
                line = input(u.mk_green("... "))
                if line.strip() == '': break    # ultra simple logic! No need to keep track of dedents/indents
                lines.append(line)
            code += [codegen.parse(line) for line in lines]
            to_undo = len(lines)
        else:
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



