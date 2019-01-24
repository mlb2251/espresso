# UTIL
from importlib import reload

import subprocess as sp
import os
homedir = os.environ['HOME']
src_path = os.path.dirname(os.path.realpath(__file__))+'/'
#src_path = homedir+'/espresso/src/'
data_path = homedir+'/.espresso/'
error_path = data_path+'error_handling/'
#repl_path = data_path+'repl-tmpfiles/'
#pipe_dir = data_path+'pipes/'
pwd_file = data_path+'.{}.PWD'.format(os.getpid())

# initialize dirs used by espresso
def init_dirs():
    dirs = [src_path,data_path,error_path]
    for d in dirs:
        if not os.path.isdir(d):
            blue('created:'+d)
            os.makedirs(d)

# not used right now
def clear_repl_tmpfiles():
    import shutil
    shutil.rmtree(repl_path)
    os.makedirs(repl_path)
    del shutil

def die(s):
    raise VerbatimExc(mk_red("Error:"+str(s)))

def warn(s):
    print(mk_yellow("WARN:"+s))

# cause nobody likes ugly paths
def pretty_path(p):
    return p.replace(homedir,'~')

# returns ['util','repl','main',...] These are the names of the source modules.
# this is used in reload_modules()
def module_ls():
    files = os.listdir(src_path)
    files = list(filter(lambda x:x[-3:]=='.py',files)) # only py files
    mod_names = [x[:-3] for x in files] # cut off extension
    return mod_names

# takes the result of sys.modules as an argument
# 'verbose' will cause the unformatted exception to be output as well
# TODO passing in sys.modules may be unnecessary bc util.py may share the same sys.modules
# as everything else. Worth checking.
def reload_modules(mods_dict,verbose=False):
    failed_mods = []
    for mod in module_ls():
        if mod in mods_dict:
            try:
                reload(mods_dict[mod])
                #blue('reloaded '+mod)
            except Exception as e:
                failed_mods.append(mod)
                print(format_exception(e,src_path,ignore_outermost=1,verbose=verbose))
                pass
    return failed_mods # this is TRUTHY if any failed


class PrettifyErr(Exception): pass
class VerbatimExc(Exception): pass

import traceback as tb
def exception_str(e):
    return ''.join(tb.format_exception(e.__class__,e,e.__traceback__))

# Takes a shitty looking normal python exception and beautifies it.
# also writes to .espresso/error_handling keeping track of the files/line numbers where the errors happened.
# (which you can automatically open + jump to the correct line using a script like my e() bash function
# included in a comment near the bottom of this file)
# relevant_path_piece is just used to remove a lot of irrelevant parts of exceptions.
# for example it might be '/espresso/src/' to ignore all lines of traceback that dont contain that string
# "verbose" option will print out the whole original exception in blue followed by
# the formatted one.
# ignore_outermost=1 will throw away the first of (fmt,cmd) pair that the program generates, ie the first result of prettify_tb() in the list of results
# given_text = True if you pass in the actual python exception STRING to as 'e'
# TODO replace tmpfile with an optional alias list
def format_exception(e,relevant_path_piece,tmpfile=None,verbose=False,given_text=False,ignore_outermost=0):
    if verbose:
        if given_text:
            blue(''.join(e))
        else:
            blue(exception_str(e))
    try:
        if given_text:
            raw_tb = e
        else:
            raw_tb = tb.format_exception(e.__class__,e,e.__traceback__)

        assert(raw_tb[0] == 'Traceback (most recent call last):\n')
        raw_tb = raw_tb[1:-1]   #rm first+last. last is a copy of str(e)
        #blue(''.join(raw_tb))
        #magenta(raw_tb)
        formatted = raw_tb

        # any lines not starting with 'File' are appended to the last seen
        # line that started with 'File'
        # (this is for standardization bc the exceptions arent that standardized)
        # all this should really be done with whatever python under the hood
        # is generating exceptions haha
        for (i,s) in enumerate(formatted):
            if s.strip()[:4] == 'File':
                lastfile = i
            else:
                formatted[lastfile] += s
        # now remove those leftover copies that dont start with File
        formatted = list(filter(lambda s: s.strip()[:4] == 'File',formatted))

        # delete everything before the first line that relates to the tmpfile
        #for (i,s) in enumerate(raw_tb):
        #    if relevant_path_piece in s:
        #        formatted = raw_tb[i:]
        #        break
        if isinstance(relevant_path_piece,str):
            formatted = list(filter(lambda s: relevant_path_piece in s, formatted))
        elif isinstance(relevant_path_piece,list):
            def aux(haystack,needles): #true if haystack contains at least one needle
                for n in needles:
                    if n in haystack:
                        return True
                return False
            formatted = list(filter(lambda s: aux(s,relevant_path_piece), formatted))


        # Turns an ugly traceback segment into a nice one
        # for a traceback segment that looks like (second line after \n is optional, only shows up sometimes):
        #File "/Users/matthewbowers/espresso/repl-tmpfiles/113/a_out.py", line 8, in <module>\n      test = 191923j2k9E # syntax error
        def try_pretty_tb(s):
            try:
                return pretty_tb(s)
            except Exception as e:
                warn("(ignorable) Error during prettifying traceback component.\ncomponent={}\nreason={}".format(s,exception_str(e)))
                return [s,'']
        def pretty_tb(s):
            if s[-1] == '\n': s = s[:-1]
            if s.count('\n') == 0:
                includes_code = includes_arrow = False
                msg = s
            elif s.count('\n') == 1:
                includes_code=True
                includes_arrow=False
                [msg,code_line] = s.split('\n')
            elif s.count('\n') == 2:
                includes_code = includes_arrow = True
                [msg,code_line,arrow] = s.split('\n')
            elif s.count('\n') > 2:
                raise PrettifyErr('more than 3 lines in a single traceback component:{}')

            if msg.count(',') == 1:
                includes_fn = False
                [fpath,lineno] = msg.split(',')
            elif msg.count(',') == 2:
                includes_fn = True
                [fpath,lineno,fn_name] = msg.split(',')
            else:
                raise PrettifyErr('unexpected number of commas in traceback component line:{}'.format(s))


            # prettify file name (see example text above function)
            assert(len(fpath.split('"'))==3)
            fpath = fpath.split('"')[1] # extract actual file name
            fpath_abs = os.path.abspath(fpath)
            if fpath == tmpfile:
                fname = "tmpfile"
            else:
                fname = os.path.basename(fpath)
            fpath = fpath.replace(homedir,'~')

            # prettify line number (see example text above function)
            lineno = lineno.strip()
            assert(len(lineno.split(' '))==2)
            lineno_white = lineno.split(' ')[1]
            lineno = lineno.split(' ')[1]

            # prettify fn name
            if includes_fn:
                fn_name = fn_name.strip()
                assert(len(fn_name.split(' '))==2)
                fn_name = fn_name.split(' ')[1]
                if fn_name == '<fnule>':
                    fn_name = ''
                else:
                    fn_name = fn_name+'()'
            else:
                fn_name = ''

            # build final result
            command = "+{} {}".format(lineno,fpath_abs)
            result = "{} {} {}".format(
                    mk_underline(mk_red(fname+" @ "+lineno))+mk_red(':'),
                    mk_gray('('+fpath+')'),
                    mk_purple('['+str(try_pretty_tb.cmdcount)+']'),
                    mk_green(fn_name),
                    )
            if includes_code:
                code_num_spaces = code_line.index(code_line.strip()[0])
                code_line = code_line.strip()
                lineno_fmtd = '{:>6}: '.format(lineno)
                lineno_width = len(lineno_fmtd)
                result += '\n{}{}'.format(
                        mk_bold(mk_green(lineno_fmtd)),
                        mk_yellow(code_line)
                        )
            if includes_arrow:
                arrow_num_spaces = arrow.index(arrow.strip()[0])
                offset = lineno_width + arrow_num_spaces - code_num_spaces
                result += '\n{}{}'.format(' '*offset,mk_bold(mk_green('^here')))
            try_pretty_tb.cmdcount += 1
            return (result,command)

        try_pretty_tb.cmdcount = 0
        res = [try_pretty_tb(s) for s in formatted]
        res = res[ignore_outermost:]
        (formatted,commands) = zip(*res)
        with open(error_path+'/vim_cmds','w') as f:
            commands = list(filter(None,commands))
            f.write('\n'.join(commands))

        if given_text:
            # e[-2][:-1] is the exception str eg 'NameError: name 'foo' is not defined'
            # silly ['']s are just to add extra newlines
            formatted = [''] + [mk_red(e[-2][:-1])] + list(formatted) + ['']
        else:
            formatted = [''] + [mk_red(e)] + list(formatted) + ['']

        return '\n'.join(formatted)
    except Exception as e2:
        warn("(ignorable) Failed to Prettify exception, using default format. Note that the prettifying failure was due to: {}".format(exception_str(e2)))
        if given_text:
            return (mk_red(e),[])
        else:
            return (mk_red(exception_str(e)),[])




from colorama import init
init()
from colorama import Fore, Back, Style
# Fore: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
# Back: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
# Style: DIM, NORMAL, BRIGHT, RESET_ALL.

# \001 and \002 are used by readline to fix buggy prompt displaying when
# browsing history (https://stackoverflow.com/questions/9468435/look-how-to-fix-column-calculation-in-python-readline-if-use-color-prompt)
def color(s,c):
    return '\001'+c+'\002'+str(s)+'\001'+Style.RESET_ALL+'\002'

def mk_underline(s):
    return color(s,'\033[4m')
def mk_green(s):
    return color(s,Fore.GREEN)
def mk_red(s):
    return color(s,Fore.RED)
def mk_purple(s):
    return color(s,Fore.MAGENTA)
def mk_blue(s):
    return color(s,Fore.BLUE)
def mk_cyan(s):
    return color(s,Fore.CYAN)
def mk_yellow(s):
    return color(s,Fore.YELLOW)
def mk_bold(s):
    return color(s,'\033[1m')
def mk_gray(s):
    return color(s,'\033[90m')


def pc(color,msg):
    print(color+str(msg)+cols.ENDC)
def red(msg):
    pc(cols.BOLD+cols.FAIL,msg)
def red_thin(msg):
    pc(cols.FAIL,msg)
def purple(msg):
    pc(cols.MAGENTA,msg)
def blue(msg):
    pc(cols.OKBLUE,msg)
def yellow(msg):
    pc(cols.YELLOW,msg)
def green(msg):
    pc(cols.OKGREEN+cols.BOLD,msg)


# you can actually run this file and use it to format any exception you get.
# I have it running on my main python by having the following in my
# ~/.bash_profile
#p3(){
#    if [ $# -eq 0 ]; then
#        python3
#    else
#        python3 $* 2> ~/.py_err
#        python3 ~/espresso/src/util.py $1 ~/.py_err
#    fi
#}
#v3(){
#    echo "verbose" > ~/.py_err
#    python3 $* 2>> ~/.py_err
#    python3 ~/espresso/src/util.py $1 ~/.py_err
#}
## opens file based on python error idx passed to it
## defaults to last error
#e(){
#    if [ $# -eq 0 ]; then
#        line=$(tail -n1 ~/.espresso/error_handling/vim_cmds)
#    else
#        num=$(( $1 + 1  ))
#        line=$(sed -n -e "$num"p ~/.espresso/error_handling/vim_cmds)
#    fi
#    nvim $line
#}

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        red("call like: p3 "+sys.argv[0]+" /path/to/file.py /path/to/stderr/file")
    if len(sys.argv) == 5: #optionally pass a relevant path piece. otherwise it'll assume no relevant piece (often this is ok bc exceptions dont always contain full paths anyways
        relevant_path_piece = argv[3]
    file = sys.argv[1]
    #relevant_path_piece = os.environ['HOME']
    relevant_path_piece = ''
    stderr = sys.argv[2]
    with open(stderr,'r') as f:
        exception = f.read().split('\n')
    if exception == ['']:
        exit(0)
    verbose=False
    if exception[0] == 'verbose':
        verbose=True
        exception = exception[1:]
    exception = [line+'\n' for line in exception]
    fmtd = format_exception(exception,relevant_path_piece,given_text=True,verbose=verbose)
    print(fmtd)



