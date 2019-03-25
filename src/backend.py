# BACKEND


# This is the file that gets imported by the actual generated python program
# OVERVIEW:
# sh() executes a line of bash code. init_sh_backend() send() and recv() are all related to this.
# anything starting with m_ is a macro. %cat automatically becomes m_cat() etc. 

import sys
import os
import util as u
import subprocess as sp

# this determines what eval() prints
def setup_displayhook():
    def displayhook(v):
        if v is not None:
            print(v)
    sys.displayhook = displayhook


#def disablePrint(): sys.stdout = open(os.devnull, 'w')
#def enablePrint(): sys.stdout = sys.__stdout__


## From old method of sh backend
#def recv(pipe):
#    res = open(pipe).read().strip()
#    #u.red(res)
#    return res
#
#def send(s,pipe):
#    #u.blue("sending:"+s)
#    open(pipe,'w').write('cd '+os.getcwd()+';'+s)
#
# create a new pipe
# popen a backend_sh.sh with the pipe
# also return the pipe's file path
#def init_sh_backend():
#    i=0
#    in_pipe = u.pipe_dir+str(i)+'_in'
#    while os.path.exists(in_pipe):
#        i += 1
#        in_pipe = u.pipe_dir+str(i)+'_in'
#    out_pipe = u.pipe_dir+str(i)+'_out'
#    os.mkfifo(in_pipe)
#    os.mkfifo(out_pipe)
#
#    sp.Popen(['/bin/bash',u.src_path+'backend_sh.sh',in_pipe,out_pipe])
#
#    return in_pipe,out_pipe

# capture_output = True: acts like normal $() in bash
# ie captures stdout and doesn't print it to the screen.
# if you do $(vim) for example in normal bash no output
# is shown on the screen. Hence capture_output = False
# for most interactive programs and True when you want
# to save the output of a (usually non interactive) program
# Note that in bash saving the output of an interactive program
# is generally done by HEREDOCS which you should look into.
# Returns None if capture_output = False
## could do heredocs with the stdin pipe of run() probably and
## make them behave just like bash heredocs (only one big input
## not able to react to stdout or anything)
def sh(s,capture_output=True):
    if len(s) == 0: return ''

    s += '\necho $PWD > {}'.format(u.pwd_file)

    stdout = sp.PIPE if capture_output else None
    #/bin/bash -O expand_aliases -i -c 'ls'
    #res = sp.run(['/bin/bash',u.src_path+'backend.sh',s],stdout=stdout)
    try:
        res = sp.run(['/bin/bash','-O','expand_aliases','-O','checkwinsize','-l','-c',s],stdout=stdout)
    except KeyboardInterrupt: # catch child interrupts
        return ''
    new_dir = open(u.pwd_file).read().strip()
    os.remove(u.pwd_file)
    os.chdir(new_dir)
    if not capture_output:
        return
    text =  res.stdout.decode("utf-8")
    if text == '': return text
    if text[-1] == '\n': text = text[:-1]
    return text


    #mode = 'capture' if capture_output else 'nocapture'
    #stdout = open(stdout_file).read()


    #send(s,in_pipe)
    #res = recv(out_pipe)

    #s = s.replace('echo','/bin/echo') #there is prob a better way...
    #res = sp.run(s,shell=True,stdout=sp.PIPE,stderr=sp.PIPE)


#print(sh('echo -n test'))

#print(sh('for i in `ls`\n do \necho hi $i \n done'))
#sp.run(shlex.split('echo -n hi'),cwd='.',shell=False,stdout=sp.PIPE,stderr=sp.PIPE)    # this does work but then it cant do for loops etc. better to stick with the other solution of replacing echo with /bin/echo.

# do not do \n -> ; conversion bc this fails if 'if' and 'for' loops bc 'do' and 'then' cant be followed by semis. who knows what other ones exist. And after all \n works just fine!


# takes an input string (could be the result of an sh{} or py expr or paren block or whatever --
# at this point it just looks like an 'str' to python in this final generated code
# and takes a list of (idx,cast) tuples
def m_parse(oneline_s,idx_cast_list,delim=' ',try_parselines=True):
    # promote to m_parselines if multiple \n or just 1 \n but it's not the last character of the input
    # can be forced off with try_parselines=False
    if try_parselines:
        if oneline_s.count('\n') > 1 or (oneline_s.count('\n') == 1 and oneline_s[-1] != '\n'):
            return m_parselines(oneline_s,idx_cast_list,delim)

    if oneline_s[-1] == '\n':
        oneline_s = oneline_s[:-1]

    tokens = oneline_s.split(delim)
    tokens = list(filter(None,tokens))
    result = []
    for (idx,cast) in idx_cast_list:
        idx = int(idx)
        cast = eval(cast)
        try:
            result.append(cast(tokens[idx-1]))
        except IndexError:
            print(u.mk_yellow("Warning[line parse failed: out of bounds].\n rule:(idx={},cast={})\n line:{}\n note that (idx-1) is what's actually used".format(idx,cast.__name__,oneline_s)))
            return None
        except ValueError:
            print(u.mk_yellow("Warning[line parse failed: cast failure].\n rule:(idx={},cast={}) => {}({}) failed".format(idx,cast.__name__,cast.__name__,tokens[idx-1])))
            return None

    return tuple(result)

# figures out whether you want to undo parselines or just parse
def m_unparse(input,delim=' '):
    pass
    # input can be a:
        # list of list of tuples from parselines1
        # tuple of list of tuples from parselines
        # list of 


def m_parselines1(multiline_s,idx_cast_list,delim=' '):
    lines = list(filter(None,multiline_s.split('\n')))  # throw out empty lines
    result = list(filter(None,[m_parse(line,idx_cast_list,delim=delim) for line in lines]))
    return result

# trick to turn single list of many tuples into single tuple of many lists
def m_parselines(multiline_s,idx_cast_list,delim=' '):
    result = tuple(zip(*(m_parselines1(multiline_s,idx_cast_list,delim=delim))))
    return tuple([list(x) for x in result]) #convert inner tuples into lists

def m_argv():
    return sys.argv

def m_argc():
    return len(sys.argv)

def m_cat(filename):
    with open(filename,'r') as f:
        text = f.read()
    return text

def m_exists_qmark(the_path):
    return os.path.isfile(the_path)

def m_direxists_qmark(the_path):
    return os.path.isdir(the_path)

# in bytes
def m_fsize(the_path):
    return os.path.getsize(the_path)

def m_fullpath(the_path):
    return os.path.abspath(the_path)

def m_basename(the_path):
    return os.path.basename(the_path)

def m_die(s):
    u.red("Error:"+str(s))
    exit(1)

def m_head(filename,nlines=10):
    with open(filename,'r') as f:
        hd = ''.join([next(f) for x in range(nlines)])
    return hd

def m_p(s):
    print(str(s))


# not used for now
def b_p_ignoreNone(s):
    if s is None:
        return
    if isinstance(s,list):
        m_p('   '.join(s))
        return
    m_p(s)



def m_blue(s):
    return u.mk_blue(s)
def m_red(s):
    return u.mk_red(s)
def m_green(s):
    return u.mk_green(s)
def m_yellow(s):
    return u.mk_yellow(s)
def m_purple(s):
    return u.mk_purple(s)

def m_ls(dir = '.',regex=None,show_hidden=False):
    dir = os.getcwd()+'/'+dir
    if regex is None:
        listing = os.listdir(dir)
    else:
        listing = list(filter(lambda s: re.match(regex,s) is not None, os.listdir(dir)))
    listing.sort()
    if show_hidden == False:
        listing = list(filter(lambda x: x[0]!='.', listing))
    return listing

def m_mkdir(the_path):
    os.mkdir(the_path)
def m_mkdirp(the_path):
    os.makedirs(the_path)

def m_cd(the_path):
    #print('old:',os.getcwd())
    os.chdir(the_path)
    #print('new:',os.getcwd())
def m_pwd():
    return os.getcwd()

