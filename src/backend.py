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


"""
capture_output = T/F
capture_error = T/F
capture_retcode = T/F
sh() will return a tuple of length 0 to 3 depending on which of these are True.
If all are true then (stdout,stderr,retcode) are returned
If none are true then None is returned
If capture_stdout=True then just stdout is returned.
If capture_stdout=True and capture_retcode=True then (stdout,retcode) is returned
etc.

Q: should sh run .split()? should it trip trailing newlines? If it splits and ends up with a single line should it return it verbatim?

-often times we want it to be .split() right away
-extremely often we want a \n stripped
-occasionally we want to remove all empty lines

>result: remove \n automatically until `raw` specified, but leave it to the user to .split it? Hmm

"""


def sh(s, capture_output=True, capture_error=False, exception_on_retcode=None):
    debug = u.Debug(False)

    if exception_on_retcode is None: # set default
        if capture_output or capture_error:
            exception_on_retcode = True
        else:
            exception_on_retcode = False

    if s == '': # Make a fake completedprocess that looks fine, since the '' program should always return empty strings and success.
        return SHVal(CompletedProcess(args='',
                returncode=0,
                stdout=('' if capture_output else None),
                stderr=('' if capture_error else None)))

    s += f'\necho $?\001$PWD > {u.pwd_file}' # Note that this will only change the directory if the whole script finishes

    stdout = sp.PIPE if capture_output else None
    stderr = sp.PIPE if capture_error else None
    #/bin/bash -O expand_aliases -i -c 'ls'
    #res = sp.run(['/bin/bash',u.src_path+'backend.sh',s],stdout=stdout)
    try:
        #res = sp.run(['/bin/bash','-O','expand_aliases','-O','checkwinsize','-l','-c',s],stdout=stdout)
        res = esrun(['/bin/bash','-O','expand_aliases','-O','checkwinsize','-l','-c',s],
                stdout=stdout,
                stderr=stderr,
                debug=debug,
                text=True) # replace \r\n with \n, replace \r with \n, decode with utf-8. Basically as long as output is text and not arbitrary binary data then this should be used.
    except KeyboardInterrupt: # should not happen, tho with race conditions it may
        raise ValueError(f"esrun() was interrupted at a bad time and was unable to recover. Command may or may not have executed, but stdout, stderr, and error code were unable to be recovered.")
        #if capture_output or capture_error:
            # if they requested the output and we can't give it to them then their logic is at risk, so we shouldn't return an empty string instead we should raise an error.
        #    raise ValueError(f"output of sh() was requested via capture_* but unable to provide it because of ctrl-c during sh setup or teardown")
        #return SHVal(None, exception_on_retcode)

    try:
        with open(u.pwd_file,'r') as f:
            returncode, new_dir = f.read().strip().split('\001')
        res.returncode = int(returncode) # without this the completeprocess return code is always just 0 since the /bin/bash process succeeds even tho the child (which is our actual process) did not succeed
        os.remove(u.pwd_file)
        os.chdir(new_dir)
    except OSError:
        pass # common case if pwd_file is not created bc the sh script terminated early. This is fine.

    ret = SHVal(res, exception_on_retcode)
    global _prev_shval
    _prev_shval = ret
    return ret


class SHVal:
    def __init__(self,completed_process, exception_on_retcode):
        #if completed_process is None and exception_on_retcode:
        #    raise ValueError(f"nonzero return code: {retcode} and `exception_on_retcode` was specified")

        self.retcode = completed_process.returncode
        # these will end up being None if it isnt captured and ret() will remove them as well
        self.out = completed_process.stdout
        self.err = completed_process.stderr

        if exception_on_retcode and self.retcode != 0:
            raise ValueError(f"nonzero return code: {self.retcode} and `exception_on_retcode` was specified")

    def line(self,*casters,rest=None): # coerce into a single line if possible by throwin out empty lines, and return that one line
        if len(casters) == 1 and not callable(casters[0]):
            casters = casters[0] # thus people can pass in either .line(int,int) or .line([int,int]). Note checking if __iter__ is present is bad bc apparently the type "type" has an __iter__ method, so instead we check callability. Also isinstance() can't be used with typeclass "type"
        ret = list(filter(None,self.out.split('\n')))
        if len(ret) != 1:
            raise ValueError(f".line() was called with multiple lines of output:{self.out}")
        if len(casters) == 0:
            return ret[0] # no casting case
        ret = ret[0].split(' ')
        if len(casters) == 1 and rest is None:
            return [casters[0](tkn) for tkn in ret]
        # len(caster) > 1
        assert len(casters) <= len(ret)
        if rest is not None: # `rest` argument autofills the rest of the casters with the given cast
            casters = list(casters) + [rest]*(len(ret)-len(casters))
        assert len(casters) == len(ret), f"number of casters ({len(casters)}) does not match number of tokens ({len(ret)}) \ntokens:{ret}\ncasters:{casters}"
        ret = [casters[i](ret[i]) for i in range(len(ret))]
        return ret

    def item(self, caster=str): # coerce into single word
        ret = self.out.strip()
        if ret.count(' ') == ret.count('\n') == 0:
            return caster(ret)
        raise ValueError(f".item() was called with more than just a single word of output:{self.out}")

    """
    Note that if `length` `casters` or `rest` are specified then lines are returned as a list of lists (lines outer list, tokens inner list)

    Of the following 2 things, it's unclear which is better. The second is much easier to implment. The first could potentially be done by writing a class that inherits from `list` and just add extra methods to it. First is more linear to write perhaps, tho maybe the second way is essentially just as linear. First is more composable ofc. Remember at a certain point you'll want to leave the SshVal world anyways and just manipulate lists and stuff and use all the features of python, and piping will help with that. Just nice to have these convenience methods for the first steps in SshVal.

    1: sh{whatever}.lines().length(4).filter(lambda s: 'example' in s).cast(str,rest=int)
    2: sh{whatever}.lines(str,rest=int,length=4,filter=lambda s: 'example' in s)
    """
    def lines(self,*casters, length=None, rest=None): # return a list of lines
        if rest is not None and length is None:
            raise ValueError("for .lines() `rest` can only be used if `length` is also used so it's clear what the proper length of the line should be")
        if len(casters) == 1 and not callable(casters[0]):
            casters = casters[0] # for if first argument of `casters` is a list of casters

        lines = self.out.split('\n')
        if len(casters) == 0 and length is None:
            return lines # argumentless .lines() call

        lines = [line.split(' ') for line in lines] # tokenize

        # is no length is given but more than one caster is given, then the length is the number of casters
        # (note that `rest` must be None since `length` is None)
        if length is None and len(casters) > 1:
            length = len(casters)

        if length is not None: # kill all lines of wrong lengths
            lines = list(filter(lambda toks: len(toks) == length, lines))

        res = []

        # single caster case with no length restrictions
        if length is None and len(casters) == 1:
            for line in lines:
                try:
                    res.append([casters[0](tkn) for tkn in line])
                except ValueError:
                    u.y(f'cast failure warning: ignoring line {"".join(line)} with caster {caster[0]}. This may be intentional')
            return res

        # generic case (any number of casters, potentially `rest`, and definitely `length`)
        if rest is not None: # `rest` argument autofills the rest of the casters with the given cast
            casters = list(casters) + [rest]*(length-len(casters))

        assert length == len(casters), f"{length} != {len(casters)}"
        for line in lines:
            # apply the ith caster to the ith token and append the resulting list to `res`.
            try:
                res.append([caster[i](line[i]) for i in range(length)])
            except ValueError:
                u.y(f'cast failure warning: ignoring line {"".join(line)} with caster {caster}. This may be intentional')
        return res

    # map `fn` over all lines of output
    def apply(self,fn):
        ret = list(map(fn, self.lines()))
    # filter to only include lines that `fn` returns True on
    # Note that the default value of None will cause all empty lines to be filtered out

    def filter(self,fn=None):
        ret = list(filter(fn, self.lines()))
        return ret
    # filter but `fn` takes a list of tokens for each line
    def filtertokens(self,fn): # run given fn on each line after splitting the line into tokens by spaces
        ret = list(filter(fn, [line.split(' ') for line in self.lines()]))

    """
    To tokenize your string simply do .line(str) or .lines(str) which will space-separate the strings then do the identity-cast to strings
    """
    def empty(self): # bool, True if stdout with all whitespace stripped is empty
        return (self.out.strip() == '')

    def raw(self):
        return self.out
    def err(self):
        return self.err
    def code(self):
        return self.retcode

    def __len__(self):
        return len(self.out.split('\n'))
    def __repr__(self): # TODO make your custom display hook not print SHVals if they have capture_out/err both as None, eg for full line commands.
        res = 'SHVal('
        if self.out is not None:
            res += f"out={self.out},"
        if self.err is not None:
            res += f"err={self.err},"
        res += f"code={self.retcode},"
        res = res[:-1] # kill the last comma
        return res + ')'


from subprocess import Popen,CompletedProcess
_prev_shval = SHVal(CompletedProcess(args='',
                returncode=0,
                stdout=None,
                stderr=None),False)

# Note that blank lines and specific ctrl-c related failures (very rare ones only) will not set this previous return code tracker
def retcode():
    global _prev_shval
    return _prev_shval


"""

Sometimes we want stdout
Sometimes we want stderr
We never want stdin. Stdin will be handled by bash with "|" and such, we dont need to manage any of that at this level

"""

def esrun(*popenargs, debug=u.Debug(False), **kwargs):
    with Popen(*popenargs, **kwargs) as process:
        try:
            debug.print(f"starting communicate()")
            #stdout, stderr = process.communicate(None)
            stdout, stderr = escommunicate(process, debug=debug)
            debug.print(f"ended communicate()")
        except KeyboardInterrupt:
            debug.g(f"ended comm by: found keyboard interrupt in child")
        except:
            debug.g(f"ended comm by: myrun() is killing a child of {os.getpid()}")
            process.kill()
            # We don't call process.wait() as .__exit__ does that for us.
            raise
        retcode = process.poll()
    return CompletedProcess(process.args, retcode, stdout, stderr)


def escommunicate(process, debug=u.Debug(False)):
    try:
        stdout, stderr = _escommunicate(process, debug=debug)
    except KeyboardInterrupt:
        debug.print("keyboard int in escommunicate()")
    finally:
        process._communication_started = True # good to set it just in case it's used by someone else
    return stdout, stderr


import selectors

def _escommunicate(self, debug=u.Debug(False)):
    stdout = None
    stderr = None

    self._fileobj2output = {}
    if self.stdout:
        self._fileobj2output[self.stdout] = []
    if self.stderr:
        self._fileobj2output[self.stderr] = []

    if self.stdout:
        stdout = self._fileobj2output[self.stdout]
    if self.stderr:
        stderr = self._fileobj2output[self.stderr]

    with sp._PopenSelector() as selector:
        if self.stdout:
            debug.print("registering stdout")
            selector.register(self.stdout, selectors.EVENT_READ)
        if self.stderr:
            debug.print("registering stderr")
            selector.register(self.stderr, selectors.EVENT_READ)

        while selector.get_map():
            try:
                ready = selector.select()
                # XXX Rewrite these to use non-blocking I/O on the file
                # objects; they are no longer using C stdio!
                for key, events in ready:
                    if key.fileobj in (self.stdout, self.stderr):
                        data = os.read(key.fd, 32768)
                        if not data:
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                        self._fileobj2output[key.fileobj].append(data)
            except KeyboardInterrupt:
                pass

    eswait(self, debug=debug)

    # All data exchanged.  Translate lists into strings.
    if stdout is not None:
        stdout = b''.join(stdout)
    if stderr is not None:
        stderr = b''.join(stderr)

    # Translate newlines, if requested.
    # This also turns bytes into strings.
    if self.text_mode:
        if stdout is not None:
            stdout = self._translate_newlines(stdout, self.stdout.encoding, self.stdout.errors)
        if stderr is not None:
            stderr = self._translate_newlines(stderr, self.stderr.encoding, self.stderr.errors)

    return (stdout, stderr)


def eswait(self, debug=u.Debug(False)):
    debug.print("entering eswait")
    while self.returncode is None:
        try:
            with self._waitpid_lock:
                if self.returncode is not None:
                    break  # Another thread waited.
                (pid, sts) = self._try_wait(0)
                # Check the pid and loop as waitpid has been known to
                # return 0 even without WNOHANG in odd situations.
                # http://bugs.python.org/issue14396.
                if pid == self.pid:
                    self._handle_exitstatus(sts)
        except KeyboardInterrupt:
            pass
    debug.print("exiting eswait")



"""

some Awk ideas:

we want to print up until $3 exceeds 1000:
def aux(self,toks):
    if(toks[3]) > 1000:
        return HALT
    return VERBATIM
Awk(aux)(lines)

oneline:
Awk('if $3 > 1000: HALT;; VERBATIM')
Awk('(if $3 > 1000: HALT) VERBATIM') alternate syntax, both should be accepted. it's harder to write this parenthesized version while looking ahead.
^dollarsign vars
^HALT and VERBATIM expand to proper return statements
^ double semicolon to indicate a dedent


count the number of lines with 5 tokens:
def aux(self,toks):
    if len(toks) == 5:
        self.count += 1
Awk(aux).begin(lambda self: self.count=0)(lines).count

Awk('if $NR == 5: count++').count
^self.count transformation. All locals are extracted to become state variables.
^ints like self.count are created and initialized to 0
^The postfix ++ operator was created
^$NR special variable


"""

VERBATIM = object() # just a generic object with a unique id
HALT = object()

class Awk:
    def __init__(self, fn):
        self._step = fn
        self._begin = lambda self, toks: toks
        self._end = lambda self, toks: toks
    def begin(self, fn):
        self._begin = fn
    def end(self, fn):
        self._end = fn
    def __call__(self, lines):
        self.out = []
        self._begin(self)
        for line in lines:
            if isinstance(str,line):
                toks = line.split(' ')
            else:
                toks = line
            res = self._step(self,line)
            if res is None:
                continue
            if res is VERBATIM:
                self.out.append(line)
            if res is HALT:
                break
            self.out.append(res)
        self._end(self)
        return self.out





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

