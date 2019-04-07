# CODEGEN
# OVERVIEW:
# class Tok: an enum of the possible tokes, eg Tok.WHITESPACE
# class Token: a parsed token. has some associated data
# class AtomCompound, etc: These Atoms are AST components. Each has a .gentext() method that generates the actual final text that the atom should become in the compiled code
# parse() is the main function here. It goes string -> Token list -> Atom list -> Atom list (w MacroAtoms) -> final python code


## TODO next: make SInitial more pretty. I dont htink overloader is the way to go, we shd start in SInit then trans to Snormal not just wrap Snormal.


## assertion based coding. After all, we're going for slow-but-effective. And assertions can be commented in the very final build. This is the python philosophy - slow and effective, but still fast enough



from enum import Enum,unique
import re
import os
from util import die,warn,Debug
import util as u
import inspect

# Tok is the lowest level thing around. 
# It's just an enum for the different tokens, with some built in regexes
@unique
class TokTyp(Enum):
    # IMPORTANT: order determines precedence. Higher will be favored in match
    #MACROHEAD = re.compile(r'%(\w+\??)')
    WHITESPACE= re.compile(r'(\s+)') ##note you can never have multiple whitespaces in a row!
    COMMA     = re.compile(r',')
    COLON     = re.compile(r':')
    EXCLAM     = re.compile(r'!')
    FLAG     = re.compile(r'-(\w+)')
    PERIOD     = re.compile(r'\.')
    EQ     = re.compile(r'=')
    SH_LBRACE     = re.compile(r'sh{')
    SH_LINESTART     = re.compile(r'sh\s+')
    LPAREN    = re.compile(r'\(')
    RPAREN    = re.compile(r'\)')
    LBRACE    = re.compile(r'{')
    RBRACE    = re.compile(r'}')
    LBRACKET  = re.compile(r'\[')
    RBRACKET  = re.compile(r'\]')
    ESCQUOTE2    = re.compile(r'\\\"')
    ESCQUOTE1    = re.compile(r'\\\'')
    QUOTE2    = re.compile(r'\"')
    QUOTE1    = re.compile(r'\'')
    DOLLARESC = re.compile(r'\$\$')
    DOLLARPAREN = re.compile(r'\$\(')
    DOLLARVAR = re.compile(r'\$(\w+)')
    ID      = re.compile(r'([a-zA-z_]\w*)')
    INTEGER    = re.compile(r'(\d+)')
    UNKNOWN   = re.compile(r'(.)')
    SOL = 0 # should never be matched against since 'UNKOWN' is a catch-all
    EOL = 1 # should never be matched against since 'UNKOWN' is a catch-all
    def __repr__(self):
        if self.name in ["SH_LINESTART","SH_LBRACE"]:
            return u.mk_g(self.name)
        if self.name == "MACROHEAD":
            return u.mk_p(self.name)
        if self.name == "DOLLARVAR":
            return u.mk_y(self.name)
        if self.name == "WHITESPACE":
            return u.mk_gray("WS")
        return self.name

def closer_of_opener(opener_tok):
    if isinstance(opener_tok,Tok):
        opener_tok = opener_tok.typ

    if opener_tok == TokTyp.LPAREN:
        return Tok(TokTyp.RPAREN,'',')')
    if opener_tok == TokTyp.DOLLARPAREN:
        return Tok(TokTyp.RPAREN,'',')')
    if opener_tok == TokTyp.LBRACE:
        return Tok(TokTyp.RBRACE,'','}')
    if opener_tok == TokTyp.LBRACKET:
        return Tok(TokTyp.RBRACKET,'',']')
    if opener_tok == TokTyp.QUOTE1:
        return Tok(TokTyp.QUOTE1,'','\'')
    if opener_tok == TokTyp.QUOTE2:
        return Tok(TokTyp.QUOTE2,'','"')
    if opener_tok == TokTyp.SOL:
        return Tok(TokTyp.EOL,'','')
    raise NotImplementedError


# This is a parsed token
class Tok:
    def __init__(self,typ,data,verbatim):
        self.typ=typ            #e.g. Tok.IDENTIFIER
        self.data=data          # the contents of the capture group of the Tok's regex, if any.
        self.verbatim=verbatim  # e.g. 'foo'. The verbatim text that the regex matched on
    def __repr__(self):
        out = self.typ.__repr__()
        if self.data != '':
            out += '('+ u.mk_y(self.data) + ')'
        return out

# a fancy iterator globally used for keeping track of the current state of processing the token stream
class TokStream:
    def __init__(self,token_list):
        self.tkns = token_list
        self.idx = 0

    # returns the number of remaining tokens (including whatever is currently pointed to)
    ##CRFL!
    def __len__(self):
        return max(0,len(self.tkns)-self.idx)
    # a request beyond the end will return None
    ##CRFL!
    def __getitem__(self,rel_idx):
        if self.idx + rel_idx >= len(self.tkns):
            return None #EOL and beyond indicator
        return self.tkns[self.idx+rel_idx]
    # a request of 4 items starting at 2 before the end would yield [tok, tok, None, None]
    # thus it is safe to do v=tstream[:3] then do v[2] for example, as long as you handle the None case
    # (the indexing itself will never throw an error)
    ##CRFL!
    def __getslice__(self, start, end):
        res = self.tkns[self.idx+start:self.idx+end]
        res += [None] * (end-start - len(res)) # fill in rest of request length with Nones
        return res
    def step(self,ntoks=1):
        self.idx += ntoks
        u.gray("'"+self[0].verbatim+"'" if self[0] is not None else 'None')
    def skip_whitespace(self):
        if self[0].typ == TokTyp.WHITESPACE:
            self.step() #note you can never have mult whitespaces in a row since they consolidate by \s+

    ## left commented for now in hopes that we'll have nice clean code that will never need to do this
    def rewind(self,ntoks=1):
        self.idx -= ntoks


    ## TokStream: LEFT AS WIP. unclear what best design is for these methods. best to make form fit the function by designing States first

def toks_to_typs(tok_list):
    return [t.typ for t in tok_list]



#def initialize(token_list,globals):
#    global tstream
#    tstream = TokStream(token_list,globals)

def assertmsg(cond,msg):
    if cond is False:
        die("assert failed: "+msg)

## Specifications and Assumptions:
## * transition(t) is always called with tstream[0] being t
## * whenever possible we favor run()ing new states only after pointing tstream[0] to the first item that they will be parsing, and we pass any metadata that resulted in their discovery to them. For example when starting a new space-call from "foo 1 2" we were in SNormal mode pointing to 'foo', then we advance twice (through the whitespace) to reach '1' at which point we run_same() SSPACECALL which is fed the proper fname='foo'. Similarly SNormal is initialized by passing it its 'opener' token (e.g. '(') while tstream[0] points one past the opener, as opposed to starting SNormal with tstream[0] pointing to the opener and allowing it to figure that out itself.
        ## * IT IS UNCLEAR WHETHER THIS METHOD OR THE OPPOSITE METHOD MAKES MORE SENSE. The opposite method means that all __init__ constructors for State subclasses generally wouldn't need extra arguments (except in special cases that haven't been encountered yet). Really the answer should be to use whichever makes more sense. Do we want the setup work done by __init__ or do we want it done by the parent
        ## In the following example SSpacecall requires quite a bit of setup involving skipping whitespace that it seems like you might want to package that into init().
        ## note by the way that you can't include skip_whitespace in the __init__ currently because if you do run_next that runs AFTER __init__ which will fuck you up.
#            tstream.step() # now tstream[0] pointing to the whitespace
#            tstream.step() # now tstream[0] pointing one beyond whitespace
#            tstream.skip_whitespace() # skip over any inital whitespace if it exists
#            self.run_same(SSpacecall(self,t.data))
        # EHH ACTUALLY THIS IS A BAD EXAMPLE BC YOU CAN DELETE THE SKIP_WS LINE AND IT WORKS SINCE ALL WS IS AT MOST 1-length by virtue of \s+
        ## Regardless of the method, all States should be clearly annotated with an indicator for what they should be pointing to when run() is called
        ## Note that **good** idea is to combine both approaches: pass in the 'opener' or whatever but also leave it pointing to the opener, and let it do any verification through assert() calls. This would make parser logic bugs much easier to find. State can even have an abstract assert_proper function that gets called at the start of __init__ and both checks validity thru assert calls and does the proper setup. OH WAIT NO thats ***not*** a great idea bc super.__init__ gets called at the start of __init__ so super.init cant be doing all the tests. Also it would mess with the whole build-and-run oneliner tho thats not the end of the world.
    #####^^^^^^^^If it aint broke don't fix it: Just wait until you have a REASON that this is a bad system. Bc right now it works, and it works well^^^^^^^^
    ##### ALSO when you enter SNormal due to a '(' you def wanna be one past that when you first call transition() bc otherwise itll see the '(' and enter into ANOTHER SNormal 
## 
## 
## 
NONE = 0
VERBATIM = 1
POP = -1


### the rules of writing a new State subclass: ###
# Flow of execution for a state, from creation to death:
# __init__(): use this to declare any variables you need, and of course any constructor arguments you want. Start it by calling super().__init__(parent). The key with this is you should NOT interact with tstream as it is at an undefined position at this point.
    # (note that the undefined position thing is because we allow the constructor to be run, then .step() to be called, then .run_*() to be called. And in fact run_next needs to be called using a constructed state, then it .step()s and then .run()s. Hence __init__ will not be seeing the tstream at any sort of consistent, predictable position when it is initialized)
# next run() is invoked (generally by a different state calling run_same or run_next). You can't modify run(). It will call the following few functions. Note that we always assume that run() gets called with tstream[0] being the first token that should be processed, and run() will exit with it pointing to the last token that was processed (it will not step beyond this one).
# pre(): run() will call this first, with no arguments. Use for any setup that depends on tstream. It's best practice to include assertmsg() calls that check that tstream is properly lined up (vastly improves ease of debugging).
# transition():
#   -only call run_same(child) when tstream[0] points to the first token you want child to see. (Call run_next(child) when tstream[1] points to the first token you want child to see)
#   -return POP when the NEXT token (tstream[1]) is the one you want your parent to see. Think of POP as pop_next. (Pop does not actually .step(), but you will return from it into the completion of the parent's transition() function so the step() will automatically occur as transition() returns into run() which calls .step and then .transition again)
# post(text): this will be called with the final accumulated result of the transition() loop, and should do any necessary post processing on it, for example wrapping it in '(' ')' characters for a parenthetical state. post() should also have assert() calls to verify that tstream is properly aligned.



class State:
    def __init__(self,parent,tstream=None, globals=None, debug=None):
        self.parent = parent
        self.halt = False # halt is effectively returning a value AND popping
        self._tstream = parent._tstream if (tstream is None) else tstream
        self._globals = parent._globals if (globals is None) else globals
        self.debug_name = "[debug_name not set: {}]".format(self.__class__)
        self.debug_depth = parent.debug_depth+1 if (parent is not None) else 0
        self.d = parent.d if debug is None else debug
        #self.popped = False
        #self.nostep = False
        #self.tmp = None # just a useful temp var for subclasses that dont want to go to the work of overriding init

    def run(self):
        text = ''
        self.pre()
        self.d.r("starting: "+self.debug_name)
        while True:
            if self.tok() is None:
                assertmsg(len(self._tstream)==0,'if tstream[0] is None that must mean we are out of tokens')
                self.d.print("EOL autopop from run()")
                break # autopop on EOL

            tmp_str = "{}.transition('{}')".format(self.debug_name,self.tok().verbatim)
            self.d.y(tmp_str)
            res = self.transition(self.tok())
            assertmsg(res is not None, 'for clarity we dont allow transition() to return None. It must return '' or NONE (the global constant) for no extension)')
            tmp_res = res
            if res == VERBATIM:
                tmp_res = self.tok().verbatim
            elif res == NONE:
                tmp_res = '[none]'
            elif res == POP:
                tmp_res = 'POP'
            self.d.y("{} -> '{}'".format(tmp_str,tmp_res))

            if isinstance(res,str): # extend text
                text += res
            elif res == POP: # no step on pop
                break
            elif res == NONE: # do nothing. same as res=''
                pass # not a 'continue' bc we still want the later .step() to run
            ## careful. VERBATIM will use whatever tstream[0] is pointing to when transition() exits
            elif res == VERBATIM: # verbatim shortcut
                text += self.tok().verbatim

            if self.halt: # no step on pop
                break
            self.step()
        self.d.p("before postproc: "+text)
        out = self.post(text)
        self.d.b("{}:{}".format(self.debug_name,out))
        return out
    def pre(self): # default implementation
        pass
    def post(self,text): # default implementation
        return text
    def run_same(self,state):
        return state.run() # will always return -1
    def run_next(self,state):
        self._tstream.step()
        return state.run() # will always return -1
    def pop(self,value):
        self.popped = True
        return value
    def transition(self,t):
       raise NotImplementedError # subclasses must override

    def assertpre(self,cond,msg=None):
       m = "pre-assert failed for {}".format(self.debug_name)
       if msg is not None:
           m += ' : ' + msg
       assertmsg(cond,m)
    def assertpost(self,cond,msg=None):
       m = "post-assert failed for {}".format(self.debug_name)
       if msg is not None:
           m += ' : ' + msg
       assertmsg(cond,m)
    # functions for managing _tstream and _globals
    # tok()
    # tok(1)
    # tok(list=True)[1:3]
    def tok(self,idx=0,list=False):
        if list:
            return self._tstream
        return self._tstream[idx]
    def step(self):
        return self._tstream.step()
    def rewind(self):
        return self._tstream.rewind()
    # if fname is not in globals or is not callable, return False
    def check_callable(self,fname):
        return callable(self._globals.get(fname))
    # always call check_callable() at some point before argc_of_fname()
    def argc_of_fname(self,fname):
        func = self._globals.get(fname)
        params = inspect.signature(func).parameters.values()
        required_argc = len(list(filter(lambda x: x.default==inspect._empty, params)))
        return required_argc
    def skip_whitespace(self):
        return self._tstream.skip_whitespace()

#   (X      or     {X   etc
#    ^              ^
class SNormal(State):
    def __init__(self,parent,opener): #opener is a Tok
        ## careful. init is called when tstream[0] will not be pointing to whatever token the first transition(t) will be called on. The class e.g. SNormal() will be initialized THEN tstream.idx will be updated (e.g. +1 or +0) THEN run() will be called
        super().__init__(parent)
        self.opener = opener
        self.closer = closer_of_opener(opener) # closer is a TokTyp
        self.debug_name = "SNormal[{}]".format(opener.verbatim)
    def pre(self):
        if self.opener.typ != TokTyp.SOL:
            self.assertpre(self.tok(-1)==self.opener)
    def transition(self,t):
        ## transition(t) always ASSUMES that tstream[0] == t. Feeding an arbitrary token into transition is undefined behavior. Though it should only have an impact on certain peeks
        assertmsg(self.tok() is t, "transition(t) assumes that tstream[0] == t and this has been violated")

        if t.typ == self.closer.typ:
            return POP
        elif t.typ == TokTyp.SH_LBRACE:
            return self.run_next(SShmode(self))
        elif t.typ in [TokTyp.LPAREN, TokTyp.LBRACKET, TokTyp.LBRACE]:
            return self.run_next(SNormal(self,t))
        elif t.typ in [TokTyp.QUOTE1, TokTyp.QUOTE2]:
            return self.run_next(SQuote(self,t))
        elif t.typ == TokTyp.ID and self.check_callable(t.data) and (self.tok(1) is None or self.tok(1).typ == TokTyp.WHITESPACE):
            self.step() # now tstream[0] pointing to the whitespace
            self.step() # now tstream[0] pointing one beyond whitespace (which can no longer be a whitespace since WS = \s+)
            return self.run_same(SSpacecall(self,t.data))
        return VERBATIM
    def post(self,text):
        if self.opener.typ != TokTyp.SOL:
            self.assertpost(self.tok().typ == self.closer.typ)
        return self.opener.verbatim + text + self.closer.verbatim


#   "X      or      'X
#    ^               ^
class SQuote(State):
    def __init__(self,parent,opener): #opener is a Tok
        super().__init__(parent)
        self.opener = opener
        self.closer = opener # for quotes a closer is it's own opener
        self.debug_name = "SQuote[{}]".format(opener.verbatim)
    def transition(self,t):
        if t.typ == self.closer.typ:
            return POP
        return VERBATIM
    def post(self,text):
        return self.opener.verbatim + text + self.closer.verbatim

#   "X      or      'X
#    ^              ^
class SShquote(State):
    def __init__(self,parent,opener): #opener is a Tok
        super().__init__(parent)
        self.opener = opener
        self.closer = opener # for quotes a closer is it's own opener
        self.debug_name = "SShquote[{}]".format(opener.verbatim)
    def transition(self,t):
        if t.typ == self.closer.typ:
            return POP
        return VERBATIM
    def post(self,text):
        return self.opener.verbatim + text + self.closer.verbatim

# sh{X
#    ^
class SShmode(State):
    def __init__(self,parent,capture_output=True):
        super().__init__(parent)
        self.brace_depth = 1
        self.debug_name = "SShmode"
        self.capture_output = capture_output
    def transition(self,t):
        if t.typ == TokTyp.LBRACE:
            self.brace_depth += 1
            return VERBATIM
        elif t.typ == TokTyp.RBRACE:
            self.brace_depth -= 1
            if self.brace_depth == 0:
                return POP
            return VERBATIM
        elif t.typ in [TokTyp.QUOTE1, TokTyp.QUOTE2]:
            return self.run_next(SQuote(self,t))
        elif t.typ == TokTyp.DOLLARPAREN:
            return self.run_next(SNormal(self,t))
        return VERBATIM
    def post(self,text):
        return 'backend.sh("{}",capture_output={})'.format(text,self.capture_output)


# foo    a b c
#        ^non whitespace (note that all contig whitespace is at most length 1 bc of \s+)
# foo a bar b car 
# foo(a,bar(b,car()))
class SSpacecall(State):
    def __init__(self,parent,func_name):
        super().__init__(parent)
        self.func_name = func_name
        self.argc = self.argc_of_fname(self.func_name)
        self.argc_left = self.argc
        self.debug_name = "SSpacecall[{}]".format(func_name)

    def transition(self,t):
        if self.argc == 0:
            return POP ##todo make this better so that "sum one one" works
        self.argc_left -= 1
        if self.argc_left == 0:
            self.halt = True

        dont_abort_verbatim = ['>','>=','<=','==','<','=','*','+','-','/','//'] ##UNFINISHED, Add more to this! in general boolop/binop/unop/cmp. Note i left '=' in since right now we parse '==' as '=','='.
        over = Overloader(self,SNormal(self,Tok(TokTyp.SOL,'','')))
        over.prev_non_ws = None # local var used by lambdas. Last non-whitespace char seen
        def pre_trans(t): # keep track of last non-whitespace seen
            if t.typ != TokTyp.WHITESPACE:
                over.prev_non_ws = t

        over.pre_trans = pre_trans # by closure 'pre' will properly hold the correct references to 'over'
        over.pop = lambda t: (t.typ == TokTyp.WHITESPACE and (over.prev_non_ws is None or over.prev_non_ws.verbatim not in dont_abort_verbatim))
        return self.run_same(over) + ','

    def post(self,text):
        self.rewind() ## TODO fix this grossness, or just accept it. it aligns it so when caller calls next() they recv the last space of the macro, which is imp for recursive macros
        return self.func_name+'('+text[:-1]+')' # kill the last comma with :-1

class Overloader(State):
    def __init__(self,parent,state):
        super().__init__(parent)
        self.debug_name = "Overloader[{}]".format(state.debug_name)
        self.inner = state
        self.pre_trans = lambda t: None
        self.post_trans = lambda t: None
        self.pop = lambda t: False
        self.override = lambda t: NONE
        self.use_override = [] # list of tokens for which override(t) should be used in place of inner.transition(t)
    def transition(self,t): ##maybe allow pre/post to modify res, or something like that. Or be able to selectively use an over.alt(t) function instead of inner.trans whenever self.usealt(t) is true?
        if self.pop(t): return POP
        self.pre_trans(t)
        if t in self.use_override:
            res = self.override(t)
        else:
            res = self.inner.transition(t)
        self.post_trans(t)
        return res

# the first state just used at the start of the line
class SInitial(State):
    def __init__(self,parent,tstream, globals, debug):
        super().__init__(parent, tstream=tstream, globals=globals, debug=debug)
        self.debug_name = "SInitial"
        self.debug = debug
    def transition(self,t):
        self.halt = True # this is a 1 shot transition function
        ##This should handle the ">a" syntax and the quick-fn-def syntax, and should do a self.run_same to SNormal if neither case is found

        ## and : linestart syntax for sh line
        if t.typ == TokTyp.COLON:
            res = self.run_next(SShmode(self,capture_output=False)) ##allow it to kill itself from eol
        else:
            res = self.run_same(SNormal(self,Tok(TokTyp.SOL,'','')))
        assertmsg(len(self._tstream)==0,'tstream should be empty since SInitial should consume till EOL')
        return res


# turns a string into a list of Tokens
def tokenize(s):
    remaining = s
    tkns = []
    while remaining != '':
        for t in list(TokTyp):
            match = t.value.match(remaining)
            # 'continue' if no match
            if match is None: continue
            # 'continue' if sh_linestart isn't at start of line
            if t == TokTyp.SH_LINESTART and remaining != s: continue

            remaining = remaining[match.end():]
            grps = match.groups()
            data = grps[0] if grps else '' # [] is nontruthy
            tkns.append(Tok(t,data,match.group()))
            break #break unless you 'continue'd before
    return tkns


# the main function that run the parser
# It goes string -> Token list -> Atom list -> Atom list (w MacroAtoms) -> final python code
def parse(line,globals,debug=False):
    debug = True
    debug = Debug(debug)
    token_list = tokenize(line)
    debug.print(token_list)
    tstream = TokStream(token_list)
    init = SInitial(parent=None, tstream=tstream, globals=globals, debug=debug)
    out = init.run()
    return out

#out = parse('test',{})
#u.blue('========')
#out = parse(':test',{})
#out = parse('x = (sh{test}) + 1',{})
#print(out)

#parse("fname,linect = %parse sh{wc -l $file} (str $1, int $2)")
#parse("z = %parselines1  x (str $1, int $2)")
#parse("sh{echo \"hi \"$there\" you're the \"$one}")
#parse("if %exists? filename:")
#parse("vi_list = %parselines1 (%cat file) (int $1, int $2)")




