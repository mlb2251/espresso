# CODEGEN
# OVERVIEW:
# class Tok: an enum of the possible tokes, eg Tok.WHITESPACE
# class Token: a parsed token. has some associated data
# class AtomCompound, etc: These Atoms are AST components. Each has a .gentext() method that generates the actual final text that the atom should become in the compiled code
# parse() is the main function here. It goes string -> Token list -> Atom list -> Atom list (w MacroAtoms) -> final python code


from enum import Enum,unique
import re
import os
from util import *
from util import die
import inspect



# Tok is the lowest level thing around. 
# It's just an enum for the different tokens, with some built in regexes
@unique
class Tok(Enum):
    # IMPORTANT: order determines precedence. Higher will be favored in match
    MACROHEAD = re.compile(r'%(\w+\??)')
    WHITESPACE= re.compile(r'(\s+)')
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
    IDENTIFIER= re.compile(r'([a-zA-z_]\w*)')
    INTEGER    = re.compile(r'(\d+)')
    UNKNOWN   = re.compile(r'(.)')
    EOL = 0 # should never be matched against since 'UNKOWN' is a catch-all
    def __repr__(self):
        if self.name in ["SH_LINESTART","SH_LBRACE"]:
            return mk_green(self.name)
        if self.name == "MACROHEAD":
            return mk_purple(self.name)
        if self.name == "DOLLARVAR":
            return mk_yellow(self.name)
        if self.name == "WHITESPACE":
            return mk_gray("WS")
        return self.name

# num of args each macro takes.
macro_argc = {
    'argc':0,
    'argv':0,
    'basename':1,
    'blue':1,
    'cat':1,
    'cd':1,
    'die':1,
    'direxists?':1,
    'exists?':1,
    'exists?':1,
    'fsize':1,
    'fullpath':1,
    'green':1,
    'head':1,
    'ls':0,
    'mkdir':1,
    'mkdirp':1,
    'p':1,
    'parse':2,
    'parselines':2,
    'parselines1':2,
    'purple':1,
    'pwd':0,
    'red':1,
    'yellow':1,
}
def get_macro_argc(name):
    return macro_argc[name]

# for macros with args that need to be evaluated in an unusual manner
# e.g. the idx/cast list argument of %parse
def has_special_args(name): # returns boolean
    return name in ['parse', 'parselines', 'parselines1']

# This is a parsed token
class Token:
    def __init__(self,tok,data,verbatim):
        self.tok=tok            #e.g. Tok.IDENTIFIER
        self.data=data          # the contents of the capture group of the Tok's regex, if any.
        self.verbatim=verbatim  # e.g. 'foo'. The verbatim text that the regex matched on
    def __repr__(self):
        return self.tok.__repr__()


# turns a string into a list of Tokens
def tokenize(s):
    remaining = s
    tkns = []
    while remaining != '':
        for t in list(Tok):
            name = t.name
            match = t.value.match(remaining)
            # 'continue' if no match
            if match is None: continue
            # 'continue' if sh_linestart isn't at start of line
            if t == Tok.SH_LINESTART and remaining != s: continue

            remaining = remaining[match.end():]
            grps = match.groups()
            data = grps[0] if grps else '' # [] is nontruthy
            tkns.append(Token(t,data,match.group()))
            break #break unless you 'continue'd before
    tkns.append(Token(Tok.EOL,'','')) # end of line indicator
    return tkns



# may want to give custom codegen bodies later
# wow super dumb python bug: never have __init__(self,data=[]) bc by putting '[]' in the argument only ONE copy exists of it for ALL instances of the class. This is very dumb but in python only one instance of each default variable exists and it gets reused.

class TokenIter:
    def __init__(self,token_list,globals):
        self.token_list = token_list
        self.idx = 0 # points to the next result of next() or peek()
        self.globals=globals
    def __next__(self):
        result = self.token_list[self.idx].tok
        self.idx += 1
        #yellow('next() yielded: '+str(result))
        if result == Tok.EOL:
            self.back() # rewind on EOL to infinitely yield it
        return result
    def back(self):
        #yellow('backed up')
        self.idx -= 1
    def peek(self):
        return self.token_list[self.idx].tok
    def again(self): # don't advance, just show the last token you showed again
        return self.token_list[self.idx-1].tok
    def token_data(self): # gets the Token.data for the last token returned
        return self.token_list[self.idx-1].data
    def token_verbatim(self): # gets the Token.verbatim for the last token returned
        return self.token_list[self.idx-1].verbatim
    def ignore_whitespace(self):
        while self.peek() == Tok.WHITESPACE:
            next(self)
        gray('skipped WS')


# An atom that contains a list of other atoms. e.g. AtomParen
# Atoms are AST nodes.
class AtomCompound:
    def __init__(self):
        self.data=[]
    def add(self,atom):
        #magenta("adding "+str(atom)+" to "+str(self.__class__))
        self.data.append(atom)
    # does not recurse. removes top level whitespace of an AtomCompound
    def rm_whitespace(self):
        self.data = list(filter(lambda a: not is_tok(a,Tok.WHITESPACE),self.data))
    # buildin overrides
    def __repr__(self):
        return self.pretty()
    def __str__(self):
        return self.__repr__()
    def pretty(self,depth=0):
        colorer = mk_cyan
        name = self.__class__.__name__
        if isinstance(self,AtomMacro):
            colorer = mk_purple
            name = self.name
        contents = colorer('[') + ' '.join([x.pretty(depth+1) for x in self.data]) + colorer(']')
        return '\n' + '\t'*depth + colorer(name) + contents
    def __bool__(self):  # important in macroize(). None == False and any atom == True
        return True
    def __iter__(self):
        yield from self.data
    def __getitem__(self, key):
        return self.data[key]
    def __setitem__(self, key, value):
        self.data.__setitem__(key,value)
    def __delitem__(self, key):
        self.data.__delitem__(key)
    def __len__(self):
        return len(self.data)


# Shorthand that parses entries into the following
# compound atoms and returns the resulting atom
# or returns None if t is not an entry symbol for an espresso-mode
# atom. Note dollar_parens are sh mode compounds hence not included here
# Entry symbols: Quote1/2, Paren/Brace/Bracket, Sh_brace
def try_esmode_compound(t,tkns,parent_exit_toks):
    atom = None
    if t == Tok.QUOTE1:
        atom = AtomQuote1(tkns)
    elif t == Tok.QUOTE2:
        atom = AtomQuote2(tkns)
    elif t == Tok.LPAREN:
        atom = AtomParen(tkns)
    elif t == Tok.LBRACE:
        atom = AtomBrace(tkns)
    elif t == Tok.LBRACKET:
        atom = AtomBracket(tkns)
    elif t == Tok.SH_LBRACE:
        atom = AtomSHBrace(tkns)
    elif t == Tok.IDENTIFIER:
        name = tkns.token_data()
        # check if macro-style call to a function
        # TODO change to this: within a macro-style fn call, if you pass in the function foo it assumes you're evaluating it like foo() or foo(...) for 'foo A B' so any of those cases are possible. If you want to pass the function itself in as an argument you must use "&foo"! Imp details:
            # bar(foo,1,2) where foo is a function. For backcompatibility this is NOT treated as bar(foo(),1,2)
            # 'head ls' using the functions head(list) and ls(). this translates to head(ls()).
            # 'head(ls)' (will throw runtime error ofc) - translates to head(ls)
            # 'head(cat "test.txt")' - translates to head(cat("test.txt"))
            # so basically:
                # a zero-arg function will not be autoevaluated unless it's part of an outer macro expression ie treated as an ArgAtom
                # if it is an argatom you need to call it &foo instead of foo to suppress the evaluation. Same applies to multiarg functions
                # principle -- unfortunately deviates from python normality (tho still backcompatible), but passing a function object in is WAY less common than passing in the result of evaluating the function. In particular in the case of chaining macros.
        if tkns.peek() == Tok.EXCLAM and (name in tkns.globals) and callable(tkns.globals[name]):
            next(tkns) # consume that Tok.EXCLAM
            fn = tkns.globals[name]
            # use fn.parse_hook(tkns) if the function exists.
            # parse_hook() should return an AtomMacro
            # TODO change this around so you can basically either fully control the AtomMacro
            # from the parse_hook or you can just for example make one particular argument 
            # parsed in your own way -- you're given the option somehow.
            # e.g. could either extend AtomMacro or could extend AtomArg perhaps
            if 'parse_hook' in dir(fn):
                return fn.parse_hook(tkns)
            params = inspect.signature(fn).parameters.values()
            # required params are any without defaults
            # TODO this process doesn't handle KEYWORD_ONLY and other kinds
            # of args bc I don't know how functions can be used completely and need
            # to research it more (e.g. theres some true kwargs stuff etc)
            required_argc = len(list(filter(lambda x: x.default==inspect._empty,params)))
            blue(f"required_argc:{required_argc}")
            # first all args without defaults must be provided
            # TODO then after than kwargs can be provided
            arg_atoms = []
            for i in range(required_argc):
                tkns.ignore_whitespace() # allow arbitrary whitespace between args
                arg_atom = AtomArg(tkns,parent_exit_toks=parent_exit_toks)
                arg_atoms.append(arg_atom)
            atom = AtomMacro(name,arg_atoms)
    else:
        atom = None
    return atom

def try_shmode_compound(t,tkns):
    if t == Tok.DOLLARVAR:
        atom = AtomDollar(tkns) # TODO really merge AtomDollar into AtomDollarParen
    elif t == Tok.DOLLARPAREN:
        atom = AtomDollarParen(tkns)
    else:
        atom = None
    return atom


#The parent atom for a line of code
class AtomMaster(AtomCompound):
    def __init__(self,tkns):
        super().__init__()
        t = next(tkns)
        if t == Tok.SH_LINESTART:
            atom = AtomSHLine(tkns)
            self.add(atom)
            return
        # Enter: Quote1/2, Paren/Brace/Bracket, Sh_brace
        # Default to AtomTok: DollarParen, AtomDollar, etc
        while t is not Tok.EOL:
            #cyan("Master step")
            # compound
            atom = try_esmode_compound(t,tkns,parent_exit_toks=[Tok.EOL])
            if not atom:
                # non-compound
                atom = AtomTok(tkns)
            self.add(atom)
            t = next(tkns)
    def gentext(self):
        return ''.join([x.gentext() for x in self])

# the sh{} atom
class AtomSHBrace(AtomCompound):
    def __init__(self,tkns):
        super().__init__()
        t = next(tkns)
        depth = 1
        # Enter: DollarParen, Dollarvar
        # Count: '{' and '}'
        # Exit: '}' and zero depth
        # Default to AtomTok: rest (including for shbrace bc just becomes verbatim)
        while t is not Tok.EOL:
            # compound
            atom = try_shmode_compound(t,tkns)
            if not atom:
                # brace-counting
                if t == Tok.LBRACE:
                    depth += 1
                    atom = AtomTok(tkns)
                elif t == Tok.RBRACE:
                    depth -= 1
                    if depth == 0: #EXIT
                        return
                    else:
                        atom = AtomTok(tkns)
                # non-compound
                else:
                    atom = AtomTok(tkns)
            self.add(atom)
            t = next(tkns)
    def gentext(self):
        body = ''.join([x.gentext() for x in self]).replace('"','\\"').replace('\1CONSERVEDQUOTE\1','"') # escape any quotes inside
        return 'backend.sh("' + body + '")'



# superclass for all the simple compound atoms
# with basic exit logic etc
class AtomCompoundSimple(AtomCompound):
    def __init__(self,tkns,mode,exit_toks):
        super().__init__()
        t = next(tkns)
        while t != Tok.EOL:
            # exit
            if t in exit_toks:
                return
            # compound
            atom = None
            if mode == 'es':
                atom = try_esmode_compound(t,tkns,parent_exit_toks=exit_toks)
            elif mode == 'sh':
                atom = try_shmode_compound(t,tkns)
            elif mode == 'quote':
                atom = None
            else:
                die('unrecognized mode for AtomCompoundSimple:'+str(mode))
            if not atom:
                # non-compound
                atom = AtomTok(tkns)
            self.add(atom)
            t = next(tkns)
        if Tok.EOL not in exit_toks:
            raise Exception(f'EOL with unclosed AtomCompoundSimple: {type(self)}. Closer should be: {exit_toks}')

class AtomArg(AtomCompoundSimple):
    def __init__(self,tkns,parent_exit_toks):
        super().__init__(tkns,mode='es',exit_toks=[Tok.WHITESPACE]+parent_exit_toks)
        blue(tkns.again())
        if tkns.again() in parent_exit_toks and tkns.again() != Tok.EOL: # EOL not included bc it autorewinds
            tkns.back()
    def gentext(self):
        return ''.join([x.gentext() for x in self])

class AtomMacro(AtomCompound):
    def __init__(self,fn_name,arg_atoms):
        super().__init__()
        self.data = arg_atoms
        self.name = fn_name
    def gentext(self):
        return self.name+"(" + ','.join([x.gentext() for x in self]) + ")"

class AtomSHLine(AtomCompoundSimple):
    def __init__(self,tkns):
        super().__init__(tkns,mode='sh',exit_toks=[Tok.EOL])
    def gentext(self):
        body = ''.join([x.gentext() for x in self]).replace('"','\\"').replace('\1CONSERVEDQUOTE\1','"') # escape any quotes inside
        return 'backend.sh("' + body + '",capture_output=False)'

class AtomQuote1(AtomCompoundSimple):
    def __init__(self,tkns):
        super().__init__(tkns,mode='quote',exit_toks=[Tok.QUOTE1])
    def gentext(self):
        return "\'" + ''.join([x.gentext() for x in self]) + "\'"

class AtomQuote2(AtomCompoundSimple):
    def __init__(self,tkns):
        super().__init__(tkns,mode='quote',exit_toks=[Tok.QUOTE2])
    def gentext(self):
        return "\"" + ''.join([x.gentext() for x in self]) + "\""

class AtomParen(AtomCompoundSimple):
    def __init__(self,tkns):
        #purple('entering Paren')
        super().__init__(tkns,mode='es',exit_toks=[Tok.RPAREN])
        #purple('exiting Paren')
    def gentext(self):
        return '('+''.join([x.gentext() for x in self]) + ')'

class AtomBracket(AtomCompoundSimple):
    def __init__(self,tkns):
        super().__init__(tkns,mode='es',exit_toks=[Tok.RBRACKET])
    def gentext(self):
        return '['+''.join([x.gentext() for x in self]) + ']'

class AtomBrace(AtomCompoundSimple):
    def __init__(self,tkns):
        super().__init__(tkns,mode='es',exit_toks=[Tok.RBRACE])
    def gentext(self):
        return '{'+''.join([x.gentext() for x in self]) + '}'

class AtomDollarParen(AtomCompoundSimple):
    def __init__(self,tkns):
        super().__init__(tkns,mode='es',exit_toks=[Tok.RPAREN])
    def gentext(self):
        py_expr =  ''.join([x.gentext() for x in self])
        return "\"+str({})+\"".format(py_expr).replace('"','\1CONSERVEDQUOTE\1')




# assert that the object 'child' is and instance of the class 'parent'
# optionally 'parent' can be a list of possible classes
def assertInst(child,parent):
    do_die = True
    if isinstance(parent,list):
        for cls in parent:
            if isinstance(child,cls):
                do_die = False
    elif isinstance(child,parent):
        do_die = False
    if do_die:
        die(str(child) + " is not an instance of " + str(parent))

# any token that isn't converted to some other Atom
class AtomTok: # not a subclass bc we don't want it to inherit add()
    def __init__(self,tkns):
        self.tok = tkns.again() # reshow last tok
        self.data = tkns.token_data()
        self.verbatim = tkns.token_verbatim()
    def gentext(self):
        return self.verbatim
    # builtin overrides
    def __repr__(self):
        return self.pretty()
    def __str__(self):
        return self.__repr__()
    def __bool__(self):
        return True
    def pretty(self,depth=0):
        if self.tok == Tok.MACROHEAD:
            return mk_purple(self.data)
        if self.tok == Tok.DOLLARVAR:
            return mk_yellow(self.data)
        if self.tok == Tok.WHITESPACE:
            return mk_gray('WS')
        return self.tok.name


# dollar variables, like for sh{} interpolation with $foo
class AtomDollar(AtomTok):
    def __init__(self,tkns):
        super().__init__(tkns)
    def gentext(self): #for dollar this is only called when NOT part of a special macro construct like idx_casts (int $1, int $3) etc
        #if self.mode == 'global': #TODO make it actually use this
        #    return 'os.environ["'+self.data+'"]'
        #elif self.mode == 'interp':
        return "\"+str({})+\"".format(self.data).replace('"','\1CONSERVEDQUOTE\1')
        #else:
        #    die("unrecognized mode:{}".format(self.mode))

# turn a token list into a MASTER atom containing all other atoms
def atomize(tokenlist,globals):
    tkns = TokenIter(tokenlist,globals)
    atoms = AtomMaster(tkns)
    return atoms



#    curr_quote = None #None, Tok.QUOTE1 or Tok.QUOTE2
#    parents = [('master',None)] # the None can hold extra data eg brace depth for SH
#    atoms = [AtomMaster()]
#    def parent(): return parents[-1][0]
#    def data(): return parents[-1][1]
#    def parent_atom(): return atoms[-1]
#    for token in tokenlist:
#        t = token.tok #the Tok
#        if t == Tok.DOLLARVAR and parent() in ['sh_brace','sh_line']:
#            as_tok = AtomDollar(token,'interp')
#        elif t == Tok.DOLLARVAR and parent() not in ['sh_brace','sh_line']:
#            as_tok = AtomDollar(token,'global')
#        else:
#            as_tok = AtomTok(token)
#        # IF IN QUOTE ATOM, then everything is a plain token other than the exit quote token
#        if parent() == 'quote': #highest precedence -- special rules when inside quotes
#            # EXIT QUOTE ATOM?
#            if t == data()['curr_quote']: #close quotation
#                assertInst(atoms[-1],AtomQuote)
#                parents.pop()
#                atoms[-2].add(atoms.pop())  #pop last atom onto end of second to last atom's data
#            # STAY IN QUOTE ATOM
#            else:
#                atoms[-1].add(as_tok)
#        # ENTER QUOTE ATOM?
#        elif t == Tok.QUOTE1 or t == Tok.QUOTE2:
#            parents.append(('quote',{'curr_quote':t}))
#            atoms.append(AtomQuote(token))    #start a new quote atom
#        # IF IN SH, then we don't care about anything but { } counting and DollarParens
#        # everything else is flattened (so no worries about needing to recurse or keep track
#        # of having a sh{} somewhere earlier on the stack)
#        elif parent() in ['sh_brace','sh_line']:
#            if parent() == 'sh_brace':
#                if t == Tok.SH_LBRACE: die("SH_LBRACE inside of an sh{}")
#                elif t == Tok.LBRACE:
#                    data()['brace_depth'] += 1
#                    continue #these continues are important
#                elif t == Tok.RBRACE:
#                    data()['brace_depth'] -= 1
#                    # LEAVE SH
#                    if data()['brace_depth'] == 0:
#                        assertInst(atoms[-1],AtomSH)
#                        parents.pop()
#                        atoms[-2].add(atoms.pop())
#                    continue #these continues are important
#            # ENTER DOLLARPARENS? (only possible from within SH)
#            if t == Tok.DOLLARPAREN:
#                parents.append('dollarparens')
#                atoms.append(AtomDollarParens())
#            #STAY IN SH
#            else:
#                atoms[-1].add(as_tok)
#
#        ###### REST IS FOR OUTSIDE SH OUTSIDE QUOTE:
#        # ENTER SH?
#        elif t == Tok.SH_LBRACE:
#            parents.append(('sh_brace',{'brace_depth':1}))
#            atoms.append(AtomSH())
#        elif t == Tok.SH_LINESTART:
#            parents.append(('sh_line',None))
#            atoms.append(AtomSHLine())
#        # OPEN PARENS
#        elif t == Tok.LPAREN:
#            parents.append(('parens',None))
#            atoms.append(AtomParens())
#        # CLOSE PARENS
#        elif t == Tok.RPAREN:
#            assertInst(atoms[-1],[AtomParens,AtomDollarParens])
#            parents.pop()
#            atoms[-2].add(atoms.pop())
#        else:
#            atoms[-1].add(as_tok)
#    if len(atoms) == 2 and parent() == 'sh_line':
#        assertInst(atoms[-1],AtomSHLine)
#        parents.pop()
#        atoms[-2].add(atoms.pop())
#    if len(atoms) != 1: die("There should only be the MASTER left in the 'atoms' list! Actual contents:"+str(atoms))
#    return atoms.pop()



# is_tok(atom,Tok.WHITESPACE) will be true if atom is an AtomTok with a token of type Tok.WHITESPACE
is_tok = lambda atom,tok: isinstance(atom,AtomTok) and atom.tok == tok



# transforms the AST to figure out the arguments for each macro
# Turns AtomTok of type Tok.MACROHEAD into AtomMacro containing a list of the argument Atoms
def macroize(base_atom):
    #if isinstance(base_atom,

    #helper fn to tell if an AtomTok is a macro and shd be converted to an AtomMacro
    curr_macro = None
    for i,atom in enumerate(base_atom):
        # end macro creation if all args filled
        if curr_macro and len(curr_macro.data) == curr_macro.argc:
            curr_macro = None

        # Recursively macroize all parens you come across regardless of if currently building a macro
        # very importantly this CANNOT be "elif"
        if isinstance(atom,AtomParens):
            macroize(atom)

        # always ignore whitespace
        if is_tok(atom,Tok.WHITESPACE):
            continue

        # handle adding arguments to curr macro (parens, sh, quotes, or macros with argc=0)
        elif curr_macro and (isinstance(atom,AtomParens) or isinstance(atom,AtomSH) or isinstance(atom,AtomQuote)):
            curr_macro.add(atom)
            del base_atom[i]
        elif curr_macro and is_tok(atom,Tok.MACROHEAD) and get_macro_argc(atom.data) == 0:
            curr_macro.add(AtomMacro(atom.data,0))
            del base_atom[i]
        elif curr_macro and is_tok(atom,Tok.MACROHEAD) and get_macro_argc(atom.data) != 0:
            die("You must parenthesize macros of argc>0 when using them as arguments to other macros.")
        elif curr_macro and is_tok(atom,Tok.IDENTIFIER):
            curr_macro.add(atom)
            del base_atom[i]

        # enter new macro creation
        elif not curr_macro and is_tok(atom,Tok.MACROHEAD):
            name = atom.data
            base_atom[i] = AtomMacro(name,get_macro_argc(name))    # overwrite the AtomTok with a new MacroAtom
            curr_macro = base_atom[i]   #save a ref to the new MacroAtom

    if curr_macro and len(curr_macro.data) == curr_macro.argc:
        curr_macro = None
    if curr_macro:
        die("Macro did not receive enough args:"+str(curr_macro))




# returns the string for a call to a macro
# takes macro fn name (without the m_ prefix)
# also takes 'args', a list of arguments. any that are not 'str' will have .gentext() called on them.
def build_call(fname,args):
    fname = fname.replace('?','_qmark')
    if has_special_args(fname):
        args = build_special_args(fname,args) #e.g. (int $1, int $2) in parse
    def text_if_needed(x):
        return x if isinstance(x,str) else x.gentext()
    return 'backend.m_'+fname+'('+','.join([text_if_needed(x) for x in args])+')'

# used in build_call() for macros like %parse that have arguments that shouldn't be evaluated directly, like the idx/cast list of %parse.
def build_special_args(fname,args):
    if fname in ['parse','parselines','parselines1']:
        idx_casts_tkns = args[1] # idx_casts_tkns = [ID DOLLAR COMMA ID DOLLAR COMMA ID DOLLAR COMMA ID DOLLAR COMMA]
        idx_casts_tkns.rm_whitespace()
        if len(idx_casts_tkns)%3 != 2:
            die("build_special_args() failed. Len of (int $1, int $2, ...) mod 3 should be 2 always (eg ID DOLLAR COMMA ID DOLLAR). Offending token list:{}".format(idx_casts_tkns))
        atomiter = iter(idx_casts_tkns)
        result = []
        while True:
            try:
                id = next(atomiter)
                dollar = next(atomiter)
                if not is_tok(id,Tok.IDENTIFIER): die("build_special_args() for parse/parselines/parselines1 failed. first item of every three in (int $1, int $2,...) should be an IDENTIFIER. (note: 3rd item=comma)")
                if not isinstance(dollar,AtomDollar): die("build_special_args() for parse/parselines/parselines1 failed. second item of every three in (int $1, int $2,...) should be an AtomDollar. (note: 3rd item=comma)")
                try:
                    idx = int(dollar.data)
                except:
                    die("unable to cast idx in parse/parlines/etc's format string. failed on:int({})".format(dollar.data))
                try: #this try-except is not necessary, tho its good bc it replaces a runtime error with a compiletime error even tho .__name__ shows up later which undoes this work.
                    cast = eval(id.data)
                except:
                    die("unable to eval identifier.data in parse/parlines/etc's format string. The program attempts to do eval('int') to yield the int() function to use as a caster. Failed on:eval({})".format(id.data))
                result.append((idx,cast.__name__))
                next(atomiter) # consume the comma
            except StopIteration:
                break
        args[1] = str(result)
        return args
    die('Failed to match on fname={} in build_call_special_args()'.format(fname))


# the main function that runs the parser
# It goes string -> Token list -> Atom list -> Atom list (w MacroAtoms) -> final python code
def parse(line,globals,debug=False):
    if debug:
        red("========================="*3)
        red("=Input=")
        print(line)
    tkns = tokenize(line)
    #tkns = sanitize_tkns(tkns)
    if debug:
        red("=Tokens=")
        print(tkns)
    a = atomize(tkns,globals)
    if debug:
        red("=Atoms=")
        print(a)
    #macroize(a)
    #if debug:
    #    red("=Atoms Post Macroization=")
    #    print(a)
    out = a.gentext()
    if debug:
        blue(out)
    return out


#parse("fname,linect = %parse sh{wc -l $file} (str $1, int $2)")
#parse("z = %parselines1  x (str $1, int $2)")
#parse("sh{echo \"hi \"$there\" you're the \"$one}")
#parse("if %exists? filename:")
#parse("vi_list = %parselines1 (%cat file) (int $1, int $2)")




