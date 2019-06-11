# CODEGEN
# OVERVIEW:
# class Tok: an enum of the possible tokes, eg Tok.WHITESPACE
# class Token: a parsed token. has some associated data
# class AtomCompound, etc: These Atoms are AST components. Each has a .gentext() method that generates the actual final text that the atom should become in the compiled code
# parse() is the main function here. It goes string -> Token list -> Atom list -> Atom list (w MacroAtoms) -> final python code
from collections import namedtuple
from keyword import kwlist
keywords = set(kwlist)



# ===TODO===
## WHERE I LEFT OFF
# Continue at implementing `Subscript` and `Slice`, then implement the rest of the Expr nodes.
# Precedence table. Note that the ',' precedence should avoid issue of tuple recursion
# Rewrite stmt() to loop over a list of Nodeclasses similar to how expr does, except it's much easier bc you can use the __init functions rather than .build. Just make Stmt have a .identify, 
# You'll need to write token() keyword() etc etc. Since they have v similar input types (str or token constant, list or single elem, fail=..., you should share a lot of their similarities in helper methods (prefix with _ underscore since shouldnt be called outside token/keyword/etc)
# That's pretty much it i think, then you just gotta get it to compile. Do a first pass thru with ALE to check for common errors that might be inside if-statements and wont show up right away when running.
# Dont forget to see the `MANY amazing es ideas` note on phone!
# 
# 
## 

# ===TODO===
## Type inference
# First construct namespaces by figuring out what every ident refers to. A Namespace is a (Bind -> Value) dict. Any Bind  As you'll see later a Value is like a Node but parallel to it.
    # Vars, but also things like Args, Formals, Etc etc. To make this easier you should really have a class for a Bind and have Var.val = Bind,  Arg.val = Bind etc. A bind is NOT a pointer to a Value bc changing the bind in one place changes it for all futures. Therefore a bind is a pointer to a Namespace, and you can do bind.resolve() or something to get whatever the Value result of the bind looking itself up in the namespace is.
#
# Then we run the program in branching-exec mode
# See all the notes on a piece of paper in colored pen, and phone, and more.
#
#
#
#
#
#
## 




# TODO you frequently call token(elems,'*',fail=BOOL) for an if-statement followed by the exact same thing again, which ends up being the same as elems = elems[1:]. You could add a fn to do this for you, or actually it's more clear to just say elems = elems[1:] directly. You should probably switch to doing that as a compromise of readability and speed

#TODO fig out missing colon errors ahead! You prob have to do it during the atomize step while handling an IndentationError

#TODO dont spend any more time on LocTagged


#TODO: currently comment lines are included in StmtLists, and trailing comments on the ends of lines are also included in the lines. In both cases `AComment` is used. I suppose this is all good because it's useful to have comments in the AST for future projects.
#TODO: newline escaping with a \ as the final character of a line


#TODO should prob have py{} and (py ) to mirror the sh ones.
#TODO should also have sh''' ... ''' syntax for truly verbatim sh
#TODO hmm think all this thru before doing it. Prob want just one syntax really.


#TODO be careful to use .verbatim on the actual QUOTE Tok that opened a verbatim quote block bc that .verbatim includes whatever trailing whitespace it had, which is an important part of the actual quote.


# TODO in order to get debug info on line numbers for an unfinished atom just call .finish() then the .loc will be populated. So you prob wanna do this kind of thing during exception handlers for useful debug info

# kill all lines with comments (dedent level does NOT ever effect anything). Note that line numbers are still important.
# Tokenize by regex -> [Tok] *modify to include line numbers
# Atomize by a linear pass -> [Atoms] *include line numbers
#   "",'',(),{},[]
#   Additionally colons ':' and commas ','
#   Colons that are NOT at the end of a line group till the end of the line.
#   Colons that ARE at the end of the line group till the next line with nonwhitespace dedented to the right amt or even more dedented
#   Commas are the only divider within a paren group and nothing beats them in precedence
#   Flag all keywords at the same time, and save pointers to all instances of each of them in a dict
#   Throw out all newlines within (), {}, []. Now every line is a statement.
#   Colon groups are lists of statements, and they are the ONLY list of statements besides the master list
#   Semicolons merely become newlines with same indent level


# Atomization pass: takes a token list and outputs an atom tree, which is basically like a very coarse AST that just has broad structural info



def tokinfo(tok):
    return f"'{tok.verbatim}' at {tok.loc.start.line}:{tok.loc.start.char}"

def atomize(tokenized_lines, interpreter):
    if atomize.saved_stack is not None:
        stack = atomize.saved_stack
        atomize.saved_stack = None
    else:
        stack = [AMasterList()]

    def pop():
        stack[-1].finish()
        stack[-2].add(stack.pop())

    for linedata in tokenized_lines:
        if linedata.linetkns == []:
            continue # ignore lines that are purely comments or empty
        # Handle any dedenting
        while stack[-1].linestart_trypop(linedata) is True:
            pop()
        for tok in linedata.linetkns:
            #u.gray(tok)
            if tok.typ == stack[-1].closer: # `closer` must be defined by each atom
                pop()
            elif tok.typ in stack[-1].error_on:
                raise SyntaxError(f"Invalid token {tokinfo(tok)} when parsing contents of atom {stack[-1]}")
            elif tok.typ not in stack[-1].allowed_children: # `allowed_children` must be defined by each atom
                stack[-1].add(tok) # if we aren't allowed to transition on a token we just add the raw token
            elif tok.typ in [QUOTE1,QUOTE2]:
                stack.append(AQuote(tok, isinstance(stack[-1],ASH)))
            elif tok.typ in [SH_LBRACE,SH_LPAREN,SH]:
                stack.append(ASH(tok))
            elif tok.typ == LBRACE:
                stack.append(ABrace(tok))
            elif tok.typ == LPAREN:
                stack.append(AParen(tok))
            elif tok.typ == LBRACKET:
                stack.append(ABracket(tok))
            elif tok.typ == COLON:
                stack.append(AColonList(tok,linedata,stack[-1].pop_curr_stmt()))
            #elif tok.typ == COMMA:
            #    stack.append(ACommaList())
            elif tok.typ == PYPAREN:
                stack.append(ADollarParen(tok))
            #elif tok.typ == HASH:
                #stack.append(AComment())
        # End of line. Handle ASH newline pops and any postprocessing on lines like inserting \n tokens if needed
        while stack[-1].lineend_trypop(linedata) is True: # any necessary end of line processing
            pop()
    # Now we autodedent plenty
    while len(stack) > 1 and isinstance(stack[-1],AStmtList):
        pop()
    if len(stack) > 1:
        if interpreter:
            atomize.saved_stack = stack
            print(stack)
            return INCOMPLETE
        else:
            raise SyntaxError(f"Unclosed {stack[-1].name} expression. Full stack:\n{stack}")
    stack[0].finish()
    return stack[0]
atomize.saved_stack = None


# Keywords: ^def ^class ^if ^with ^while ^return ^import


# we value features over speed!


#def empty(toklist):
#    if len(toklist) == 0 or (len(toklist) == 1 and getattr(toklist[0],'typ',None) == WHITESPACE):
#        return True # note the None in getattr makes it false even if this is an Atom instead of a Tok bc the .typ field is missing
#    return False

class Atom(LocTagged):
    def __init__(self,start_tok):
        super().__init__()
        self.body = []
        self.start_tok = start_tok # may be None e.g. for AMasterList
        # pretty name for class
        classstr = str(self.__class__)
        self.name = classstr[classstr.index('A'):classstr.index('>')-1]
    def add(self,tok):
        assert not self.finished # we don't want any additions to a finished atom, that will mess up line numbers and such
        self.body.append(tok)
    def linestart_trypop(self,linedata):
        return False # does nothing for most atoms
    def lineend_trypop(self,linedata):
        self.add(linedata.newline_tok)
        return False
    def finish(self):
        """
        Figuring out our start and end locations in the original text. Start location is based on the token that created us for Atoms that aren't AStmtLists, for AColonList it's the start of the head, for AMasterList it's line 1 char 1.
        """

        # loc.start
        if isinstance(self,AStmtList):
            self.finish_stmt() # finish stmt, this is our last chance
            if isinstance(self,AColonList):
                self.loc_start(self.head[0])
            if isinstance(self,AMasterList):
                self.loc_start(Loc(1,1))
        else:
            self.loc_start(self.start_tok)

        # loc.end
        if len(self.body) > 0:
            self.loc_end(self.body[-1])
        else: # empty body (rare case)
            if isinstance(self,AColonList):
                self.loc_end(self.start_tok) # the ':'
            if isinstance(self,AMasterList):
                self.loc_end(Loc(1,2))
            else: # non AStmtList
                self.loc_end(self.start_tok)

        self.finished = True # must come after self.finish_stmt()
    def __repr__(self):
        body = ' '.join([repr(elem) for elem in self.body])
        return u.mk_b(f"{self.name}(") + f"{body}" + u.mk_b(")")

class AStmtList(Atom):
    def __init__(self,tok):
        super().__init__(tok)
        self.curr_stmt = []
        self.closer = None # only gets closed by the linestart_trypop function
    def finish_stmt(self):
        assert not self.finished
        if len(self.curr_stmt) > 0: # we ignore blank lines and pure comment lines
            self.body.append(self.curr_stmt)
            self.curr_stmt = []
    def add(self,tok):
        if getattr(tok,'typ',None) is SEMICOLON: # this will drop to 'else' case for non Toks like LPAREN too thanks to the None
            self.finish_stmt()
        else:
            self.curr_stmt.append(tok)
    # so that AColonList can steal its lhs from us if it wants
    def pop_curr_stmt(self):
        tmp = self.curr_stmt
        self.curr_stmt = []
        return tmp
    def lineend_trypop(self,linedata):
        self.finish_stmt()
        return False
    def __repr__(self):
        return self.repr_depth(depth=0)
    def repr_depth(self,depth=0):
        head_indent = '  '*depth
        block_indent = '  '*(depth+1)
        res = f'{head_indent}{self.name}'
        if hasattr(self,'head'):
            res += '(' + ' '.join([repr(elem) for elem in self.head]) + ')'
        res += ':'
        for line in self.body:
            res += f'\n'
            if len(line) == 1 and isinstance(line[0],AStmtList):
                res += line[0].repr_depth(depth=depth+1) # print stmt list
            else:
                res += f'{block_indent}' # only need for nonstmtlists bc stmtlists handle indents themselves
                for elem in line: # print a line full of non-stmtlist elements
                    assert not isinstance(elem,AStmtList), "AStmtList should only appear as the only element on a line if it appears"
                    res += repr(elem) + ' '
                res = res[:-1] # kill last space
        return res

class AMasterList(AStmtList):
    def __init__(self):
        super().__init__(None) # There is no token that created AMasterList
        self.allowed_children = COMMON_ALLOWED_CHILDREN | {COLON} # set union
        self.error_on = COMMON_ERROR_ON
    def linestart_trypop(self,linedata):
        if linedata.leading_whitespace != '':
            raise IndentationError(f"unexpected indent: should have zero indent")
        return False # never pops

SEMICOLONS_ONLY = -1

class AColonList(AStmtList):
    def __init__(self,tok,linedata,head):
        super().__init__()
        self.prev_leading_whitespace = linedata.leading_whitespace # this is the indent on the line ending with a ':', so it's not actually the indent level of the block
        self.leading_whitespace = None
        self.head = head
        self.allowed_children = COMMON_ALLOWED_CHILDREN | {COLON}
        self.error_on = COMMON_ERROR_ON
    def linestart_trypop(self,linedata):
        if self.leading_whitespace is SEMICOLONS_ONLY:
            return True # pop! even if a comment is approaching, just pop

        # This will fire on every new line until the first real line of code is hit
        if self.leading_whitespace is None:
            if len(linedata.leading_whitespace) <= len(self.prev_leading_whitespace):
                raise IndentationError(f"unexpected indent: not enough indent / empty colon block")
            if len(self.body) > 0: # this means something of the form `def foo(): return 1` where there was a statment on the same line as the colon and no newline.
                self.leading_whitespace = SEMICOLONS_ONLY # in this case only semicolons can be used to separate statements and there is NO valid indent level
            self.leading_whitespace = linedata.leading_whitespace # yay, we've determined our indent level
            return False # no pop

        if linedata.leading_whitespace == self.leading_whitespace:
            return False # no pop

        if len(linedata.leading_whitespace) > len(self.leading_whitespace):
            # note that it would be misleading to say 'too much indent' because this actually fires even when the indent level is just some invalid level between two others, as described below.
            raise IndentationError(f"unexpected indent: not an existing indent level")
        return True # pop
        # if the new line is below us we can safely dedent
        # note that if the new line is an invalid level BETWEEN us and the lower valid level, then when the lower valid level (which is necessarily our direct parent since statment lists are only children of other statement lists) receieves this linestart_trypop, it will raise an indentation error. The only reason im doing this is bc i dont wanna pass the list of all current valid indents around, though that would be cleaner so you should do that.



#class AComment(Atom): # useful for future projects
#    def __init__(self):
#        super().__init__()
#        self.closer = NEWLINE # this is just an annotation, NEWLINE is never fed in. We get popped by .lineend_trypop()
#        self.allowed_children = []
#        self.error_on = []
#        def lineend_trypop(self,linedata):
#            return True

class AQuote(Atom):
    def __init__(self, tok, in_sh):
        super().__init__(tok)
        self.closer = tok.typ
        self.in_sh = in_sh # may be useful later e.g. fstrings and more
        self.allowed_children = []
        self.error_on = []

class AParen(Atom):
    def __init__(self,tok):
        super().__init__(tok)
        self.closer = RPAREN
        self.allowed_children = COMMON_ALLOWED_CHILDREN
        self.error_on = COMMON_ERROR_ON
class ADollarParen(Atom):
    def __init__(self,tok):
        super().__init__(tok)
        self.closer = RPAREN
        self.allowed_children = COMMON_ALLOWED_CHILDREN
        self.error_on = COMMON_ERROR_ON
class ABracket(Atom):
    def __init__(self,tok):
        super().__init__(tok)
        self.closer = RBRACKET
        self.allowed_children = COMMON_ALLOWED_CHILDREN
        self.error_on = COMMON_ERROR_ON
class ABrace(Atom):
    def __init__(self,tok):
        super().__init__(tok)
        self.closer = RBRACE
        self.allowed_children = COMMON_ALLOWED_CHILDREN
        self.error_on = COMMON_ERROR_ON

class ASH(Atom):
    def __init__(self,tok):
        super().__init__(tok)
        self.opener = tok.typ
        self.depth = 0
        if typ == SH:
            self.eventual_closer = NEWLINE # note NEWLINE is just an annotation, in reality .lineend_trypop() will pop it
        elif typ == SH_LBRACE:
            self.eventual_closer = RBRACE
        elif typ == SH_LPAREN:
            self.eventual_closer = RPAREN
        self.closer = self.eventual_closer
        self.allowed_children = [PYPAREN,QUOTE1,QUOTE2]
        self.error_on = []
    def add(self,tok):
        super().add(tok)
        if self.opener is SH: # no depth / nesting for full line sh statements
            return
        if tok == self.opener:
            self.depth += 1
            self.closer = None # disable atomize() from popping us until we reassign this to something other than None
        elif tok == self.eventual_closer:
            self.depth -= 1
            if self.depth == 0:
                self.closer = self.eventual_closer
        assert self.depth >= 0
    def lineend_trypop(self,linedata):
        if self.closer == NEWLINE:
            return True
        return False
#class ACommaList(Atom):
#    def __init__(self):
#        self.allowed_children = COMMON_ALLOWED_CHILDREN


# fundamentally {} (and some other syntaxes) are used for expr lists and ':' are used for stmt lists


# check if the elem at index `idx` in statment `stmt` exists AND is a token AND is a keyword AND is the specific keyword `kw`
def iskeyword(stmt,idx,kw):
    if len(stmt) <= idx: # idx doesnt exist in stmt
        return False
    x = stmt[idx]
    if not isinstance(x,Tok): # must be a Tok in order to be a keyword
        return False
    if x.typ != KEYWORD:
        return False
    if x.data != kw:
        return False
    return True

def istoken(stmt,idx,typ_or_list):
    if len(stmt) <= idx: # idx doesnt exist in stmt
        return False
    x = stmt[idx]
    if not isinstance(x,Tok): # must be a Tok in order to be a keyword
        return False
    if isinstance(typ_or_list,list):
        if x.typ not in typ_or_list:
            return False
    else:
        if x.typ != typ:
            return False
    return True

# returns None if no leading keyword
def get_leading_keyword(elems):
    if len(elems) == 0:
        return None
    x = elems[0]
    if not isinstance(x,Tok):
        return None
    if x.typ != KEYWORD:
        return None
    return x.data



def stmt(elems):
    is_compound = (len(elems) == 1 and isinstance(elems[0],AColonList))
    # compound statement
    if is_compound:
        compound = elems[0] # compound is an AColonList
        kw = get_leading_keyword(compound.head)
        if kw is not None:
            if kw == 'def':
                return FuncDef(compound)
            if kw == 'if':
                return If(compound)
            if kw == 'for':
                return For(compound)
            if kw == 'while':
                return While(compound)
            if kw == 'class':
                return ClassDef(compound)
            if kw == 'with':
                return With(compound)
            if kw == 'try':
                return Try(compound)
            if kw == 'except':
                return Except(compound)
            if kw == 'elif':
                return Elif(compound)
            if kw == 'else':
                if get_leading_keyword(compound.head[1:]) == 'if':
                    return Elif(compound)
                return Else(compound)

            raise SyntaxError(f"Improper placement for '{kw}' keyword")
        raise SyntaxError(f"Unrecognized start to a colon block")
    # simple statement
    kw = get_leading_keyword(elems)
    if kw is not None:
        if kw == 'return':
            return Return(elems)
        if kw == 'pass':
            return Pass(elems)
        if kw == 'raise':
            return Raise(elems)
        if kw in ['import','from']:
            return Import(elems)
        if kw == 'break':
            return Break(elems)
        if kw == 'continue':
            return Continue(elems)
        if kw == 'del':
            return Delete(elems)
        if kw == 'assert':
            return Assert(elems)
        if kw == 'global':
            return Global(elems)
        if kw == 'nonlocal':
            return Nonlocal(elems)
        raise SyntaxError(f"Improper placement for '{kw}' keyword")
    # simple statements that don't start with keywords
    e, rest = expr(elems)
    # ExprStmt
    if rest == []:
        return ExprStmt(elems) # to make a .identify for this probably just have it run .expr_assert_empty with fail=BOOL or whatever
    # Asn
    if istoken(rest,0,EQ):
        return Asn(elems)
    # AugAsn
    if istoken(rest,0,BINOPS) and istoken(rest,1,EQ):
        return AugAsn(elems)
    raise SyntaxError(f"Unrecognized statment where first token is not a keyword and after parsing the first section as an expression the first token of the remainder is not EQ nor in BINOPS followed by EQ so it can't be Asn nor AugAsn.\nFirst expr parsed:{e}\nRest:{rest}")


def expr_assert_empty(*args,**kwargs):
    *ignore,elems = expr(*args,**kwargs)
    empty(elems)
    return e

"""
Returns the first valid expression that starts at elems[0] and is not the subexpression of a larger expression in `elems`, and also returns the remains of `elems`

note: don't worry if you see stuff like ABracket being translated into a dict always. That's only what happens when we ask for an expression and it START with {}. You can still make your own nondict syntax using {} as long as you don't start a generic expr with the '{' symbol. If you do want to override the kinds of things that can happen when an expr starts with '{' you are also welcome to do that, just modify the code to look harder at it and decide between Dict() and whatever your AST node is.

"""


# class decorator
def left_recursive(cls):
    cls.left_recursive = True
    return cls

# class decorator
# defines a gathers() method that returns true if called with one of the classes originally passed into the decorator (in `gatherable_classes`)
def gathers(*gatherable_classes):
    def aux(cls):
        cls.gathers = (lambda other_cls: other_cls in gatherable_classes)
    return aux

class Expr(Node):
    left_recursive = False
    gathers = lambda x: False

    @staticmethod
    def identify(elems):
        """
        Takes [elem] and returns True if
        """
        raise NotImplementedError
    @staticmethod
    def build(elems): # left-recursive nodes take an additional parameter lhs_node
        raise NotImplementedError
    @staticmethod
    def finish(**finish_kwargs):
        pass
#    @staticmethod
#    def targetable():
#        raise NotImplementedError


# crfl this should only have the master classes in it like Binop not Add
exprnodes []



"""

finish_kwargs is only passed to the .finish() method of the outermost expr. Unfortunately we don't know which is the outermost expr until we've already build()ed it, hence we have this finish() system. We do NOT pass any kwargs to build() bc we fundamentally do not know the context for the expression we're building until after we've built it and built its left-recursive parent etc etc. So when our caller calls expr() and passes it some info, that info should be sent to the .finish() method of the outermost expression that's being returned, and it's up to the finish method to do whatever is required (which may involve recursing down its children)



if `till` is provided it can be a seq of strs | functions | TOKENs and optionally end with a CONSTANT. If it would just be a single element seq then it doesn't need to be wrapped in a seq.

till_function converts a till-list to a boolean function taking `elem` that returns true if the till condition fires

"""

def expr(elems, *, leftnodeclass=None, till=None, finish_kwargs=None, no_tuple=False):
    finish_kwargs = dict() if finish_kwargs is None else finish_kwargs # can't use {} or dict() as a default argument or that one dict gets shared among all calls to expr() which is horrible.

    till = till_function(till)

    identified = []
    for nodeclass in exprnodes:
        if not nodeclass.left_recursive:
            subclass = nodeclass.identify(elems)
            if subclass is not None:
                identified.append(subclass)

    if len(identified) > 1:
        raise NotImplementedError("{identified}\n{elems}") # if multiple appear to conflict we just go down both paths and throw out whichever one yields an Error. If neither yields an Error then we raise an Error. This could happen (in a recoverable way) if two language extensions were written by different people and had different syntax but similar enough syntax that the .identify test passed (e.g. they wrote same fairly minimal .identify function). We should probably also Warn people even when this does pass fine.

    # Parens: If we see parens at the start of an expr that just means the leftmost subexpr must be the result of expr_assert_empty on the paren contents
    if isinstance(elems[0],AParen):
        # (1==1)==1 is not same as 1==1==1
        # (a,),b is not same as a,b so we need to be careful abt this stuff esp in conjunction for the build() methods of Tuple and Compare
        assert len(identified) == 0
        node = expr_assert_empty(elems[0].body) # most keywords dont wanna be passed in here
        elems = elems[1:]
    elif len(identified) == 0:
        raise NotImplementedError("{elems}") # the whole "eval empty expr to None" thing only applies in certain cases like Return statements so it shouldn't be a general thing. Also there's a lot of annoying error checking you have to do if you need to make sure expr didn't return None. So perhaps it should raise an error. And callers should use a wrapper like something similar to the expr_assert_empty() one.
    else: # COMMON CASE
        nodeclass = identified[0]
        print(f"identified node class {nodeclass} for elems: {elems}")
        node,elems = nodeclass.build(elems)
        print(f"built node {node}\nremaining elems:{elems}")

    # left-recursion to extend this lefthand expression as much as possible
    identified = []
    while True:
        if len(elems) == 0:
            break
        if till(elems):
            break
        for nodeclass in exprnodes:
            if nodeclass.left_recursive:
                subclass = nodeclass.identify(elems)
                if subclass is not None:
                    identified.append(subclass)
        if len(identified) > 1:
            raise NotImplementedError("{identified}\n{elems}") # Same issue as above
        if len(identified) == 0:
            break
        rightnodeclass = identified[0]
        prec = precedence(leftnodeclass,rightnodeclass)
        if prec is LEFT or prec is EQUAL:
            break # left op is more tight so we return into our caller as a completed subexpr. Our caller will very quickly be finding this rightnodeclass again (unless kwargs somehow change things ofc)
            # all/most left recursive grammars associate such that the thing on the left binds more tightly so EQUAL precedence behaves like LEFT
        if prec is GATHER: # TODO rn gather is jank but it works. It's fine for now, bigger fish to fry.
            node._gather = True
            return node,elems
        # prec is RIGHT
        print(f"identified (left-recursive) node class {nodeclass} for elems: {elems}")
        node,elems = rightnodeclass.build(node,elems) # build a larger left-recursive expr `node` from our original subexpr `node`
        print(f"built (left-recursive) node {node}\nremaining elems:{elems}")

    return node,elems




#def expr(elems,usage=None,leftop=None):
#    elem = elems[0]
#    e = None
#    # ATOMS
#    if isinstance(elem,Atom):
#        if isinstance(elem,AQuote):
#            e = Str(elem)
#        elif isinstance(elem,AParen):
#            if len(elem.body) == 0:
#                e = Tuple([],usage=usage)
#            else:
#                e = expr_assert_empty(elem.body,usage)
#        elif isinstance(elem,ABracket):
#            e = List(elem,usage=usage)
#        elif isinstance(elem,ABrace):
#            e = braced_expr(elem)
#        elif isinstance(elem,ASH):
#            e = Sh(elem)
#        elif isinstance(elem,ADollarParen):
#            e = EmbeddedPy(elem) # EmbeddedPy can just do expr_assert_empty(elem.body)
#        else:
#            raise NotImplementedError("{elem}")
#    # TOKENS
#    else:
#        # LITS AND VARS
#        if token(elem,[INTEGER,FLOAT,COMPLEX]):
#            e = Num(elem)
#        elif token(elem,ID):
#            e = Var(elem,usage=usage)
#        elif keyword(elem,['True','False','None']):
#            e = NamedConstant(elem)
#        # Anything we can figure out from the token(s) starting an expression
#        # (or any further analysis. It just happens that we only need the very first token
#        # to determine all of these, and anything that can't be determined by the first token
#        # happens to begin with an expr as the leftmost item. If any constructs did not begin
#        # with an expr nor keyword nor unique token then we would handle it here by doing
#        # further analysis on `elems`
#        elif token(elem,'*'):
#            e,elems = Starred.init(elems,usage=usage) # note that more than just `elem` is needed here
#        elif token(elem,UNOPSL):
#            e,elems = UnopL.init(elems) # more than just first elem needed
#        elif keyword(elem,'lambda'):
#            e,elems = Lambda.init(elems)
#        elif keyword(elem,'yield'):
#            e,elems = Yield.init(elems)
#        else:
#            raise NotImplementedError("{elem}")
#    """
#    To this point we have parsed everything short of left-recursive expression, in particular:
#        > Binop, Boolop, Compare, Call, Ternary, Attr, Subscript, ListComp
#    However left-recursions have a fatal flaw that makes them easy to capture: they can't recurse forever, and eventually their lefthand expr must be one of the expressions we've already parsed.
#    Furthermore, at this point the only way to parse a larger expression is through a left-recursive expression, because we already have used a parser to produce an expression `e` and therefore in order to extend this expression the extended expression must have the Expr `e` as its leftmost component and thus by definition the extended expression is left-recursive.
#    """
#
#    raise NotImplementedError # TODO at this point trim `elems` to have dealt with consuming whatever was needed to make `e`
#
#
#    #TODO fig out how to handle returning `elems` since __init__ can't do it, and to indicate how much of elems has been eaten
#
#
#    # left-recursive
#    while True:
#        e_lhs = e # using the old `e` as the lhs
#        if len(elems) == 0:
#            break
#        elem = elems[0]

        # You have a complete subexpression `e_lhs`. You see a comma. This means that you need to build a tuple. This belongs in the `while` loop bc Tuples are effectively left-recursive so they could show up at any point once you have a subexpression.
#        if token(elem,','):
#            if no_tuple_recursion:
#                return e,elems # return, with leading comma included so Tuple.init knows there's more to come.
#            else:
#                e_lhs,elems = Tuple.init(e_lhs,elems) # completely construct tuple. This completes a sub-expression and we can then move into more left-recursive items to extend it as usual.
#
#        # ATOM
#        if isinstance(elem,Atom):
#            if isinstance(elem,AParen):
#                op = Call
#            elif isinstance(elem,ABracket):
#                op = Subscript
#            else:
#                raise NotImplementedError("{elem}")
#        # TOKEN
#        else:
#            if token(elem,BINOPS):
#                rightop = Binop
#            elif token(elem,BOOLOPS):
#                rightop = Boolop
#            elif token(elem,CMPOPS):
#                rightop = Boolop
#            elif keyword(elem,'if'):
#                rightop = Ternary
#            elif token(elem,'.'):
#                rightop = Attr
#            elif keyword(elem,'for'):
#                rightop = Comprehension
#            elif token(elem,','):
#                rightop = Tuple
#            else:
#                break # the very important case where the rest of elems is not useful for constructing an extended Expr
#        # found an `op`
#        rightop._typ = elem # needed by `precedence`
#        if precedence(leftop,rightop) is LEFT:
#            break # left op is more tight so we return into our caller as a completed subexpr
#        else:
#            kw = {'usage':usage} if isinstance(rightop,Targetable) else {}
#            e = rightop.init(e_lhs,elems,leftop=rightop, **kw)
#
#    return e,elems


# takes an operator to your left and an operator to your right and tells you which one applies to you first (binds more tightly). The None operator always loses. `left` and `right` are Node classes


# For clarity everything must be written as the most exact subclass, not any superclasses like UnopL. Also this is good if people add more operators that subclass existing groups but with different precedence than is normal for the group. Easier to do this than to make a mechanism for enabling that.
precedence_table = [
[Lambda], # Tightest binding
[Ternary],
[Or],
[And],
[Not],
[Compare], # normally we do NOT allow superclasses in this list. Note that Compare is NOT a superclass as it is actually the node type that gets used, and Neq/Is/In/etc are not classes as they can be chained together in expressions like x <= 1 == 2 where the whole expression can't rightfully be called anything other than a Compare (unless a CompoundCompare or something were introduced). Calling it Compare forces everyone to deal with this unusual case explicitly which is good probably, unless it makes it unreadable for any reason.
[BitOr],
[BitXor],
[BitAnd],
[ShiftL, ShiftR],
[Add, Sub],
[Mul, MatMul, Div, FloorDiv, Mod],
[UAdd, USub, Invert],
[Exp],
[Await],
[Subscript, Slice, Call, Attr],
[Tuple, List, Dict, Set], # Weakest binding
]

LEFT = CONSTANT()
RIGHT = CONSTANT()
SAME = CONSTANT()
GATHER = CONSTANT()

def precedence(leftnodeclass,rightnodeclass):
    if leftnodeclass is None:
        return RIGHT
    if leftnodeclass.gathers(rightnodeclass):
        return GATHER
    left_tier = None
    right_tier = None
    for tier,classes in enumerate(precedence_table):
        if leftnodeclass in classes:
            left_tier = tier
        if rightnodeclass in classes:
            right_tier = tier
        if left_tier is not None and right_tier is not None:
            break
    # higher tier is weaker binding
    assert left_tier is not None and right_tier is not None, f"{leftnodeclass,rightnodeclass}"
    if left_tier < right_tier:
        return LEFT
    if left_tier > right_tier:
        return RIGHT
    if left_tier == right_tier:
        return SAME




# TODO note that the .usage of things often isn't known when they're created initially, and is rather added during left-recursion when the target becomes part of a larger statement. So it should really be up to the larger statement to update the .usage for its targets. In other cases it is known when you call expr() for example when already inside a larger statement and calling expr to construct a nonleftrecursive smaller expr. Also ExprStmt for example could set things to LOAD, etc.



"""
The issue with Tuples, and the solution:
    The Tuple constructor must necessarily call expr() somewhat recursively in order to parse the expressions that make it up.
    An initially promising idea is to have the Tuple.init(e_lhs,elems) function call expr(elems) to recurse just as Binops, Boolops, and all other left-recursive forms have been doing. Say we're parsing expr("x,y,z") then the first call would be Tuple.init("x",",y,z") which would result in a new expr call expr("y,z"), which would subsequently yield an expr("z") after yet another Tuple.init. It's obvious that expr("y,z") should return a Tuple and expr("z") should return a Var. So logically Tuple.init() could use the result of its expr() call to A) if the result is a tuple then prepend whatever your lhs is onto the Tuple to create a tuple of slightly larger size and B) if the result is not a tuple then for a new 2-tuple from your lhs and the expression result. The issue with this: what if the input was expr("x,y,(a,b)")? Then the expr("a,b") call would return a Tuple whic would get merged with the lhs ("y") to form a 3-tuple y,a,b.
    The solution: add a flag to expr so that it doesn't recurse on tuple-building. Yep it's lame, really the interesting bit was the way that the initially promising idea was wrong.
"""




"""
The following method is guaranteed to work for binary and unary operators of the form:
    | Expr op Expr # binop
    | Expr op      # right unop
    | op Expr      # left unop
Following the example `x + 5*z.y - 3`
1. Read elems until you have the smallest possible valid Expr e that starts from the first elem
    e=`x`. We may be fed an operator `prevop` by our caller. If we aren't then `op` is None.
2. Check if the next elem(s) indicate that e could be a lhs subexpression
    `+` is part of the rule [Expr:==Expr '+' Expr]
    Additionally `+` isn't part of any other syntax that has an lhs Expr immediately to its left
3.1 If False simply return e.
3.2 If True and `op` is lower precedence than `prevop`, return e
3.3 If True and `op` is higher precedence than `prevop` (or prevop is None), call expr() on the remaining expression and your final result is the the result of unify(e,op,expr())
*if at any point you fail, simply return the last valid expression you had that starts at the first elem

Non unop/binop expressions:
    | various literals / vars -> trivial
    | Lists/tuples/sets/dicts -> easy recognize, known exact Expr locs
    | comprehension -> easy recognize, known exact Expr locs
    | ternary -> easy recognize, known exact Expr locs
    | lambda -> easy recognize, known exact Expr locs
    | yield -> easy recognize, known exact Expr locs


"""





    # Tuple
    # Note: An empty pair of parentheses yields an empty tuple object
    # A trailing comma creates a 1-tuple
    # a lack of commas yields the expr (non tuple)


"""

Var List Tuple Starred Attr Subscript should assert that .usage is not None when provided


Non unop/binop expressions:
    | various literals / vars -> trivial
    | Lists/tuples/sets/dicts -> easy recognize, known exact Expr locs
    | comprehension -> easy recognize, known exact Expr locs
    | ternary -> easy recognize, known exact Expr locs
    | lambda -> easy recognize, known exact Expr locs
    | yield -> easy recognize, known exact Expr locs

commas look out for. They turn you into a tuple with implied parens i think. In fact in the case of parens starting an expression i literally just call expr() on aparen.body and leave it up to expr() to return me a tuple if appropriate

"""


"""

===Stmts===
FuncDef:
    .name: str
    .args: Formals
    .body: [Stmt]
    .decorators: [Var]
If:
    .cond: Expr
    .body: [Stmt]
    .elifs:[Elif]
    .else: Else # empty list if no else
Elif: # only present inside If
    .cond: Expr
    .body: [Stmt]
Else: # only present inside other stmts
    .body: [Stmt]
For: for i
    .target: [Target] # specifically: Var | Tuple | List
    .iter: Expr
    .body: [Stmt]
    .else: [Stmt] # empty if no else
While:
    .cond: Expr
    .body: [Stmt]
    .else: [Stmt] # empty if no else
Return:
    .val: Expr
ClassDef:
    .name: str
    .bases: [Var] # base clases to inherit from
    .keywords: [Keyword] # e.g. the 'metaclass' keyword
    .body: [Stmt]
    .decorators: [Var]
Raise:
    .exception: Expr | None
    .cause: Expr | None # where Expr is another exception.
      # (niche, prints an error message that blames .exception happening
      # on the fact that the prior .cause exception happened)
Import:
    .importitems: [Importitem]
    .from: str | None
    .numdots: int
      # from ..a import b          #.val=[(.var=b,.alias=None)],                  .from='a'  .numdots=1
      # from ... import b as c, d  #.val=[(.var=b,.alias=c),(.var=d,.alias=None)] .from=None .numdots=3
      # import b                   #.val=[(.var=b,.alias=None)],                  .from=None .numdots=0
  Importitem(namedtuple)
      .item: str
      .alias: str | None
Break:
    [no fields]
Pass:
    [no fields]
Continue:
    [no fields]
Delete:
    .targets: [Reference]
Assert:
    .cond: Expr
    .expr: Expr # if the assertion fails then AssertionError(.expr) is run, so .expr should return a string but could also be a generic function that has side effects then returns the string
Try:
    .body: [Stmt]
    .excepts: [Except]
    .else: [Stmt] # empty if none
    .finally: [Stmt] # empty if none
Except: # only ever present inside a Try node
    .type: Var | None # None is catch-all
    .name: str | None
    .body: [Stmt]
Global:
    .names: [str]
Nonlocal:
    .names: [str]
With:
    .withitems: [withitem]
    .body: [Stmt]
      # with a as b, c as d, e
  Withitem(namedtuple):
      .contextmanager: Expr
      .targets: Reference # specifically Var | Tuple | List
        # the result of evaluating .contextmanager is assigned to .variables or something i think (if they arent None)
ExprStmt:
    .expr: Expr
Asn:
    .targets: [Reference]
    .val: Expr
AugAsn:
    .target: Name | Subscript | Attribute (not Tuple | List)
    .op: BINOP
    .val: Expr
Module:
    .body: [Stmt]


===Exprs===
Num: super of Int Float Complex
Int:
    .val: int
Float:
    .val: float
Complex:
    .val: complex
Str:
    .val: str
Bytes:
    .val: bytes
List:
    .vals: [Expr]
    .usage: USAGE
Tuple:
    .vals: [Expr]
    .usage: USAGE
Set:
    .vals: [Expr]
Dict:
    .keys: [Expr]
    .vals: [Expr]
Ellipsis:
    [no fields]
      # for the '...' syntax used extremely rarely in indexing
NamedConstant:
    .val= None|True|False
Var:
    .name: str
    .usage: USAGE
Starred:
    .val: Var | some other options
    .usage: USAGE
Unop:
    .op: UNOP
    .val: Expr
Binop:
    .op: BINOP
    .left: Expr
    .right: Expr
Boolop:
    .op: BOOLOP
    .vals: [Expr]
      # `a or b or c` is collapsed into one Boolop
Compare:
    .ops: [CMPOPS]
    .vals:[Expr]
    # e.g. `a > b > c`
Call:
    .func: Name | Attr | some other stuff
    .args: [Expr]
    .keywords: [Keyword]
  Keyword(namedtuple):
      .name: str
      .val: Expr
Ternary:
    .cond: Expr
    .if_branch: Expr
    .else_branch: Expr
Attr:
    .expr: Expr
    .attr: str
    .usage: USAGE
Subscript:
    .expr: Expr
    .index: Expr
    .usage: USAGE
Slice:
    .expr: Expr
    .start: Expr
    .stop: Expr
    .step: Expr
    .usage: USAGE
Comprehension:
    .expr: Expr
    .comprehensions: [Comprehension]
      # sadly [ord(c) for line in file for c in line] is valid Python
  Comprehension(namedtuple):
      .target: Name | Tuple | other things
      .iter: Expr # the thing we iterate over
      .conds: [Expr] # can have mult in a single comprehension
Lambda:
    .args: [Arg] # cant take kwargs etc i think
    .body: Expr # i may extend this to allow same-line statements.
Arg(Dependent):
    .name: str
Args(Dependent):

Formals(Dependent):
    .args: [Arg]
    .defaults: [Expr] # specifically corresponding to the last len(.defaults) positional args (bc disallowed to put any defaultable args before required args)
    .stararg: Arg # *arg. It's an Arg('') for a raw '*' or an Arg('name') for '*name', and it's None if no star is present at all.
    .doublestararg: Arg  # **kwargs
    .kwonlyargs: [Arg]
    .kwdefaults: [Expr] # for kwonlyargs. if one is None then that kwarg is required. Unclear what non-kwonly defaults are...
      # def f(a, b=1, c=2, *d, e, f=3, **g)
      # a,b,c -> .args
      # 1,2 -> .defaults # note how defaults[i] corresponds to args[i+1] here, but more useful: defaults[-i] corresponds to args[-i]
      # d -> .vararg
      # e,f -> .kwonlyargs
      # 3 -> .kwdefaults # same deal, kwonlyargs[-i] corresponds to kwdefaults[-i]
      # g -> .kwargs
Yield: # yes, this is actually an expression
    .val: Expr  # value to yield or generator to yield from
    .from: bool # whether this is a `yield from` expression


USAGE = LOAD | STORE | DEL


Target = Var | Attr | Subscript | Slice | '*' Target
Reference = Var | Attribute | Subscript


"""

Pos = namedtuple('Pos', 'line char')
# these numbers are inclusive as start* and exclusive as end*
# so endchar is one past the last char
Loc = namedtuple('Loc', 'start end')




class LocTagged:
    def __init__(self):
        self.loc = Pos(Loc(None,None),Loc(None,None))
        self.finished = False
    def loc_start(self,other):
        if isinstance(other,Pos):
            self.loc = Loc(other,self.loc.end)
            return
        assert isinstance(other,LocTagged)
        assert other.finished
        self.loc = Loc(other.loc.start,self.loc.end)
    def loc_end(self,other):
        if isinstance(other,Pos):
            self.loc = Loc(self.loc.start,other)
            return
        assert isinstance(other,LocTagged)
        assert other.finished
        self.loc = Loc(self.loc.start,other.loc.end)
    def finish():
        # must override in subclasses to properly set `.loc` in stone
        # And you must have this at some point in the finish call:
        #   self.finished = True
        raise NotImplementedError

# assert_contiguous was thrown out bc things like Comments make stuff non contiguous
# also it's really not worth spending this much time on location stuff when the vast majority of editing will be done in the AST world.
#    @staticmethod
#    def assert_contiguous(llist):
#        """
#        takes a list of LocTagged items and ensures that there are no location gaps between them, so each one has a loc.start equal to the loc.end of the previous one.
#        """
#        for i in range(len(llist)-1):
#            if llist[i].loc.end != llist[i+1].loc.start:
#                raise LocError(f"Non contiguous LocTagged elements in list: {llist[i].loc.end} != {llist[i+1].loc.start} for \n{llist[i]} \nand \n{llist[i+1]} \nin \n {llist}")
#
#class LocError(Exception): pass


class Node(LocTagged): # abstract class, should never be instatiated
    def __init__(self,partial_stmt):
        super().__init__()
        pass
#    def finish():
#        attrs = list(self.__dict__.keys())
#        for attr in attrs:




# elems is always the first argument
# elems is always the last thing returned (ALWAYS)
def keyword(): pass # no return
def token():
    """
    `query` is an elem or [elem]
        -if [elem] is given then the 0th index elem is used
        -if a single (nonlist) elem is given, then the return type is a bool indicating success/failure. After all, there's nothing useful to pass back to the caller since they didn't give us any list we can modify. This makes for nice if-statements too.
    `tok` is a token CONSTANT() or str
        -if a str is given then it's converted to the right token e.g. '(' -> RPAREN
        (we allow both options since INTEGER for example has no valid str form other than making up an arbirary one which is unnecessary and confusing.)
        -if a list (of CONSTANTs or strs) then matching on one token results in overall success
    `fail` = FAIL | SILENT | None | BOOL
        -Note that in all following descriptions `remains` is the rest of elems after consuming the processed token, and `original` is the original elems input.
        -None (default) - SyntaxError raised on failure. Success returns: remains
        -FAIL - returns (OK,remains) or (FAIL,original)
        -SILENT - returns remains on success and original on failure. No exception raised.
        -BOOL - returns True or False and nothing else
        -*see note in `query` about how passing in a single elem will result in a boolean return type. An error will be thrown on any attempt to specify the fail= option.


    """
    pass



def identifier(): pass
def target(): pass
def contains(): pass


def stmts(elem_list_list): # [[elem]] -> [Stmt]
    the_stmts = []
    for elems in elem_list_list:
        the_stmt = stmt(elems)
        if the_stmt.dependent and the_stmts[-1].offer_neighbor(the_stmt):
            continue # e.g. Try consuming Except by accepting it as a neighbor
        elif the_stmt.dependent:
            raise SyntaxError(f"dependent statement {the_stmt} not accepted by neighbor {the_stmts[-1]}")
        the_stmts.append(the_stmt)

def empty(): pass # assert that the elem list given is empty, if not throw an error about trailing symbols


# STMTS

class Stmt(Node): # abstract
    def __init__(self):
        self.dependent = False
    def offer_neighbor(self, stmt):
        """
        All statements that want to consume their same-indent-level neighbors should override this, for example Try should override it to add stmt to its self.excepts if stmt is an Except node, and should return True to indicate that the stmt no longer needs to be added to the normal stmt list (bc instead it's inside of Try)
        """
        return False

class CompoundStmt(Stmt): pass # abstract
class SimpleStmt(Stmt): pass # abstract
class SynElem(Node): pass # Syntatic Element. Things like Formals that are neither Exprs nor Stmts

## Stmts that take an AColonList as input
class FuncDef(CompoundStmt):
    def __init__(self,compound):
        super().__init__()
        head = compound.head

        head = keyword(head,"def")
        self.name,head = identifier(head)
        self.args,head = Formals(head)
        empty(head)
        self.body = stmts(compound.body)

class Module(CompoundStmt):
    def __init__(self,compound):
        self.body = stmts(compound.body)

class ClassDef(CompoundStmt):
    def __init__(self,compound):
        super().__init__()
        head = compound.head
        self.bases = []
        self.keywords = {}

        head = keyword(head,"class")
        self.name,head = identifier(head)
        if isinstance(head[0],AParen):
            params,head = Formals(head)
            try:
                assert params.stararg is None
                assert params.doublestararg is None
                assert len(params.kwonlyargs) == 0
                assert len(params.kwonlydefaults) == 0
            except AssertionError:
                raise NotImplementedError("I know that there are some situations where starargs and such can be passed to a class header but have not yet implemented them. Furthermore my current logic that follows for finding self.bases and self.keywords may need to be modified based on this. But as long as these assertions hold my current method works")
            num_keywords = len(params.defaults)
            self.bases = params.args[:num_keywords]
            self.keywords = dict(list(zip(param.args[-num_keywords:],params.defaults)))
        empty(head)

        self.body = stmts(compound.body)


#def comma_split(elems): # [elem] -> [[elem]]
#    res = []
#    prev_comma_idx = -1
#    for i in range(elems):
#        if isinstance(elems[i],Tok) and elems[i].typ is COMMA:
#            res.append(elems[prev_comma_idx+1:i])
#            prev_comma_idx = i
#    res.append(elems[prev_comma_idx+1:])
#    return res

# parses: identifier '=' Expr
# not to be confused with keyword() which deals with language keywords
# Note that it will parse using expr_sep(elems,',') since it takes a single expression rather than an expression_list and shouldnt be consuming any commas.
def keyword_arg(elems): # [elems] -> KeywordArg,[elems] (or boolean if BOOL, etc)
    raise NotImplementedError
# TODO maybe valued_arg is a better name since it's used for defaults too? idk. Both Args and Formals use it


class Arg(SynElem):
    def __init__(self,name):
        self.name = name
class KeywordArg(SynElem):
    def __init__(self,name,val):
        self.name = name
        self.val = val

class Args(SynElem):
    def __init__(self,aparen_or_elem_list):
        super().__init__()
        if isinstance(aparen_or_elem_list,AParen):
            elems = aparen_or_elem_list.body
        else:
            elems = aparen_or_elem_list

        self.args = []
        self.starargs = []
        self.doublestarargs = []
        self.kwargs = []

        """
        There are 3 stages and there's no going backwards:
        1.Comma-sep list mix of Expr | *Expr
        2.As soons as an ident=Expr shows up, switch to a comma-sep list of (ident=Expr) | *Expr
        3.As soon as a **Expr shows up, switch a comma-sep list of (ident=Expr) | **Expr
        A trailing comma at the end is okay
        """

        while not empty(elems,fail=BOOL):
            # ident=Expr: Valid at any stage, shift to stage 2 if in stage 1
            # *Expr: Valid at stage <=2, shift 1->2
            # **Expr: Valid at any stage, shift to 3 always
            # Expr: Valid at stage 1, no shift
            if keyword_arg(elems,fail=BOOL):
                if stage == 1:
                    stage = 2
                kwarg,elems = keyword_arg(elems)
                self.kwargs.append(kwarg)
            elif token(elems,'*',fail=BOOL):
                if stage == 3:
                    raise SyntaxError("Starred arg must go before all double-starred args")
                elems = token(elems,'*',fail=BOOL)
                argname = identifier(elems)
                self.starargs.append(Arg(argname))
            elif token(elems,'**',fail=BOOL):
                stage = 3
                elems = token(elems,'**',fail=BOOL)
                argname = identifier(elems)
                self.doublestarargs.append(Arg(argname))
            elif identifier(elems,fail=BOOL):
                if stage != 1:
                    raise SyntaxError("normal arg must go before all keyword args and double-starred args")
                argname,elems = identifier(elems)
                self.args.append(Arg(argname))
            else:
                raise SyntaxError("Unable to parse argument {elems}")
            # end of an arg
            if empty(elems,fail=BOOL):
                break
            elems = token(elems,',')
        return None # end of Args __init__




class Formals(SynElem):
    def __init__(self,aparen_or_elem_list):
        super().__init__()
        if isinstance(aparen_or_elem_list,AParen):
            elems = aparen_or_elem_list.body
        else:
            elems = aparen_or_elem_list

        """
        There are 6 stages and there's no going backwards:
            Btw the BNF grammar fails to capture the fact that defaulted args can only come after nondefaulted args
        1. ident list
        2. (ident=Expr) list
        3. *ident | *
        4. ident list
        5. (ident=Expr) list
        6. **ident | **
        A trailing comma at the end is okay
        """

        self.args = []
        self.defaults = []
        self.stararg = None
        self.kwonlyargs = []
        self.kwonlydefaults = []
        self.doublestararg = None

        stage = 1

        # process each comma separated elemlist
        while not empty(elems,fail=BOOL):
            if stage == 6:
                raise SyntaxError("Nothing can come after a double-starred parameter")
            if keyword_arg(elems,fail=BOOL):
                kwarg,elems = keyword_arg(elems)
                if stage <= 2:
                    self.args.append(Arg(kwarg.name))
                    self.defaults.append(kwarg.value)
                    stage = 2
                elif stage <= 5:
                    self.kwonlyargs.append(Arg(kwarg.name))
                    self.kwonlydefaults.append(kwarg.value)
                    stage = 5
            elif token(elems,'*',fail=BOOL):
                if stage == 3:
                    raise SyntaxError("Can't have two single-starred parameters")
                if stage == 4:
                    raise SyntaxError("single-starred parameter must go before all kwonly parameters")

                elems = elems[1:]
                argname = identifier(elems,fail=FAIL)
                self.stararg = Arg(argname) if (argname is not FAIL) else ''
                stage = 3
            elif token(elems,'**',fail=BOOL):
                elems = elems[1:]
                argname = identifier(elems,fail=FAIL)
                self.doublestararg = Arg(argname) if (argname is not FAIL) else ''
                stage = 6
            elif identifier(elems,fail=BOOL):
                if stage == 2:
                    raise SyntaxError("all positional parameters must go before all default-valued positional parameters")
                if stage == 5 :
                    raise SyntaxError("all kwonly parameters must go before all default-valued kwonly parameters")
                elems = identifier(elems,fail=BOOL)
                argname = identifier(elems)
                if stage == 1:
                    self.args.append(Arg(argname))
                elif stage <= 4:
                    self.kwonlyargs.append(Arg(argname))
                    stage = 4
            # end of a formal
            if empty(elems,fail=BOOL):
                break
            elems = token(elems,',')

        if position == 3:
            raise SyntaxError(f"named arguments must follow bare *")
        return # end of __init__ for Formals





#            if len(elems) == 0:
#                if i != len(items)-1:
#                    raise SyntaxError(f"two commas in a row not allowed in argument list") # TODO point out exactly where it is / suggest fix / setup vim commands for fix
#
#            # kwonlynondefaulted or nondefaulted or stararg(no identifier)
#            if len(elems) == 1: # nondefaulted arg or stararg with no identifier
#                argname,elems = identifier(elems,fail=FAIL)
#                # kwonlynondefaulted or nondefaulted
#                if argname is not FAIL: # successfully parsed a nondefaulted arg
#                    if position > _kwonlynondefaulted:
#                        raise SyntaxError(f"{err_str[_kwonlynondefaulted]} must go before {err_str[position]}")
#                    # kwonlynondefaulted
#                    if _nondefaulted > position >= _kwonlynondefaulted:
#                        position = _kwonlynondefaulted
#                        self.kwonlyargs.append(Arg(argname))
#                        continue
#                    # nondefaulted
#                    position = _nondefaulted
#                    self.args.append(Arg(argname))
#                    continue
#                # failed kwonlynondefaulted and nondefaulted so lets try stararg with no ident
#                status,elems = token(elems,'*',fail=FAIL)
#                # stararg with no identifier. Note that theres no '**' without an identifier
#                if status is not FAIL: # successfully parsed a stararg with no identifier
#                    if position == _stararg:
#                        raise SyntaxError(f"not allowed to have multiple * or *arg arguments")
#                    if position > _stararg:
#                        raise SyntaxError(f"{err_str[_stararg]} must go before {err_str[position]}")
#                    position = _stararg
#                    self.stararg = Arg('')
#                    continue
#                raise SyntaxError(f"unable to parse argument {elems}")
#            # stararg with identifier
#            if len(elems) == 2: # *arg or **arg
#                status,elems = token(elems,'*',fail=FAIL)
#                if status is FAIL:
#                    status2,elems = token(elems,'**',fail=FAIL)
#                if status is FAIL and status2 is FAIL:
#                    raise SyntaxError(f"unable to parse argument {elems}")
#                if status is not FAIL:
#                    _mode = _stararg
#                else:
#                    _mode = _doublestararg
#
#                argname,elems = identifier(elems,fail=FAIL)
#                if status is not FAIL and argname is not FAIL: # successfully parsed a stararg with identifier
#                    if position == _mode:
#                        raise SyntaxError(f"not allowed to have multiple {'*arg' if _mode == _stararg else '**kwargs'} arguments")
#                    if position > _mode:
#                        raise SyntaxError(f"{err_str[_stararg]} must go before {err_str[position]}")
#                    position = _mode
#                    if _mode == _stararg:
#                        self.stararg = Arg(argname)
#                    else:
#                        self.doublestararg = Arg(argname)
#                    continue
#                raise SyntaxError(f"unable to parse argument {elems}")
#            # defaulted and kwonlydefaulted
#            if len(elems) > 2:
#                argname,elems = identifier(elems,fail=FAIL)
#                status,elems = token(elems,'=',fail=FAIL)
#                if argname is FAIL or status is FAIL:
#                    raise SyntaxError(f"unable to parse argument {elems}")
#                val,elems = expr(elems,fail=FAIL)
#                if val is FAIL:
#                    raise SyntaxError(f"unable to parse expression for default {elems}")
#                status = empty(elems,fail=FAIL)
#                if status is FAIL:
#                    raise SyntaxError(f"trailing tokens in default. The expression was parsed as {val} but there are trailing tokens: {elems}")
#                if position > _kwonlydefaulted:
#                    raise SyntaxError(f"{err_str[_kwonlydefaulted]} must go before {err_str[position]}")
#                # kwonlydefaulted
#                if _defaulted > position >= _kwonlydefaulted:
#                    position = _kwonlydefaulted
#                    self.kwonlyargs.append(Arg(argname))
#                    self.kwonlydefaults.append(val)
#                    continue
#                # defaulted
#                position = _defaulted
#                self.args.append(Arg(argname))
#                self.defaults.append(val)
#                continue
#
#        if position == _stararg:
#            raise SyntaxError(f"named arguments must follow bare *")
#        return # end of __init__ for Formals

class If(CompoundStmt):
    def __init__(self,compound):
        super().__init__()
        head = compound.head

        head = keyword(head,"if")
        self.cond,head = expr(head)
        empty(head)
        self.body = stmts(compound.body)

class CONSTANT(object):
    instances = []
    def __init__(self,name,**kwargs):
        self.name=name
        for k,v in kwargs.items():
            assert k != 'name'
            setattr(self,k,v)
        CONSTANT.instances.append(self)
        for i,instance in enumerate(CONSTANT.instances):
            for instance2 in enumerate(CONSTANT.instances[i+1:):
                assert instance != instance2
                assert instance is not instance2
                assert instance.name != instance2.name
    def __eq__(self,other):
        return self is other
    def __neq__(self,other):
        return self is not other
    def __repr__(self):
        return self.name
    # convenience methods so you can do a['test'] instead of a.test if it makes sense e.g. when looping thru lists of attrs or something
    def __getitem__(self,key):
        return getattr(self,key)
    def __setitem__(self,key,val):
        return setattr(self,key,val)

FAIL = CONSTANT('FAIL')
SILENT = CONSTANT('SILENT')
OK = CONSTANT('OK')

class Elif(CompoundStmt):
    def __init__(self,compound):
        super().__init__()
        head = compound.head
        self.dependent = True

        status,head = keyword(head,"elif",fail=FAIL)
        if status is FAIL:
            head = keyword(head,"else")
            head = keyword(head,"if")
        self.cond = expr(head)
        empty(head)
        self.body = stmts(compound.body)

class Else(CompoundStmt):
    def __init__(self,compound):
        super().__init__()
        head = compound.head
        self.dependent = True

        head = keyword(head,"else")
        empty(head)
        self.body = stmts(compound.body)


class For(CompoundStmt):
    def __init__(self,compound):
        super().__init__()
        head = compound.head

        head = keyword(head,"for")
        self.target,head = target(head)
        head = keyword(head,"in")
        self.iter,head = expr(head)
        empty(head)
        self.body = stmts(compound.body)

class While(CompoundStmt):
    def __init__(self,compound):
        super().__init__()
        head = compound.head

        head = keyword(head,"while")
        self.cond,head = expr(head)
        empty(head)
        self.body = stmts(compound.body)

class With(CompoundStmt):
    def __init__(self,compound):
        super().__init__()
        head = compound.head
        self.withitems = []

        head = keyword(head,"with")
        while not empty(elems,fail=BOOL):
            contextmanager,elems = expr(elems,till=(',',INCLUSIVE))
            target = None
            if keyword(elems,'as',fail=BOOL):
                keyword(elems,'as')
                target,elems = target(elems,till=(',',INCLUSIVE))
            self.withitems.append(Withitem(contextmanager,target))

        if len(self.withitems) == 0
            raise SyntaxError("empty `with` statement header")
        self.body = stmts(compound.body)
class Withitem(SynElem):
    def __init__(self,contextmanager,target):
        self.contextmanager = contextmanager
        self.target = target
class Try(CompoundStmt):
    def __init__(self,compound):
        super().__init__()
        head = compound.head

        head = keyword(head,"try")
        empty(head)
        self.body = stmts(compound.body)



## Stmts that take an elem list as input
class Return(SimpleStmt):
    def __init__(self,elems):
        super().__init__()

        elems = keyword(elems,"return")
        self.val,elems = expr(elems)
        empty(elems)
class Pass(SimpleStmt):
    def __init__(self,elems):
        super().__init__()

        elems = keyword(elems,"pass")
        empty(elems)
class Raise(SimpleStmt):
    def __init__(self,elems):
        super().__init__()
        self.exception = None
        self.cause = None

        elems = keyword(elems,"raise")
        if not empty(elems,fail=BOOL):
            self.exception,elems = expr(elems)
        if not empty(elems,fail=BOOL):
            elems = keyword(elems,'from')
            notempty(elems) # TODO it's important to have guards like this since otherwise an empty expr might just eval to None which Python does not do. This is along the lines of expr_assert_empty except like a expr_preassert_nonempty except with a better name. Oh also if there are random symbols for example that dont make up an expr then expr should tell us it couldnt make anything.
            self.cause,elems = expr(elems)
        empty(elems)


# `by` and `till` strings, tokens, or functions
# VERY IMPORTANT: if you split by commas in particular, the final sublist will be discarded if it is an empty list. This is because this is the desired behavior the vast majority of the time (for example foo(a,) is a valid function call on the argument `a` (NOT the tuple `(a,)`). Likewise `def foo(a,)` is allowed, `lambda x,:` is allowed.
# Also throws an error if there are two commas in a row only applies to `by`=',' again
"""CAREFUL. split should never be used if an expr may be somewhere inside whatever you're splitting. Because for example commas within lambda expressions are valid, or colons within lambdas -- if you split by commas or colons (e.g. when parsing slicing) it would get messed up. See examples of code like in Slice parsing for how to properly deal with these cases"""
def unsafe_split(): # [elem] -> [[elem]]
    raise NotImplementedError



def join(): # [[elem]] -> [elem]
    raise NotImplementedError

# [elem] -> | str         if `till` is None
#           | str,[elem]  if `till` is not None
# as usual `till` can be a string/token/function
def raw_string():
    raise NotImplementedError

class Import(SimpleStmt):
    def __init__(self,elems):
        super().__init__()
        self.importitems = []
        self.from = None
        """
        | import ident[.ident.ident] [as ident][, ident.ident.ident as ident, ident.ident.ident as ident]
        | from [..]ident[.ident.ident] import x [as ident][, y as ident, ident as ident]
        | from [..]ident[.ident.ident] import *
        * only `from` statments can use numdots != 0 like .mod or ..mod
        """

        # `from`
        if keyword(elems,"from",fail=BOOL):
            elems = keyword(elems,"from")
            self.from,elems = raw_string(imp,till='import')

        elems = keyword(elems,"import")

        # `from x import *`
        if token(elems,'*',fail=BOOL):
            if self.from is None:
                raise SyntaxError("`import *` not valid, must do `from [module] import *`")
            self.importitems = ['*']
            return

        imports = unsafe_split(elems,',') # the one case where it's basically fine to be unsafe because there is no expr parsing
        for imp in imports:
            alias = None
            modstr,imp = raw_string(imp,till='as')
            if not empty(imp,fail=BOOL):
                imp = keyword(imp,'as')
                alias = identifier(imp)
            self.withitems.append(Withitem(modstr,alias))

class Importitem(SynElem):
    def __init__(self,var,alias):
        self.var=var
        self.alias=alias
class Break(SimpleStmt):
    def __init__(self,elems):
        super().__init__()

        elems = keyword(elems,"break")
        empty(elems)
class Continue(SimpleStmt):
    def __init__(self,elems):
        super().__init__()

        elems = keyword(elems,"continue")
        empty(elems)
class Delete(SimpleStmt):
    def __init__(self,elems):
        super().__init__()
        self.targets = []

        elems = keyword(elems,"del")
        while not empty(elems,fail=BOOL):
            target,elems = target(elems,till=(',',INCLUSIVE),usage=DEL)
            self.targets.append(target)
        if len(self.targets) == 0
            raise SyntaxError("`del` statement must have at least one target")
        empty(elems)

class Global(SimpleStmt):
    def __init__(self,elems):
        super().__init__()
        self.names = []

        elems = keyword(elems,"global")
        while not empty(elems,fail=BOOL):
            name,item = identifier(item)
            self.names.append(name)
        if len(self.names) == 0
            raise SyntaxError("`global` statement must have at least one target")
        empty(elems)

class Nonlocal(SimpleStmt):
    def __init__(self,elems):
        super().__init__()
        self.names = []

        elems = keyword(elems,"nonlocal")
        while not empty(elems,fail=BOOL):
            name,item = identifier(item)
            self.names.append(name)
        if len(self.names) == 0
            raise SyntaxError("`global` statement must have at least one target")

        empty(elems)

class ExprStmt(SimpleStmt):
    def __init__(self,elems):
        super().__init__()
        self.expr,elems = expr(elems)
        empty(elems)

class Asn(SimpleStmt):
    def __init__(self,elems):
        super().__init__()
        self.targets = []

        while not empty(elems,fail=BOOL):
            tar,elems_new = target(elems,till=('=',INCLUSIVE),usage=STORE,fail=FAIL)
            # if we consume our last target and finish the stmt then that final target was actually an expr that just happened to parse as valid, so now we rewind and use expr()
            if empty(elems_new,fail=BOOL) or tar is FAIL:
                self.expr,elems = expr_assert_empty(elems)
                break
            self.targets.append(tar)


class AugAsn(SimpleStmt):
    def __init__(self,elems):
        super().__init__()
        self.target,elems = target(elems)
        self.op = elems[0]
        elems = token(elems,BINOPS)
        elems = token(elems,'=')
        self.val = expr(elems)
        empty(elems)

class Assert(SimpleStmt):
    def __init__(self,elems):
        super().__init__()
        elems = keyword(elems,"assert")
        self.cond,elems = expr(elems)
        if token(elems,',',fail=BOOL):
            self.expr,elems = expr(elems)
        empty(elems)




# EXPRS
class Targetable:
    def __init__(self,elems,**kw):
        assert isinstance(self,Expr)
        if 'usage' not in kw:
            raise InheritanceError("{self} is a Targetable object and must be called with the 'usage' kwarg")
        self.usage = kw.pop('usage')



# TODO All e_lhs inputs to a constructor (left recursive) should call set_usage on e_lhs. We enforce this by having all .targetable things ensure a non-None .usage during a tree traverasal after the AST is made



# TODO would be ultra nice to say identifier() vs identifer?() where the latter sends an extra argument which is just like fail=BOOL!

class Var(Expr):
    def __init__(self,name):
        super().__init__()
        self.name = name
    @staticmethod
    def identify(elems):
        if identifier(elems,fail=BOOL):
            return Var
        return None
    @staticmethod
    def build(elems):
        name,elems = identifier(elems)
        return Var(name),elems
class Starred(Expr):pass
class List(Expr):pass


# a,b,c
# "a," in Tup we call expr(left=Tup) which finds "b" then sees Tup and rates precedence as EQUAL which causes it to break and return "b" to us


# TODO add "leftnodeclass=" everywhere


"""
if you see a "," and are trying to expand via left recursion and ur not a

"""

@left_recursive
@gathers(Tuple)
class Tuple(Expr):
    def init(self,e_lhs,elems):
        self.vals = [e_lhs]
        if len(elems) == 1 and token(elems,',',fail=BOOL):
            return self,elems # singleton Tuple
        more = True
        while token(elems,','): # `more` means expr() ended on a comma last time
            e,elems = expr(elems,leftnodeclass=Tuple) # note that if there is more to this tuple it will
            self.vals.append(e)
        return self,elems
    def __init__(self,vals):
        super().__init__()
        self.vals=vals
    @staticmethod
    def identify(elems):
        if token(elems,',',fail=BOOL):
            return Tuple
        return None
    @staticmethod
    def build(lhs_node,elems):
        elems = token(elems,',')
        vals = [lhs_node]
        while True:
            # get an expr which cant be a tuple but might end right before a comma which is what happens in the gathering case
            elems,e = expr(elems[1:],leftnodeclass=Tuple)
            vals.append(e)
            if hasattr(e,'_gather') and e._gather is True:
                elems = token(elems,',')
                continue
            break
        return Tuple(vals),elems

class Set(Expr):pass
class Dict(Expr):pass
class Ellipsis(Expr):pass
class Comprehension(Expr):pass

@left_recursive
class Ternary(Expr):
    def __init__(self,cond,if_branch,else_branch):
        super().__init__()
        self.cond=cond
        self.if_branch=if_branch
        self.else_branch=else_branch
    @staticmethod
    def identify(elems):
        if keyword(elems,'if',fail=BOOL):
            return Ternary
        return None
    @staticmethod
    def build(lhs_node,elems):
        elems = keyword(elems,'if')
        cond,elems = expr(elems)
        elems = keyword(elems,'else')
        else_branch,elems = expr(elems)
        return Ternary(cond,lhs_node,else_branch),elems
    @staticmethod
    def finish(elems):
        pass
class Lambda(Expr):
    def __init__(self,args,body):
        super().__init__()
        self.args = args
        self.body = body
    @staticmethod
    def identify(elems):
        if keyword(elems,'lambda',fail=BOOL):
            return Lambda
        return None
    @staticmethod
    def build(elems):
        elems = keyword(elems,'lambda')
        sp = unsafe_split(elems,':',mode='first')
        args_elems = sp[0]
        args = Formals(args_elems)
        elems = join(sp[1:])
        # note that annotations are not allowed in lambda expressions. This is probably because the colon would make them impossible to parse in some cases since it's not necessarily embedded in any parentheses nclude <math.h>or anything. Therefore I should be fine to just use split()[0] to get the arguments.
        body,elems = expr(elems)
        return Lambda(args,body),elems
    @staticmethod
    def finish(elems):
        pass

class Yield(Expr):
    """
    | yield from Expr
    | yield Expr
    """
    def __init__(self,val,from):
        super().__init__()
        self.val = val
        self.from = from
    @staticmethod
    def identify(elems):
        if keyword(elems,'yield',fail=BOOL):
            return Yield
        return None
    @staticmethod
    def build(elems):
        elems = keyword(elems,'yield')
        status,elems = keyword(elems,'from',fail=FAIL)
        from = (status != FAIL)
        val,elems = expr(elems)
        return Yield(val,from),elems
    @staticmethod
    def finish(elems):
        pass

class Lit(Expr):
    def __init__(self,val):
        super().__init__()
        self.val = val
    @staticmethod
    def identify(elems):
        if (token(elems,INTEGER,fail=BOOL):
            return Int
        if token(elems,FLOAT,fail=BOOL):
            return Float
        if token(elems,COMPLEX,fail=BOOL):
            return Complex
        if token(elems,BYTES,fail=BOOL):
            return Bytes
        if keyword(elems,'True',fail=BOOL):
            return NamedConstant
        if keyword(elems,'False',fail=BOOL):
            return NamedConstant
        if keyword(elems,'None',fail=BOOL):
            return NamedConstant
        return None
if isinstance(elems[0],AQuote)):
        return test
    @staticmethod
    def build(elems):
        if token(elems,INTEGER,fail=BOOL):
            return Int(int(elems[0].data)),elems[1:]
        if token(elems,FLOAT,fail=BOOL):
            return Float(float(elems[0].data)),elems[1:]
        if token(elems,COMPLEX,fail=BOOL):
            return Complex(complex(elems[0].data)),elems[1:]
        if token(elems,BYTES,fail=BOOL):
            return Bytes(bytes(elems[0].data)),elems[1:]
        if keyword(elems,'True',fail=BOOL):
            return NamedConstant(True),elems[1:]
        if keyword(elems,'False',fail=BOOL):
            return NamedConstant(False),elems[1:]
        if keyword(elems,'None',fail=BOOL):
            return NamedConstant(None),elems[1:]
        if isinstance(elems[0],AQuote):
            return Str(elems[0].data),elems[1:]
        return None
    @staticmethod
    def finish(elems):
        pass
class Num(Lit):pass
class Int(Num):pass
class Float(Num):pass
class Complex(Num):pass
class Str(Lit):pass
class Bytes(Lit):pass
class NamedConstant(Lit):pass

class UnopL(Expr):pass
    def __init__(self,val):
        super().__init__()
        self.val = val # no need to include `op` bc thats captured by subclass
    @staticmethod
    def identify(elems):
        if token(elems,UNOPSL,fail=BOOL):
            return unopl_subclass[elems[0].typ]
        return None
    @staticmethod
    def build(elems):
        op = elems[0].typ
        cls = unopl_subclass(op)
        val,elems = expr(elems[1:],leftnodeclass=cls)
        return cls(val),elems
        raise NotImplementedError(f"Unrecongized left unary operator: {op}")
    @staticmethod
    def finish(elems):
        pass
class UAdd(UnopL):pass
class USub(UnopL):pass
class Not(UnopL):pass
class Invert(UnopL):pass

@left_recursive
class UnopR(Expr):
    def __init__(self,val):
        super().__init__()
        self.val = val # no need to include `op` bc thats captured by subclass
    @staticmethod
    def identify(elems):
        if token(elems,'.',fail=BOOL):
            return Attr
        if isinstance(elems[0],AParen):
            return Call
        if isinstance(elems[0],ABracket):
            _,body = expr(elems[0].body, till=[',',':'])
            if empty(body,fail=BOOL):
                return Subscript # if it's a single expr then it's a subscript
            return Slice
        return None
    @staticmethod
    def build(lhs_node,elems):
        if token(elems,'.',fail=BOOL):
            # Attr
            elems = token(elems,'.')
            attr,elems = identifer(elems)
            return Attr(lhs_node,attr),elems
        if isinstance(elems[0],AParen):
            # Call
            """
            Calls can apparently take a comprehension! I will ignore this for now, tho it's probably not hard to do once you've done comprehensions.
            """
            args = Args(elems[0])
            return Call(lhs_node,args),elems[1:]
        if isinstance(elems[0],ABracket):
            body = elems[0].body
            # Subscript or Slice
            """
            Subscript = "[" Expr "]"
            Slice: has at least one comma or colon (colon can't be part of an expr tho like a lambda is fine for example)
            Slice = comma sep list of Expr | Expr:Expr | Expr:Expr:Expr
            If the slice list contains at least one comma, the key is a tuple containing the conversion of the slice items; otherwise, the conversion of the lone slice item is the key. The conversion of a slice item that is an expression is that expression. The conversion of a proper slice is a slice object (see section The standard type hierarchy) whose start, stop and step attributes are the values of the expressions given as lower bound, upper bound and stride, respectively, substituting None for missing expressions.
            sounds like we should make a slice() object since that's a real thing in python
            """
            slicers = []
            curr_slicer = []
            while not empty(body,fail=BOOL):
                e,body = expr(body,till=[',',':'])
                if token(body,':',fail=BOOL):
                    curr_slice.append(e)
                elif token(body,',',fail=BOOL):
                    slicers.append(Slicer(curr_slicer))
                    curr_slicer = []
                else:
                    raise SyntaxError("trailing characters in slicing or subscript")
            if len(slicers) == 1 and slicers[0].stop is None and slicers[0].step is None:
                # Subscript
                return Subscript(lhs_node,slicers[0].start),elems[1:]
            # Slice
            return Slice(lhs_node,slicers),elems[1:]
    @staticmethod
    def finish(elems):
        pass
class Attr(UnopR):
    def __init__(self,val,attr):
        # no super() init. Putting everything in here is clearer.
        self.val = val
        self.attr = attr
class Subscript(UnopR):
    def __init__(self,val,index):
        # no super() init
        self.val = val
        self.index = index
class Slice(UnopR):
    def __init__(self,val,slicers)
        # no super() init
        self.val = val
        self.slicers = slicers
class Slicer(Expr): # the python slice() object. Note that the name Slice is taken already
    def __init__(self,start_stop_step):
        self.start = start_stop_step[0]
        self.stop = start_stop_step[1] if len(start_stop_step) > 1 else None
        self.step = start_stop_step[2] if len(start_stop_step) > 2 else None
class Call(UnopR):
    def __init__(self,func,args):
        # no super() init
        self.func = funct
        self.args = args

unopl_subclass = {
    ADD: UAdd,
    SUB: USub,
    NOT: Not,
    INVERT: Invert
}

binop_subclass = {
    ADD: Add,
    SUB: Sub,
    MUL: Mul,
    DIV: Div,
    FLOORDIV: FloorDiv,
    EXP: Exp,
    SHIFTRIGHT: ShiftR,
    SHIFTLEFT: ShiftL,
    BITAND: BitAnd,
    BITOR: BitOr,
    BITXOR: BitXor
}

boolop_subclass = {
    AND: And,
    OR: Or
}

@left_recursive
class Binop(Expr):
    def __init__(self,lhs,rhs):
        super().__init__()
        self.lhs = lhs # no need to include `op` bc thats captured by subclass
        self.rhs = rhs
    @staticmethod
    def identify(elems):
        if token(elems,BINOPS,fail=BOOL):
            return binop_subclass[elems[0].typ]
        return None
    @staticmethod
    def build(lhs_node,elems):
        op = elems[0].typ
        cls = binop_subclass[op]
        rhs_node,elems = expr(elems[1:],leftnodeclass=cls)
        return cls(lhs_node,rhs_node),elems
    @staticmethod
    def finish(elems):
        pass
class Add(Binop):pass
class Sub(Binop):pass
class Mul(Binop):pass
class Div(Binop):pass
class FloorDiv(Binop):pass
class Mod(Binop):pass
class MatMul(Binop):pass
class Exp(Binop):pass
class ShiftR(Binop):pass
class ShiftL(Binop):pass
class BitAnd(Binop):pass
class BitOr(Binop):pass
class BitXor(Binop):pass


# eh i guess inference and such will work just as well on nested boolops as if we flattened it all out so this is fine
@left_recursive
class Boolop(Expr):
    def __init__(self,lhs,rhs):
        super().__init__()
        self.lhs = lhs # no need to include `op` bc thats captured by subclass
        self.rhs = rhs
    @staticmethod
    def identify(elems):
        if token(elems,BOOLOPS,fail=BOOL):
            return boolop_subclass[elems[0].typ]
        return None
    @staticmethod
    def build(lhs_node,elems):
        op = elems[0].typ
        cls = boolop_subclass[op]
        rhs_node,elems = expr(elems[1:],leftnodeclass=cls)
        return cls(lhs_node,rhs_node),elems
class And(Boolop):pass
class Or(Boolop):pass

@left_recursive
@gathers(Compare)
class Compare(Expr):
    def __init__(self,ops,vals):
        super().__init__()
        self.ops = ops
        self.vals = vals
    @staticmethod
    def identify(elems):
        if token(elems,CMPOPS,fail=BOOL):
            return Compare
        return None
    @staticmethod
    def build(elems,lhs_node,leftnodeclass):
        vals = [lhs_node]
        ops = [elems[0].typ]
        while True:
            # parse an expr which is at most a single comparison
            elems,e = expr(elems[1:],leftnodeclass=Compare)
            vals.append(e)
            if hasattr(e,'_gather') and e._gather is True:
                ops.append(elems[0].typ)
                continue
            break

        return Compare(ops,vals),elems


# all these might be unnecessary
#class CompareElem(SynElem): pass
#class Lt(CompareElem):pass
#class Leq(CompareElem):pass
#class Gt(CompareElem):pass
#class Geq(CompareElem):pass
#class Eq(CompareElem):pass
#class Neq(CompareElem):pass
#class Is(CompareElem):pass
#class Isnot(CompareElem):pass
#class In(CompareElem):pass
#class Notin(CompareElem):pass

class Await(Expr): pass # TODO


# TODO watch out for `lambda` expressions -- their colon can mislead atomize!!! esp since it's not necessarily any parens or anything!


## TODO next: make SInitial more pretty. I dont htink overloader is the way to go, we shd start in SInit then trans to Snormal not just wrap Snormal.

## assertion based coding. After all, we're going for slow-but-effective. And assertions can be commented in the very final build. This is the python philosophy - slow and effective, but still fast enough

## would be interesting to rewrite in Rust or Haskell. ofc doesn't have all the features we actually want bc in particular we should be able to override parse_args or whatever dynamically. And list of callable()s would have to be passed here.

import re
from util import die,warn,Debug
import util as u
import inspect

# Tok is the lowest level thing around. 
# It's just an enum for the different tokens, with some built in regexes



OFFSET=100
# str_of_token(PERIOD) == 'PERIOD'
## important: if you add a TOKEN to this list you must also add it to the list of constants and REGEX_OF_TOKEN, and it must show up in the SAME position in the former two lists (it can show up somewhere different in REGEX_OF_TOKEN and that's fine)
_str_of_token = ['INTEGER','PERIOD','COMMA','COLON','SEMICOLON','EXCLAM','PYPAREN','SH_LBRACE','SH_LPAREN','SH','LPAREN','RPAREN','LBRACE','RBRACE','LBRACKET','RBRACKET','ADD','SUB','MUL','FLOORDIV','DIV','EXP','MOD','SHIFTRIGHT','SHIFTLEFT','BITAND','BITOR','BITXOR','INVERT','NOT','AND','OR','LEQ','LT','GEQ','GT','EQ','NEQ','IS','ISNOT','IN','NOTIN','ASN','ESCQUOTE2','ESCQUOTE1','QUOTE2','QUOTE1','HASH','PIPE','ID','UNKNOWN','SOL','EOL','NEWLINE','KEYWORD']
def str_of_token(t):
    return _str_of_token[t-OFFSET]
# order doesn't matter in this list, as long as it's the same order as in the preceding list
[INTEGER, PERIOD, COMMA, COLON, SEMICOLON, EXCLAM, PYPAREN, SH_LBRACE, SH_LPAREN, SH, LPAREN, RPAREN, LBRACE, RBRACE, LBRACKET, RBRACKET, ADD, SUB, MUL, FLOORDIV, DIV, EXP, MOD, SHIFTRIGHT, SHIFTLEFT, BITAND, BITOR, BITXOR, INVERT, NOT, AND, OR, LEQ, LT, GEQ, GT, EQ, NEQ, IS, ISNOT, IN, NOTIN, ASN, ESCQUOTE2, ESCQUOTE1, QUOTE2, QUOTE1, HASH, PIPE, ID, UNKNOWN, SOL, EOL, NEWLINE, KEYWORD] = [CONSTANT() for i in range(len(_str_of_token))] # we start at 100 so that we can still safely manually define constants in the frequently used -10..10 range of numbers with no fear of interference or overlap.

CMPOPS = [LEQ,LT,GEQ,GT,EQ,NEQ,IS,ISNOT,IN,NOTIN]
BINOPS = [ADD, SUB, MUL, DIV, FLOORDIV, MOD, EXP, SHIFTRIGHT, SHIFTLEFT, BITAND, BITOR, BITXOR, MATMUL]
UNOPSL  = [ADD, SUB, NOT, INVERT] # yes, ADD and SUB can be unops
UNOPSR  = [] # right side unary operators. There aren't any... yet! Well actually () and [] and . are UNOPSR really.
BOOLOPS= [AND,OR]


COMMON_ALLOWED_CHILDREN = set([QUOTE1,QUOTE2,LBRACKET,LBRACE,LPAREN,SH_LBRACE,SH, SH_LPAREN, HASH])
COMMON_ERROR_ON = set([RPAREN,RBRACE,RBRACKET]) # note that closers are checked first in Atomize so it's safe to include RPAREN in the .error_on field of AParen for example. Hence the COMMON_ERROR_ON can be widely used


# compile the regex and also eat any whitespace that follows it
def regex_compile(regex):
    return re.compile(regex+'\s*')

# order is important in this list!
regex_of_token = {
    INTEGER   : regex_compile(r'(\d+)'),
    PERIOD    : regex_compile(r'\.'),
    COMMA     : regex_compile(r','),
    COLON     : regex_compile(r':'),
    SEMICOLON : regex_compile(r';'),
    EXCLAM    : regex_compile(r'!'),

    # GROUPINGS
    PYPAREN   : regex_compile(r'py\('), # BEFORE ID
    SH_LBRACE : regex_compile(r'sh{'), # BEFORE ID
    SH_LPAREN : regex_compile(r'\(\s*sh\s+'), # BEFORE ID
    SH        : regex_compile(r'sh\s+'), # BEFORE ID, AFTER SH_*
    LPAREN    : regex_compile(r'\('),
    RPAREN    : regex_compile(r'\)'),
    LBRACE    : regex_compile(r'{'),
    RBRACE    : regex_compile(r'}'),
    LBRACKET  : regex_compile(r'\['),
    RBRACKET  : regex_compile(r'\]'),

    # BINOPS
    ADD       : regex_compile(r'\+'),
    SUB       : regex_compile(r'-'),
    MUL       : regex_compile(r'\*'),
    FLOORDIV  : regex_compile(r'//'), # BEFORE DIV
    DIV       : regex_compile(r'/'),
    EXP       : regex_compile(r'\*\*'),
    MOD       : regex_compile(r'%'),
    SHIFTRIGHT: regex_compile(r'>>'), # BEFORE GT
    SHIFTLEFT : regex_compile(r'<<'), # BEFORE LT
    BITAND    : regex_compile(r'&'),
    BITOR     : regex_compile(r'\|'),
    BITXOR    : regex_compile(r'\^'),

    # UNOPS
    INVERT    : regex_compile(r'~'),
    NOT       : regex_compile(r'not'),
    # note that ADD and SUB can also be unops like SUB in '-x' and ADD in '+x' (leaves arg unchanged),

    # BOOLOPS
    AND       : regex_compile(r'and'), # BEFORE ID
    OR        : regex_compile(r'or'), # BEFORE ID

    # COMPARISON
    LEQ       : regex_compile(r'<='), # BEFORE LT
    LT        : regex_compile(r'<'),
    GEQ       : regex_compile(r'>='), # BEFORE GT
    GT        : regex_compile(r'>'),
    EQ        : regex_compile(r'=='), # BEFORE EQ
    NEQ       : regex_compile(r'!='),
    IS        : regex_compile(r'is'), # BEFORE ID
    ISNOT     : regex_compile(r'is\s+not'), # BEFORE ID
    IN        : regex_compile(r'in'), # BEFORE ID
    NOTIN     : regex_compile(r'not\s+in'), # BEFORE ID

    ASN       : regex_compile(r'='),
    ESCQUOTE2 : regex_compile(r'\\\"'),
    ESCQUOTE1 : regex_compile(r'\\\''),
    QUOTE2    : regex_compile(r'\"'),
    QUOTE1    : regex_compile(r'\''),
    HASH      : regex_compile(r'#'),
    PIPE      : regex_compile(r'\|'),
    ID        : regex_compile(r'([a-zA-z_]\w*)'), # must go after keywords like sh
    UNKNOWN   : regex_compile(r'(.)'),
    SOL : None, # should never be matched against since 'UNKOWN' is a catch-all
    EOL : None, # should never be matched against since 'UNKOWN' is a catch-all
    NEWLINE : None, # newlines are manually inserted hence no need for a regex
    KEYWORD : None, # custom matching as a step after ID for efficiency
}
assert len(regex_of_token) == len(_str_of_token)
TOKENS = list(regex_of_token.keys())

def closer_of_opener(opener_tok):
    if isinstance(opener_tok,Tok):
        opener_tok = opener_tok.typ

    if opener_tok == LPAREN:
        return Tok(RPAREN,'',')')
    if opener_tok == PYPAREN:
        return Tok(RPAREN,'',')')
    if opener_tok == LBRACE:
        return Tok(RBRACE,'','}')
    if opener_tok == LBRACKET:
        return Tok(RBRACKET,'',']')
    if opener_tok == QUOTE1:
        return Tok(QUOTE1,'','\'')
    if opener_tok == QUOTE2:
        return Tok(QUOTE2,'','"')
    if opener_tok == SOL:
        return Tok(EOL,'','')
    raise NotImplementedError


# This is a parsed token
class Tok(LocTagged):
    def __init__(self,typ,data,verbatim, loc):
        super().__init__()
        self.loc = loc
        self.typ=typ            #e.g. IDENTIFIER
        self.data=data          # the contents of the capture group of the Tok's regex, if any.
        self.verbatim=verbatim  # e.g. 'foo'. The verbatim text that the regex matched on
    def __repr__(self):
        data = f"({u.mk_y(self.data)})" if self.data != '' else ''
        return f"{repr(toktyp_of_const[self.typ])}{data}"
    def finish(self): # for LocTagged
        self.finished = True



leading_whitespace_regex = re.compile(r'(\s*)')


LineData = namedtuple('LineData', 'linetkns, leading_whitespace, newline_tok, commenttkns')

# turns a string into a list of Tokens
# if ur sending an interative line PLEASE strip the trailing \n
def tokenize(s):
    assert not '\t' in s, "Havent yet figured out how tabs should work with indent levels since len('\t') is 1 but they take up 4 visual spaces, just see what python allows and experiment with noexpandtab. If python doesnt allow for mixed tabs and spaces in a file then thats perfect bc then len() works on leading_whitespace"
    tokenized_lines = []
    lineno=1
    for line in s.split('\n'):
        linetkns = []
        commenttkns = []
        list_to_extend = linetkns
        leading_whitespace = leading_whitespace_regex.match(line).group()
        remaining = line[len(leading_whitespace):]
        while remaining != '':
            for TOKEN,regex in regex_of_token.items():
                charno = len(line) - len(remaining)
                match = regex.match(remaining)
                if match is None:
                    continue
                remaining = remaining[match.end():]
                if TOKEN is HASH and list_to_extend is not commenttkns:
                    list_to_extend = commenttkns # switch to adding to list of comment tokens
                    continue
                if TOKEN is ID:
                    if match.group() in keywords:
                        TOKEN = KEYWORD # for efficiency rather than having every keyword in the regex
                grps = match.groups()
                list_to_extend.append(Tok(TOKEN,
                    grps[0] if grps else '', # [] is nontruthy
                    match.group(),
                    Loc(Pos(lineno,charno),Pos(lineno,len(line)-len(remaining))) # lineno,charno TODO watch out for how \t is handled!
                    ))
                list_to_extend[-1].finish()
                break #break unless you 'continue'd before
            else: # nobreak
                raise ValueError(f"Somehow failed to match on any tokens, should be impossible because of the UNKNOWN token")
        #is_comment = (line.strip() == '' or line.strip()[0] == '#') # lines that just have comments or nothing on them
        newline_tok = Tok(NEWLINE, '\n', '\n', (lineno,len(line)+1))
        tokenized_lines.append(LineData(linetkns, leading_whitespace, newline_tok, commenttkns))
        lineno += 1
    return tokenized_lines

INCOMPLETE = -1
# the main function that run the parser
# It goes string -> Token list -> Atom list -> Atom list (w MacroAtoms) -> final python code
# `interpreter` should be True if we're parsing from an interpreter and thus atomize can return INCOMPLETE rather than throwing an error when parens/etc are unclosed
def parse(lines,globals=None,interpreter=False,debug=False):
    debug = True
    debug = Debug(debug)
    linedatalist = tokenize(lines)
    commentlinedata = [linedata.commenttkns for linedata in linedatalist] # may want for something sometime
    debug.print(f"Tokens: {[linedata.linetkns for linedata in linedatalist]}")
    masteratom = atomize(linedatalist,interpreter)
    debug.print(masteratom)
    if atoms is INCOMPLETE:
        print("INCOMPLETE")
        exit(0)
        return INCOMPLETE
    ast = make_ast(masteratom)
    #debug.print(ast)
    ast = comment_mods(commentlinedata,ast) # does whatever mods we want comments to be able to do to the tree
    debug.print(ast)
    #tstream = TokStream(token_list)
    #init = SInitial(parent=None, tstream=tstream, globals=globals, debug=debug)
    #out = init.run()
    return ast
## CAREFUL RELOADING CODEGEN IT WILL RESET THESE VALUES

def comment_mods(commentlinedata,ast):
    return ast

def make_ast(masteratom):
    res = []
    for elemlist in masteratom.body:
        stmtnode = parseStmt(elemlist)
        res.append(stmtnode)
    return Module(res)




parse('#here is a comment',{})
parse('sh echo hey there',{})

parse('({(test)})',{})
parse('print((x + 3)/2.4);print("hi")',{})
parse('test;test2',{})
parse('test\ntest2',{})


print("===")
#parse('''ret''',{})

parse('''
def foo(a,b):
    def bar():
        return a
    x=bar()
    return a+b
foo(10,20)
u
''',{})

parse('''
x
def foo():
# here is a test
''',{})
parse('x = sh{echo hey there $(py expr)}',{})
parse('sh echo hey there',{})



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




