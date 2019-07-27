# CODEGEN
# OVERVIEW:
# class Tok: an enum of the possible tokes, eg Tok.WHITESPACE
# class Token: a parsed token. has some associated data
# class AtomCompound, etc: These Atoms are AST components. Each has a .gentext() method that generates the actual final text that the atom should become in the compiled code
# parse() is the main function here. It goes string -> Token list -> Atom list -> Atom list (w MacroAtoms) -> final python code
from collections import namedtuple,defaultdict
from keyword import kwlist
keywords = set(kwlist)


"""
TODO
-note on resuming this after a break you really need to look thru example code to remind yourself all the tools. In particular read each function of Parser (some but not all were copied to this massive commant section at the top of the file)
-p.next_line() and make save_state() work with it
-p.stmts()
-fyi argless lambdas are already a thing so use them!
-assert that end of line is reached after each stmt is parsed
-Note that all expression desugaring can desugar to a function call (potentially with argless lambdas for some arguments for lazy eval).
-All stmt desugaring can desguar to a list of statements
-Runtime errors for generated code: Definitely need to try/except the errors in any generated code and reemit errors that have proper information. Shouldn't be too hard to get the right line number info. I guess any expression desurgaring can always emit a function call which can be decorated to do the try/except. Then statement desurgaring can just have the whole block wrapped in try/except. Def gotta think a bit about what to put in except, but I feel like it's doable. Just gotta be very robust with writing helpful error messages for each class tho
-use .alias() in all those places i did it manually before
- add a .bnf field to all Nodes (where applicable)
- make notes about how expression_list starred_expression and starred_list all yield Tuples or Exprs rather than python lists, unless as_expr=False is passed to them, and the LRM says this should only really be done for SetDisplay and ListDisplay. target_list, parameter_list, etc all yield python lists.
    -youll want to make sure nodes with an (expression/starred)_(list/item) have a .val or .expr field not a .vals or .exprs field -- its gonna be a single value!
-check for cases like "raise" [expression ["from" expression]] where there are nested [] and therefore you need to only ever try the inner ones in the outer ones fail
-add BNF line to every Stmt and Expr and other Node that has an LRM BNF
note: i guess we cut the _stmt postfix off most of our classes (Assignment instead of AssignmentStmt) bc they seem to needlessly add it to everything for BNF. Exception is ExpressionStmt includes Stmt.
-switch all uses of parens() brackets() etc to `with` stmts
- Make it so theres a .yield_expression() instead that does .build_node(yield_expression) and just do that for everything in the BNF. Maybe make an autogenerator based on all Nodes having a .bnf field that gives the string (not decorator-added, rather have it be a classic class field assignment thing). Then at the bottom of the module scan thru all globals() for things that are Nodes and check if they have bnf not-None and if so add their methods to Parser.
- write documentation for everything at top of this file, as youve already started!
-`Not` shd not be in UnopL bc of precedence
-comp_for needs internal recursion as in LRM bc its used in generators too so unless you wanna track dependencies you gotta go verbatim
-make a ident.ident.ident shorthand (with no_whitespace) like dotted_name (use in Decorator, Import, etc)
-make it so @compound can take '@' for example (used for decorators)
-note my argument_list() actually includes the parens and opt trailing comma already
-we need to deal with cases like AttributeRef where x.y is okay but x .y or x. y are not okay, likewise in import statement "mod1. mod2" is not okay. Basically i think this applies pretty consistently to certain tokens like '.' and maybe brackets. Could have a no_whitespace() contextmanager perhaps that changes the behavior of .tok to forbid whitespace.
-desugaring MatchExpr seems tricky. MatchStmt wouldn't be hard.
-parse_expr(str): return parse(str).expr # ExpressionStmt->Expr
    -more shorthands for parsing args and such and any other case
Quote stuff:
    - Finish QUOTE1/QUOTE2 -> QUOTE merge
    - HEREDOC token integrate
    - Note that we choose to parse quotes ourselves without a single regex so we can do stuff like fstrings or custom equivalents better. Fstrings throw an error on f"{" for unclosed brace. Ofc the normal string "{" is not an error. So we should perhaps parse strings tokenwise as we have been, but with a prefix flag that tells it what special parsing to do.
    - an alternative would be to use token style parsing for strings and merely use regexes for normal strings, but our system already has issues with multiline heredoc style regexes so really sticking to the other way would make more sense.
    - fstrings:
        f_string          ::=  (literal_char | "{{" | "}}" | replacement_field)*
        replacement_field ::=  "{" f_expression ["!" conversion] [":" format_spec] "}"
        f_expression      ::=  (conditional_expression | "*" or_expr)
                                 ("," conditional_expression | "," "*" or_expr)* [","]
                               | yield_expression
        conversion        ::=  "s" | "r" | "a"
        format_spec       ::=  (literal_char | NULL | replacement_field)*
        literal_char      ::=  <any code point except "{", "}" or NULL>

-Improve Lit parsing to cover all cases
    -see QUOTE stuff above
    -ints:
        integer      ::=  decinteger | bininteger | octinteger | hexinteger
        decinteger   ::=  nonzerodigit (["_"] digit)* | "0"+ (["_"] "0")*
        bininteger   ::=  "0" ("b" | "B") (["_"] bindigit)+
        octinteger   ::=  "0" ("o" | "O") (["_"] octdigit)+
        hexinteger   ::=  "0" ("x" | "X") (["_"] hexdigit)+
        nonzerodigit ::=  "1"..."9"
        digit        ::=  "0"..."9"
        bindigit     ::=  "0" | "1"
        octdigit     ::=  "0"..."7"
        hexdigit     ::=  digit | "a"..."f" | "A"..."F"
    -floats:
        floatnumber   ::=  pointfloat | exponentfloat
        pointfloat    ::=  [digitpart] fraction | digitpart "."
        exponentfloat ::=  (digitpart | pointfloat) exponent
        digitpart     ::=  digit (["_"] digit)*
        fraction      ::=  "." digitpart
        exponent      ::=  ("e" | "E") ["+" | "-"] digitpart
    -complex:
        imagnumber ::= (floatnumber | digitpart) ("j" | "J")

-Imporve Var parsing to cover all cases
-The new plan is sh(whatever multiline whatever you want) and py(whatever multiline whatever you want). Basically theyre special builtins that you cant override ie kinda like keywords.
    And `sh` with no immediate open paren will just continue to the end of the line
-somethign to autocheck that all identify() are staticmethod and all build are staticmethod would be nice. Dont autoinsert it if people forget, then the code is less readable.
-Figure out how Annotated class should work. Ofc it should basically be a Var. It should be able to wrap anything in the future i suppose. So maybe it should just be a fundamental part of a Node's label. These hard-annotations are done by the user and soft-annotations are figured out by the parser. HMMM but ok, right now annotations in python only apply to functions (we should maintain that but also allow it for certain other cases like assignment or perhapppps standalone / in a tuple x:int,y:int or even x,y:int as stmts. Basically a typehint stmt)
-note that .annotate() cant be used on strings, making Var more convenient to use widely. If a string needs annotation it should be a Var instead.
-Redo documentation to properly say what each type has for example for Arguments and Parameters:
    Arguments -> .args((Arg|KVPair|Starred(Arg)|DoubleStarred(Arg)) list)
    Parameters -> .args/.kwonlyargs((Arg|KVPair) list) .stararg/.doublestararg(str|''|None)
        If you want to find defaulted ones easy just do a quick filter. We don't do it for you, that unnecessary.
    Arg
    KVPair (ie defaulted args)

-maybe have token() return a Tok to make stuff like Compare better
-Args and Formals probably want to be rewritten as __init__(etc etc etc) and build(p) so that they can be instantiated without parser.
-make MatchExpr syntax more solidified

Don't worry .identify is always called with peek(), you dont need to manage that




It's important not to keep any state that needs to potentially be rewound in the Atoms or Tokens during Parser action. All that should be in the Parser and save_state/load_state should caputre it.


A Guide to our functions:

.must # may be removed
.not_none # may be removed **WAS REMOVED IN USAGE
.or_none
.or_false
.or_fail # may be removed
.or_(reval)

Note any or_* function can be used like:
    p.or_none.method_of_parser(*args,**kwargs)
        e.g. p.or_none.parens()
OR like:
    p.or_none(any_argless_fn)
        e.g. p.or_none(parse_arg_stage3)

Combinators:
.logical_xor(*fns)
.list(*fns) - call logical_xor(*fns) in a loop and return a list of the results. Stop at first SynErr.
.list(*fns,nonempty=True) - .list() but SynErr if returned list is length 0
.comma_list(*fns) - like .list() but consumes a comma after each successful fn call. If unable to consume a comma or unable to successfully call any fn in `fns`, then terminate and return the list of results. (Allows trailing comma)
.comma_list(*fns,nonempty=True) - .comma_list() but SynErr if returned list is length 0

Internal:
.next() - generally called internally to step .tok forward

Primitives:
.curr - current token
.token(tok_like)
.keyword(kw_str)
.identifier()
.empty()
.assert_empty()

BNF components:
.parameter_list()
.parameter_list(no_annotations=True) - param list in lambda fn
.expression_list()
.expression_nocond()
.expression()
.or_expr()
.or_test()
.primary()
.starred_expression()
.starred_list()
.comprehension()
.comp_for()
.comp_if()
.comp_iter()
.target()
.trunk_expr(, type)
    'enclosure':List,Dict,Set,GeneratorExpr,YieldAtom,ParenForm # tightest binding
    'atom':Var,Lit
    'primary':AttributeRef,Subscript,Slicing,Call
    'await_or_primary':AwaitExpr
    'power':Exp,
    'u_expr':UAdd,USub,Invert
    'm_expr'Mul,Div,FloorDiv,Mod,MatMul
    'a_expr':Add,Sub
    'shift_expr':ShiftL,ShiftR
    'and_expr':BitAnd,
    'xor_expr':BitXor,
    'or_expr':BitOr,
    'comparison':Compare,
    'not_test':Not,
    'and_test':And,
    'or_test':Or,
    'conditional_expression':Ternary,
    'expression':Lambda, # weakest binding


.brackets()
.braces()
.parens()
.quotes()
.brackets()
.colonlist(p)
._atom_body(,atom_cls)
._atom_head_body(,atom_cls)
.argument_list()
.target_list(p)
.logical_xor(,*option_fns)
.logical_xor(,*option_fns)
.build_node(,nodeclass,leftnode=None,**kwargs)
.stmt()


Context managers:
.peek() - always rewind stream, and reraise SynErr if it occurs
.maybe() - rewind stream on SynErr, dont reraise SynErr



make sure .assert_empty() is properly called everywhere


Special Nodes:
InitExpr(Expr) - an Expr with no build/identify functions so it must be created directly through __init__ (which takes whatever its components are, NOT a Parser.
ExprGenerator(Node) - a Node which never shows up in the AST, when you call build() it just returns some other Expr/Stmt node, possibly from a range of options.
AuxNode(Node) - Things like Parameters and Args that do not evaluate to a value and thus are not Exprs.
AbstractExpr(Expr) - Things like Binop that never exist in the AST but are useful parent classes to have for isinstance() calls and inheritance.


class decorators
@left_recursive(left_type) -
@gen_build(*args,**kwargs) -
@binop(left_type,right_type) - shorthand for left_recursive and gen_build


-(again at end) ensure all super().__init__() calls are done

-add loc_tagged stuff using the decorator i think i wrote and phone notes

-TODO to speed things up a LOT you can do use logical_or in place of xor everywhere. Maybe by just doing Parser.logical_xor = Parser.logical_or for unsafe but fast mode


if Expr has .identify then run that to get the subclass (recursive) and when you finally get a subclass run its .build


Starred and DoubleStarred InitExpr's are used to wrap nodes in stars



Docs:

_tokenize: (str with no newlines)->

The following printing ASCII characters are not used in Python. Their occurrence outside string literals and comments is an unconditional error: $ ? `







Notes on performance with throwing/catching exceptions:
    If you race a simple function that returns "text" vs one that raises SyntaxError("text") the simple function will be 4x faster, but the error raising one will still execute 1e7 times in about 4 seconds. As soon as any complexity is added to the function I believe the gap closes pretty quickly as well. And if there are recursive calls and the exception pops through them all it only is about 2x slower than the simple function (for depth of 8).
    TLDR yeah throwing/catching exceptions is slower but this whole return speed thing probably isn't the bottleneck in the program so its well worth the amazing benefits it brings.

"""


    def __init__(self,x):
        super().__init__()
        self.x = x
        self.x = x
        self.x = x
    @staticmethod
    def build(p):
        """
        annotated_assignment_stmt ::=  augtarget ":" expression ["=" expression]
        Returns Assignment | Var | AttributeRef | Subscription | Slicing (note how the Assignment "=" bit is optional)
        """




# TODO the next thing to do is get target, target_list, and starred expressions down really well in terms of what they mean and parsing, because they're super important.
# TODO furthermore the idea of hierarchy seems good. I've been doing this linear left_recursive expansion method which is GOOD but perhaps one thing you can do is rework that to be a series of or_expr() and_expr() calls etc? Like literally follow the language model? Where or_expr can be anything below a bitwise `|`, etc. Or perhaps make expr take a keyword that's literally 'target' or 'or_expr'. Then make sure it composes with multiple targets ofc.
    ## it would be overkill to have anything below or_expr(). You should have or_expr, or_test, conditional_expression, expression, expression_list, expression_nocond, starred_expression(==starred_list), [target, target_list == not expressions but still imp]. No need for lambdas separate from expression[_nocond]. We're only writing these for exprs that are reused in other places.
    ## note that an expression_list

"""

token(tok) - consume a token if it matches and ret bool indicating success
keyword(kw) - consume a keyword if it matches and ret bool indicating success
identifier() - consume an ID and return its str on success or None on failure
empty() - ret bool indicating if self.idx is pointing beyond the end of self.elems
or_expr/or_test/target/target_list/etc - consume tokens and return AST node. Raise SyntaxError on failure.
p.must.fn() - assert that a fn like token/keyword/empty/etc did not return None or False. Very important.
p.or_none.fn() - Try to run a fn that might raise a SyntaxError and return normal results if no exceptions are raised but return None if SyntaxError is raised. This also will reset the token stream to wherever the call started on failure.

or_expr - this is a bitor/bitxor/bitand/shift/a/m/u/power_expr in that order


GRAMMAR RULES
1. `a ::= (b | c) d` requires parse(b) will fail iff parse(c) will succeed. IE exactly one RHS will parse successfully.
2. `a ::= (b | c) d` requires at most ONE of b,c to be in the expression trunk. Ie if one is expression the other cant be or_test. This is because expression is everything or_test can be and more.
3. All non-trunk expressions must be identifiable with k-lookahead
    If this were not true one could use `p.can` to parse them anyways, but also that probably indicates poor language design.
4. If a,b are trunk expressions and `a` is more expressive than `b` (ie `a` is after `b` in the trunk list) then no `b ::=` production rules can contain `a` directly. Note that a `b` rule can contain a third element `c` which then yields `a` but it can't directly contain `a` itself. An important note is that the element `c` must yield a Node -- something like .comp_iter is merely a grammar feature and for all intents and purposes we substitute its definition into its location in the BNF rules. So if something like comp_iter or any other rule started with `a` that would be invalid unless the rule itself yields an expression node.
Combining 2 and 3 we can say that at each decision point in the parsing process we can either determine via lookahead (often a single token) what non-trunk expression we have, and then if all lookaheads fail we can try the potential trunk expression and if that fails then we fail.

Stmts are lookahead identifiable other than ExpressionStmt, Asn, AugAsn are very slightly more complex.






*** writing out hierarchical DAG or tree for expressions.

stmtexpres require an intermed vairable, is that ok? is that right? Yes i think that is right, bc why would you ever wanna modify the result you could have just done that in ur final expr line! tho maybe w matches idkkkk. The issue w matches is the should be expressions by nature.
^^Yeah matches shd rly be expressions.


y = (match type(x) with |int->1 |bool->2)
y = (match type(x) with
    |int->1
    |bool->2)

y ::=
    op ::=
        match x with
            |3 -> x
            |_ -> x
    op+5

collapsing the header of a stmt onto the ::= line is allowed:
y ::=
    op ::= match x with:
        |3 -> x
        |_ -> x
    op+5

y ::=
    x = 3
    y = 2+x
    y

---
def aux():
    x=3
    y=2+x
    return y
y = aux()
---

y ::= x=3;y=2+x;y
---
def aux(): x=3;y=2+x;return y
y = aux()
---

"""

def locate(fn):
    """
    Decorator where if a function that takes a Parser as the first argument and returns a Node it'll tag that node with the Pos tags from the first and last tokens parsed
    """
    def wrapper(p,*args,**kwargs):
        assert isinstance(p,Parser)
        start_i = p.idx
        ret = fn(p,*args,**kwargs)
        end_i = p.idx-1
        start = p.elems[start_i].loc.start
        end = p.elems[end_i].loc.end
        if isinstance(ret,Node):
            ret.loc = Pos(start,end)
        return ret
    return wrapper


class Parser():
    def __init__(self,elems):
        super().__init__()
        self.elems = elems # list of Tokens/Atoms
        self.idx = 0
        self.must = AssertCallWrapper(self)
        #self.not_none = NotNoneCallWrapper(self) # or_none is most encouraged
        self.or_none = Postprocessor(self,None) # or_none is most encouraged
        self.or_fail = Postprocessor(self,FAIL)
        self.or_false = Postprocessor(self,False)
        self.or_ = (lambda retval: Postprocessor(self,retval))
        self.no_ws = False # indicates if whitespace after tokens is currently allowed
        self.no_ws_initial = False # used by no_whitespace()
        self.parsed = [] # list of token/atom lists that have been parsed. Used for .head and .body elements using `is` comparisons to check the pointers and see if theyve been parsed
    def save_state(self):
        def deepcopy(val): # esp important for self.parsed
            if isinstance(val,list):
                return [deepcopy(v) for v in val]
            if isinstance(val,tuple):
                return tuple([deepcopy(v) for v in val])
            if isinstance(val,dict):
                return [deepcopy(k):deepcopy(v) for k,v in val.items()]
            return val
        return deepcopy(self.__dict__)
    def load_state(self,state):
        self.__dict__ = state
    def next(self):
        if self.idx >= len(self.elems):
            raise SyntaxError("Calling next() when already passed the last elem")

        # starting on the second call to next() while in the no_whitespace() contextmanager, make sure that theres no whitespace trailing on the token 2 before whichever one next() will result in .curr pointing to.
        if self.no_ws and not self.no_ws_initial:
            if len(self.prev.verbatim) != len(self.prev.data): # True if has trailing whitespace
                raise SyntaxError


        if self.no_ws and self.no_ws_initial:
            self.no_ws_initial = False
        self.idx += 1
        return
    ## FUNDAMENTALS
    @property
    def tok(self):
        if self.idx >= len(self.elems):
            raise SyntaxError("Ran out of elems to consume")
        token = self.elems[self.idx]

        # if self.no_ws, raise SynErr on trailing whitespace
        if self.no_ws and isinstance(token,Tok):
            if len(token.data) != len(token.verbatim):
                raise SyntaxError

        return token
    @property
    def prev(self):
        if self.idx-1 >= len(self.elems):
            raise SyntaxError("Ran out of elems to consume")
        return self.elems[self.idx-1]
    def comma_list(self,*fns,nonempty=False,as_expr=False,allow_trailing_comma=True):
        """
        Decorator that calls fn() in a loop and returns a list of the results. List is empty if first call fails.
        as_expr means the list will be turned into a Tuple, or else a non-tuple value if its a list of length 1 that didn't end in a comma
        """
        ret = []
        while True:
            try:
                ret.append(self.logical_xor(*fns))
                if not self.or_false.token(','):
                    break
            except SyntaxError:
                break
        if nonempty and len(ret) == 0:
            raise SyntaxError
        if not allow_trailing_comma and self.prev.data == ',':
            raise SyntaxError
        if as_expr:
            if len(ret) == 1 and self.prev.dat != ',':
                return body[0] # eval to the single expression in the list
            return Tuple(body)

        return ret
    def list(self,*fns,nonempty=False):
        """
        Decorator that calls fn() in a loop and returns a list of the results. List is empty if first call fails.
        """
        ret = []
        while True:
            try:
                ret.append(self.logical_xor(*fns))
            except SyntaxError:
                break
        if nonempty and len(ret) == 0:
            raise SyntaxError
        return ret
    def token(self,tok_like):
        """
        Return True if curr tok is `tok_like`, else raise SyntaxError
        Note that `True` is just returned so it can be used with .or_false
        Step forward on success
        `tok_like` can be:
            str: for example '(' is the same as RPAREN. To be specific the str should equal self.curr.data
            CONSTANT: e.g. RPAREN
            list: a list of strs/CONSTANTs, and if any of them matches it succeeds. BINOPS is an example list.
        """
        if not isinstance(tok_like,(list,tuple)):
            tok_like = [tok_like] # wrap in a list

        data = self.curr.data
        if '  ' in data:
            # Turn multispaces into single spaces
            data = ' '.join(list(filter(None,data.split(' '))))
        for tok_or_str in tok_like:
            assert isinstance(tok_or_str,(str,CONSTANT))
            if self.curr.typ == tok_or_str or data == tok_or_str:
                self.next()
                return True
        raise SyntaxError(f"Failed to match token. Wanted: {tok} Got: {self.curr}")
    def keyword(self,kw):
        """
        Return True if next token is the keyword `kw` (str), else raise SyntaxError
        Step forward on success
        ///////Returns bool indicating if next token is the keyword `kw` (str). Step forward on success.
        """
        if self.curr.typ is KEYWORD and self.curr.data == kw:
            self.next()
            return True
        raise SyntaxError(f"Failed to match keyword. Wanted: {kw} Got: {self.curr}")
    def identifier(self):
        """
        Checks if curr tok is an identifier if so step forward and return identifier string, if not raise SyntaxError
        """
        if self.curr.typ is ID:
            self.next()
            return self.curr.data
        raise SyntaxError(f"Failed to get an identifier. Got: {self.curr}")
    def empty(self):
        """
        Bool indicating if reached end of token stream
        """
        return len(self.idx) >= len(self.elems)
    def assert_empty(self):
        if not self.empty():
            raise SyntaxError
    ## CONTEXT MANAGERS
    @contextmanager
    def peek(self):
        """
        contextmanager that rewinds token stream to wherever it started, and doesnt stop error propagation
        """
        state = self.save_state()
        try:
            yield None
        finally:
            self.load_state(state)
    @contextmanager
    def maybe(self):
        """
        contextmanager that rewinds token stream to wherever it started only if a SyntaxError is raised, and catches the error.
        """
        state = self.save_state()
        try:
            yield None
        except SyntaxError:
            self.load_state(state)
    @contextmanager
    def no_whitespace(self):
        """
        Precise behavior: contextmanager that will make .next() throw a SynErr if next() is trying to consume a token that is preceded by a token that doesn't have whitespace, except no SynErr is thrown on the first next() call in the contextmanager.
        Intuitively: There can't be any whitespace between the tokens consumed during this contextmanager. A SynErr will get thrown by next() at the first instant that this doesn't hold.
        """
        self.no_ws = True
        self.no_ws_initial = True
        try:
            yield None
        finally:
            self.no_ws = False

    """these are all contextmanagers that parse the contents of compound atoms, and also confirm that those parsers ran to completion"""
    def brackets(self):
        return self._compound_atom_parser(ABracket)
    def braces(self):
        return self._compound_atom_parser(ABrace)
    def parens(self):
        return self._compound_atom_parser(AParen)
    def quotes(self):
        return self._compound_atom_parser(AQuote)
    def brackets(self):
        return self._compound_atom_parser(ABracket)
    def head(self):
        return self._compound_atom_parser(AColonList,'head')
    def body(self):
        return self._compound_atom_parser(AColonList,'body')
    def simple_body(self):
        with self.body()
            return self.stmts()
    def simple_head(self,kw):
        with self.head()
            self.keyword(kw)
    def simple_block(self,kw):
        self.simple_head(kw)
        return self.simple_body()

    @contextmanager
    def _compound_atom_parser(self,atom_cls,attr='body'):
        if not isinstance(self.curr,atom_cls):
            raise SyntaxError(f"Attempting to parse {atom_cls} but found {self.curr}")
        elems_list = getattr(self.curr,attr)

        def isin(val,list): # like 'in' but uses `is` instead of `==`
            return any([x is val for x in list])

        if isin(elems_list,self.parsed):
            raise SyntaxError("Already parsed this elems_list successfully!")

        old_elems,old_idx = self.elems, self.idx
        self.elems = elems_list
        self.idx = 0
        try:
            yield None
            # the following only runs if no SynErr is raised in the whole `with` block
            if not self.empty():
                raise SyntaxError(f"Unparsed remaining contents of a {atom_cls} is an error")
        finally:
            self.elems = old_elems
            self.idx = old_idx

        # only runs on success
        self.parsed.append(elems_list)
        if isin(p.curr.body,self.parsed) and (not hasattr(self.curr,'head') or isin(p.curr.head,self.parsed)):
            self.next() # step forward past this compound atom if 'head' (if exists) and 'body' have both been parsed
    ## BNF TYPES
    def parameter_list(self,no_annotations=False):
        return self.build_node(Parameters,no_annotations=no_annotations)
    def expression_list(self,as_expr=True):
        """
        as_expr=True means return a Tuple or expression rather than a list of expressions. This is required in most cases.
        "Except when part of a list or set display, an expression list containing at least one comma yields a tuple. The length of the tuple is the number of expressions in the list. The expressions are evaluated from left to right" -> applies to starred_expression and starred_list too.
        """
        return p.comma_list(p.expression,nonempty=True,as_expr=as_expr)
    def starred_list(self,as_expr=True):
        """
        See `Parser.expression_list` for explanation of `as_expr`
        """
        def starred():
            self.token('*')
            return Starred(self.or_expr())
        ret = comma_list(starred,self.expression,nonempty=True,as_expr=as_expr)
    def starred_expression(self,as_expr=True):
        """
        See `Parser.expression_list` for explanation of `as_expr`
        """
        return self.starred_list(as_expr=as_expr)
    def expression_nocond():
        def lambda_expr():
            return self.build_node(Lambda,nocond=True)
        return self.logical_xor(lambda_expr,or_test)
    def comprehension():
        """
        comprehension ::=  expression comp_for
        """
        lhs = p.expression()
        return p.build_node(Comprehension,lhs)
    def comp_for(self):
        is_async = self.or_false.keyword('async')
        self.keyword('for')
        targets = self.target_list()
        self.keyword('in')
        iter = self.or_test()
        comp_iter = self.or_none.comp_iter()
        return CompFor(targets,iter,comp_iter)
    def comp_if(self):
        self.keyword('if')
        cond = self.expression_nocond()
        comp_iter = self.or_none.comp_iter()
        return CompIf(cond,comp_iter)
    def comp_iter(self):
        return self.logical_xor(self.comp_for,self.comp_if)
    def target(self):
        """
        | "(" [target_list] ")"
        | "[" [target_list] "]"
        | "*" target
        | identifier
        | attributeref
        | subscription
        | slicing
        """
        def parens_or_brackets():
            # try parens or brackets
            self.or_none.parens()
            contents = self.logical_xor(self.parens,self.brackets)
            targets = contents.or_none.target_list()
            contents.assert_empty()
            return targets # success with parens or brackets means return a target_list (or None)

        # star case
        def starred():
            self.token('*')
            return Starred(self.target())

        return self.logical_xor(parens_or_brackets,starred,p.augtarget)
    def augtarget(self):
        """
        augtarget     ::=  identifier | attributeref | subscription | slicing
        """
        e = self.primary()
        if e not in [Var,AttributeRef,Subscript,Slicing]:
            raise SyntaxError
        return e

    def argument_list(self):
        return self.build_node(Arguments)
    def target_list(p):
        return p.comma_list(p.target,nonempty=True)
    ## META FNS
    def logical_or(self,*option_fns):
        for fn in option_fns: # try each fn
            with self.maybe():
                return fn()
        raise SyntaxError
    def logical_xor(self,*option_fns):
        """
        A much slower version of logical_or that requires that EXACTLY 1 match is found among option_fns.
        """
        ret = []
        fns = []
        for fn in option_fns: # try each fn
            with self.peek(): # force rewind
                with self.maybe():
                    ret.append(fn())
                    fns.append(fn) # only if prev line succeeds
        if len(ret) > 1:
            raise Exception(f"Multiple option_fns are matching the same parser in logical_xor! {fns} {ret}")
        if len(fns) == len(ret) == 1:
            return fn() # we gotta actually execute it for real now since we just peek()'d before
        raise SyntaxError # 0 matches

    def alias(self,fn):
        """
        alias ::= fn() [as identifier]
        returns an Alias
        """
        def as_ident():
            self.keyword('as')
            return self.identifier()
        val = fn()
        aliased_name = self.or_none(as_ident)
        return Alias(val,aliased_name)
    def dotted_name(self):
        """
        dotted_name ::= identifier ("." identifier)*
        """
        def dot_name():
            p.token('.')
            return p.identifier()
        with p.no_whitespace():
            return [p.identifier()] + p.list(dot_name)


    ## BUILDING AND IDENTIFYING ODES
    def build_node(self,nodeclass,leftnode=None,**kwargs):
        assert not issubclass(nodeclass,(InitExpr,InitStmt)), f"Can't use build_node with {nodeclass} because it's an InitExpr or InitStmt. Most likely you want to find a ExprGenerator or StmtGenerator that builds this InitExpr/InitStmt and call that."
        assert hasattr(nodeclass,'build') or hasattr(nodeclass,'identify'), f"{nodeclass} must have either build() or identify() methods"

        def output_check(node):
            assert node is not None, f"{nodeclass}.build() returned None, you probably forgot to return the built node"
            if not isinstance(node,(ExprGenerator,StmtGenerator)):
                assert type(node) == nodeclass, f"{nodeclass}.build() did not return an expression of type {nodeclass}. This is only allowed in ExprGenerators and StmtGenerators and is otherwise likely unintentional"

        # identify() case
        if hasattr(nodeclass,identify):
            with self.peek(): # will reset token stream but wont catch errors
                cls = nodeclass.identify(self) # may throw SyntaxError
                if cls is None:
                    raise SyntaxError
            node = self.build_node(cls,leftnode=leftnode,**kwargs)
            return node

        # left_recursive case
        if isinstance(nodeclass,Expr) and nodeclass.left_recursive:
            assert leftnode is not None, "Can't call build_node on a left_recursive node without a non-None leftnode"
            node = nodeclass.build(self,leftnode=leftnode,**kwargs)
            assert node is not None, f"{nodeclass}.build returned None"
            return node

        # not left_recursive case
        assert leftnode is None, "Can't call build_node on a NON left_recursive node with a non-None leftnode"
        node = nodeclass.build(self,**kwargs)
        assert node is not None, f"{nodeclass}.build returned None"
        return node
    def trunk_expr(self, type):
        """
        Parse an expression that lies at expression type `type` (e.g. 'or_test') along the expr trunk.

        In stage 1 we generate the leftmost subexpression, and in stage 2 we repeatedly expand it by left-recursive expansion.
        Stage 1 must yield a node that's valid for `type` as implied by Grammar Rule 4, and likewise each recursive expansion must yield a node that's valid for `type` (also Grammar Rule 4). I do not believe this will seriously limit our grammar, rather it's part of what it means to be in the `trunk`. Note that of course Nodes created by trunk_expr can internally call trunk_expr with more expressive types.
        """
        nodeclasses = get_trunk_nodes(type)

        # make fns that can be logical_or()'d together
        non_recursive = []
        for cls in nodeclasses:
            if not nodeclass.left_recursive:
                non_recursive.append(lambda:build_node(cls))

        node = self.logical_xor(*non_recursive) # this can throw synerr

        # left-recursion to extend this lefthand expression as much as possible
        ty = trunk_type_of_node(node)
        while True:
            # populate `left_recursive` for use with logical_or()
            left_recursive = []
            for cls in nodeclasses:
                if nodeclass.left_recursive and ty in nodeclass.left_types:
                    # key difference is `leftnode=node`
                    non_recursive.append(lambda:build_node(cls,leftnode=node))

            node_or_none = self.or_none.logical_xor(*non_recursive)
            if node is None: # exit on failure to expand more
                break
            node = node_or_none

            # exit if empty
            if self.empty():
                break

        return node
    def stmt(self):
        """
        Same deal as trunk_expr have something like an @decorator for labelling class with a keyword
        """
        if isinstance(self.curr,AColonList):
            # compound statement
            potential_kw = self.curr.head[0]
            kw = None if potential_kw.typ is not KEYWORD else potential_kw.data
            classes = compound_stmt_nodes[kw]
        else:
            # simple statement
            potential_kw = self.curr
            kw = None if potential_kw.typ is not KEYWORD else potential_kw.data
            classes = simple_stmt_nodes[kw]

        if len(classes) == 0:
            raise SyntaxError(f"No valid classes found for leading keyword {potential_kw}"

        build_fns = []
        for cls in classes:
            build_fns.append(lambda:self.build_node(cls))

        return p.logical_xor(*build_fns)
    def __getattr__(self,key):
        if key in trunk_nodes: # automatically adds methods for all trunk nodes!
            return self.trunk_expr(key)
        return object.__getattribute__(self,key)


class Postprocessor:
    """
    Return normal result if success, return FAIL if failure and reset the tokenstream to where it was at the start of the call.
    """
    def __init__(self,parser,retval):
        super().__init__()
        self.parser = parser
        self.retval = retval
    def __call__(self,fn):
        """
        Use like p.or_none(some_fn)
        Note some_fn must take no arguments
        """
        state = self.parser.save_state()
        try:
            return fn()
        except SyntaxError:
            self.parser.load_state(state)
            return self.retval
    def __getattr__(self,key):
        """
        A shorthand for or_none(some_fn) which also allows some_fn to take arguments rather than having an empty arg list. Note this only works for functions that are methods of `Parser`. Otherwise you need to use or_none(some_argless_fn).
        ret = p.or_none.some_fn(*args,**kwargs)
        is the same as:
        def argless_fn():
            p.some_fn(*args,**kwargs)
        ret = p.or_none(argless_fn)
        p.or_none(some_fn_argless)
        """
        fn = getattr(self.parser,key)
        assert callable(fn)
        def ret(*args,**kwargs): # wrapper that takes any args
            def argless_fn(): # quickly define an argless function
                fn(*args,**kwargs)
            return self(argless_fn) # call it with __call__ to do the normal `retval` and stream reset stuff
        return ret

#class NotNoneCallWrapper:
#    """
#    Assert that return value is not None -- real crash (not SyntaxError)
#    If the fn returns a SyntaxError is avoids this and nothing happens (intentionally)
#    """
#    def __init__(self,parser):
#        super().__init__()
#        self.parser = parser
#    def __getattr__(self,key):
#        fn = getattr(self.parser,key)
#        def wrapper(*args,**kwargs):
#            idx = self.parser.idx
#            ret = fn(*args,**kwargs)
#            assert ret is not None
#            return ret
#        return wrapper

class AssertCallWrapper:
    """
    SyntaxError if return value is not False and is not None
    """
    def __init__(self,parser):
        super().__init__()
        self.parser = parser
    def __getattr__(self,key):
        fn = getattr(self.parser,key)
        def wrapper(*args,**kwargs):
            idx = self.parser.idx
            ret = fn(*args,**kwargs)
            if ret is None or ret is False:
                raise SyntaxError("`.must` wrapper fired")
            return ret
        return wrapper




"""
LRM / official grammar notes:
expression_nocond is the same as expression except it can't be a Ternary expression.
Note that *_nocond is only really used when embedded in some grammar that has an "if" following an expression, like in comprehensions. Thus *nocond only really exists to simplify stuff




primary ::=  atom | attributeref | subscription | slicing | call
or_expr: The weakest of the numerical expressions (BitOr), anything past this can yield booleans. Any expr with an `|` or tighter as the weakest thing. Note this is a `|` not an `or`. Parent of bitxor/bitand/shift/a/m/u/power_expr (in that order. u=unary, a='+' m='*'). Note that this is maybe the highest overridable operation? Which is prob why it's used in starred_expression?
comparison: Any expr with a CMPOP or tighter as the weakest thing. or_exprs are the weakest thing that comparisons beat.
or_test: Any expr with an "or" or tighter as the weakest thing. Parent of and/not/comparison in that order.
conditional_expression(==Ternary): Ternary--. Direct parent of or_test.
expression: Lambda | Ternary--
expression_list: [expression--]+
starred_list==starred_expression: [expression-- | "*" or_expr--]+

So basically a starred_list==starred_expression== the top of the hierarchy (aside from stuff like [] brackets and such) and can be either an expression-- (==Lambda | Ternary--) or a starred or_expr--. Note that or_expr is the weakest "numerical" thing, anything above that gets into booleans.





Basically _nocond versions have no Ternary so they're or_test-- and the same applies to lambda bodies within them:
    >expression_nocond: or_test-- or lambda but the lambda body is also stuck with or_test--.
    >lambda_expr: "lambda" [parameter_list] ":" expression
    >lambda_expr_nocond: "lambda" [parameter_list] ":" (or_test-- | lambda_expr_nocond)






await_expr ::= "await" primary


"""



# ===TODO===
## WHERE I LEFT OFF
# Implement the rest of the Expr nodes
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
    while len(stack) > 1 and isinstance(stack[-1],ACompoundStmt):
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
        Figuring out our start and end locations in the original text. Start location is based on the token that created us for Atoms that aren't ACompoundStmts, for AColonList it's the start of the head, for AMasterList it's line 1 char 1.
        """

        # loc.start
        if isinstance(self,ACompoundStmt):
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
            else: # non ACompoundStmt
                self.loc_end(self.start_tok)

        self.finished = True # must come after self.finish_stmt()
    def __repr__(self):
        body = ' '.join([repr(elem) for elem in self.body])
        return u.mk_b(f"{self.name}(") + f"{body}" + u.mk_b(")")

class AStmt(Atom):
    """
    .body: (Atom|Tok) list
    """
    pass

class ACompoundStmt(Atom):
    """
    .head: (Atom|Tok) list | None  -- None if AMasterList
    .body
    """
    def __init__(self,tok):
        super().__init__(tok)
        self.curr_stmt = []
        self.closer = None # only gets closed by the linestart_trypop function
        self.parsed = []
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
        ret = self.curr_stmt
        self.curr_stmt = []
        return ret
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
            if len(line) == 1 and isinstance(line[0],ACompoundStmt):
                res += line[0].repr_depth(depth=depth+1) # print stmt list
            else:
                res += f'{block_indent}' # only need for nonstmtlists bc stmtlists handle indents themselves
                for elem in line: # print a line full of non-stmtlist elements
                    assert not isinstance(elem,ACompoundStmt), "ACompoundStmt should only appear as the only element on a line if it appears"
                    res += repr(elem) + ' '
                res = res[:-1] # kill last space
        return res

class AMasterList(ACompoundStmt):
    def __init__(self):
        super().__init__(None) # There is no token that created AMasterList
        self.allowed_children = COMMON_ALLOWED_CHILDREN | {COLON} # set union
        self.error_on = COMMON_ERROR_ON
    def linestart_trypop(self,linedata):
        if linedata.leading_whitespace != '':
            raise IndentationError(f"unexpected indent: should have zero indent")
        return False # never pops

SEMICOLONS_ONLY = -1

class AColonList(ACompoundStmt):
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

"""
Note the advanced class decorator they suggested is not great.
A better one would just be a fn that replaces Foo.fn = thedecorator(Foo.fn) instead of crazy getattribute stuff. It can use types to figure out what attrs are functions.
"""


#    # simple statements that don't start with keywords
#    e, rest = expr(elems)
#    # ExpressionStmt
#    if rest == []:
#        return ExpressionStmt(elems) # to make a .identify for this probably just have it run .expr_assert_empty with fail=BOOL or whatever
#    # Asn
#    if istoken(rest,0,EQ):
#        return Asn(elems)
#    # AugAsn
#    if istoken(rest,0,BINOPS) and istoken(rest,1,EQ):
#        return AugAsn(elems)

"""
Returns the first valid expression that starts at elems[0] and is not the subexpression of a larger expression in `elems`, and also returns the remains of `elems`

note: don't worry if you see stuff like ABracket being translated into a dict always. That's only what happens when we ask for an expression and it START with {}. You can still make your own nondict syntax using {} as long as you don't start a generic expr with the '{' symbol. If you do want to override the kinds of things that can happen when an expr starts with '{' you are also welcome to do that, just modify the code to look harder at it and decide between Dict() and whatever your AST node is.

"""


# fn that returns a class decorator
def left_recursive(left_type):
    """
    left_type is the trunk type of the left hand expression that it is valid for us to expand on.
    """
    def class_decorator(cls):
        assert isinstance(cls,Expr)
        cls.left_recursive = True
        cls.left_types = get_trunk_types(left_type) if left_type is not None else None
        return cls
    return class_decorator

# super simple decorator class Add:pass then Add.gen_build('m_expr') is the same as @gen_build('m_expr') then class Add:pass
def gen_build(*args,**kwargs):
    def class_decorator(cls):
        assert hasattr(cls,'gen_build')
        assert isinstance(cls.gen_build,classmethod)
        cls.build = cls.gen_build(*args,**kwargs)
        assert cls.build is not None, f"{cls}.gen_build() returned None instead of a function"
        if not isinstance(cls.build,staticmethod):
            cls.build = staticmethod(cls.build) # wraps in staticmethod for convenience if you forgot
        return cls
    return class_decorator


simple_stmt_nodes = defaultdict(list)
compound_stmt_nodes = defaultdict(list)
# `None` is the key for no leading kw

# class decorator for Stmts
def simple(*kws):
    """
    `kw` is the leading keyword for the Stmt subclass this is decorating
    """
    def class_decorator(cls):
        assert isinstance(cls,Stmt)
        assert not isinstance(cls,InitStmt)
        cls.leading_kws = kws
        cls.type = 'simple'
        [simple_stmt_nodes[kw].append(cls) for kw in kws]
        assert len(kws) != 0, "Please specify at least one leading keyword in @simple(*kws). Specify None if there is not leading keyword"
    return class_decorator

# class decorator for Stmts
def compound(*kws):
    """
    `kw` is the leading keyword for the Stmt subclass this is decorating. None if theres no leading kw
    """
    def class_decorator(cls):
        assert isinstance(cls,Stmt)
        assert not isinstance(cls,InitStmt)
        cls.leading_kws = kws
        cls.type = 'compound'
        [compound_stmt_nodes[kw].append(cls) for kw in kws]
        assert len(kws) != 0, "Please specify at least one leading keyword in @compound(*kws). Specify None if there is not leading keyword"
        return cls
    return class_decorator

class Stmt(Node):
    leading_kw = None # gets set to True by @leading_kw decorator if theres a leading keyword

class Expr(Node):
    left_recursive = False # gets set to True by @left_recursive decorator
    left_types = None # gets set by @left_recursive decorator
    annotation = None
    def __init__(self):
        super().__init__()
        if self.left_recursive:
            assert self.left_type is not None, f"You need to left_recursive decorate your subclass of a left_recursive(None) node! {self}"
    def annotate(self,annotation):
        self.annotation = annotation
        return self

# An Expr with no build/identify functions so it must be created directly through __init__ (which takes whatever its components are, NOT a Parser.
class InitExpr(Expr): pass
class InitStmt(Stmt): pass
# A Node which never shows up in the AST, when you call build() it just returns some other Expr/Stmt node, possibly from a range of options.
class ExprGenerator(Expr): pass
class StmtGenerator(Stmt): pass
# Things like Parameters and Args that do not evaluate to a value and thus are not Exprs.
class AuxNode(Node): pass
# Things like Binop that never exist in the AST but are useful parent classes to have for isinstance() calls and inheritance. Unlike ExprGenerator the things it produces are actual children of it.
class AbstractExpr(Expr): pass



"""
The `trunk` is an (ordered) list of expression with names corresponding to LRM grammar names and all MUST have the property that if they appear lower in the list they include everything above them in the list. So the rule `a ::= b` must be a valid production rule for any a,b pair where a is below b.
It can't be `a ::= 'if' b` or `a ::= '(' b ')'` or anything else like that, it has to be that something like `comparison` can literally just be a single atom/enclosure/power/u_expr/etc without any other syntax necessary.

"""
def get_trunk_nodes(type):
    """
    Takes a type like 'primary' and returns all node classes from that type and all types within it, all in a flattened list.
    """
    assert type in trunk
    idx = trunk_nodes.keys().index(type)
    list_of_tuples = trunk_nodes.values()[:idx+1]
    flattened = itertools.chain(*list_of_tuples)
    return flattened

def get_trunk_types(type):
    """
    Takes a type like 'primary' and returns all types within it like ['enclosure','atom','primary']
    """
    assert type in trunk
    idx = trunk_nodes.keys().index(type)
    types = trunk_nodes.keys()[:idx+1]
    return types

def trunk_type_of_node(node):
    """
    Takes a Node instance and returns its trunk type
    """
    assert isinstance(node,Node)
    for type,cls_list in trunk_nodes.items():
        for cls in cls_list:
            if isinstance(node,cls):
                return type


trunk_nodes = {
        """
        Note contents of 'enclosure' are all converted from BNF form like so: some_name -> SomeName
        """
        'enclosure':ListDisplay,DictDisplay,SetDisplay,GeneratorExpression,YieldAtom,ParenthForm # tightest binding
        'atom':Var,Lit
        'primary':AttributeRef,SubscriptOrSlicing,Slicing,Call
        'await_or_primary':AwaitExpr
        'power':Exp,
        'u_expr':UAdd,USub,Invert
        'm_expr'Mul,Div,FloorDiv,Mod,MatMul
        'a_expr':Add,Sub
        'shift_expr':ShiftL,ShiftR
        'and_expr':BitAnd,
        'xor_expr':BitXor,
        'or_expr':BitOr,
        'comparison':Compare,
        'not_test':Not,
        'and_test':And,
        'or_test':Or,
        'conditional_expression':Ternary,
        'expression':Lambda, # weakest binding
        #'starred_expression':StarredExpr, # == starred_list
        }






# TODO note that the .usage of things often isn't known when they're created initially, and is rather added during left-recursion when the target becomes part of a larger statement. So it should really be up to the larger statement to update the .usage for its targets. In other cases it is known when you call expr() for example when already inside a larger statement and calling expr to construct a nonleftrecursive smaller expr. Also ExpressionStmt for example could set things to LOAD, etc.



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

Var List Tuple Starred AttributeRef Subscript should assert that .usage is not None when provided


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
ExpressionStmt:
    .expr: Expr
Asn:
    .targets: [Reference]
    .val: Expr
AugAsn:
    .target: Name | Subscript | AttributeRefibute (not Tuple | List)
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
    .func: Name | AttributeRef | some other stuff
    .args: [Expr]
    .keywords: [Keyword]
  Keyword(namedtuple):
      .name: str
      .val: Expr
Ternary:
    .cond: Expr
    .if_branch: Expr
    .else_branch: Expr
AttributeRef:
    .expr: Expr
    .attr: str
    .usage: USAGE
Subscript:
    .expr: Expr
    .index: Expr
    .usage: USAGE
Slicing:
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


Target = Var | AttributeRef | Subscript | Slicing | '*' Target
Reference = Var | AttributeRefibute | Subscript


"""

Pos = namedtuple('Pos', 'line char')
# these numbers are inclusive as start* and exclusive as end*
# so endchar is one past the last char
Loc = namedtuple('Loc', 'start end')




class LocTagged:
    def __init__(self):
        super().__init__()
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




def stmts(elem_list_list): # [[elem]] -> [Stmt]
    the_stmts = []
    for elems in elem_list_list:
        the_stmt = stmt(elems)
        if the_stmt.dependent and the_stmts[-1].offer_neighbor(the_stmt):
            continue # e.g. Try consuming Except by accepting it as a neighbor
        elif the_stmt.dependent:
            raise SyntaxError(f"dependent statement {the_stmt} not accepted by neighbor {the_stmts[-1]}")
        the_stmts.append(the_stmt)


# STMTS

class Stmt(Node): # abstract
    def __init__(self):
        super().__init__()
        self.dependent = False
    def offer_neighbor(self, stmt):
        """
        All statements that want to consume their same-indent-level neighbors should override this, for example Try should override it to add stmt to its self.excepts if stmt is an Except node, and should return True to indicate that the stmt no longer needs to be added to the normal stmt list (bc instead it's inside of Try)
        """
        return False

## Compound Statements

@leading_kw('def')
class FuncDef(Stmt):
    def __init__(self,name,args,body):
        self.name = name
        self.args = args
        self.body = body
    @staticmethod
    def build(p):
        head,body = p.colonlist()

        head.keyword("def")
        name = head.identifier()
        args = head.formals()
        head.assert_empty()
        body = body.stmts()
        body.assert_empty()
        return FuncDef(name,args,body)


class Module(Stmt):
    def __init__(self,compound):
        super().__init__()
        self.body = stmts(compound.body)

class ClassDef(Stmt):
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

class Arg(AuxNode):
    def __init__(self,name):
        super().__init__()
        self.name = name

class Arguments(AuxNode):
    def __init__(self,args):
        self.args = args
    @staticmethod
    def build(p):
        super().__init__()

        """
        There are 3 stages and there's no going backwards:
        1.Comma-sep list mix of Expr | *Expr
            (Expr | *Expr) list
        2.As soons as an ident=Expr shows up, switch to a comma-sep list of (ident=Expr) | *Expr
            ((ident=Expr) | *Expr) list
        3.As soon as a **Expr shows up, switch a comma-sep list of (ident=Expr) | **Expr
            ((ident=Expr) | **Expr) list
        A trailing comma at the end is okay
        """
        p = p.parens()

        def stage1():
            if p.or_false.token('*'):
                return Starred(Arg(p.identifier()))
            return Arg(p.identifier())
        def stage2():
            if p.or_false.token('*'):
                return Starred(Arg(p.identifier()))
            name = p.identifier()
            p.token('=')
            val = p.expression()
            return KVPair(name,val)
        def stage3():
            if p.or_false.token('**'):
                return DoubleStarred(Arg(p.identifier()))
            name = p.identifier()
            p.token('=')
            val = p.expression()
            return KVPair(name,val)

        args = p.comma_list(stage1) + p.comma_list(stage2) + p.comma_list(stage3)
        p.assert_empty()
        return Arguments(args)

class Parameters(AuxNode):
    def __init__(self,args,stararg,kwonlyargs,doublestararg):
        super().__init__()
        self.args = args
        self.stararg = stararg
        self.kwonlyargs = kwonlyargs
        self.doublestararg = doublestararg
    @staticmethod
    def argless():
        return Parameters(args=[],stararg=None,kwonlyargs=[],doublestararg=None)
    @staticmethod
    def build(p,no_annotations=False):
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
        Any of these can have an annotation like ident:expression or ident:expression=expression
        """

        def annotation():
            if no_annotations:
                return None
            p.token(':')
            annotation = p.expression()
            return annotation

        def stage1_4():
            name = p.identifier()
            ann = p.or_none(annotation)
            return Arg(name).annotate(ann)
        def stage2_5():
            name = p.identifier()
            ann = p.or_none(annotation)
            p.token('=')
            val = p.expression()
            return KVPair(name,val).annotate(ann)
        def stage3():
            p.token('*')
            name = p.or_('').identifier()
            ann = annotation()
            return Var(name).annotate(ann)
        def stage6():
            p.token('**')
            name = p.or_('').identifier()
            ann = annotation()
            return Var(name).annotate(ann)

        args = p.comma_list(stage1_4) + p.comma_list(stage2_5)
        stararg = p.or_none(stage3)
        kwonlyargs = p.comma_list(stage1_4) + p.comma_list(stage2_5)
        doublestararg = p.or_none(stage6)

        return Formals(args,stararg,kwonlyargs,doublestararg)

Elif = namedtuple('Elif','cond body') # expression, (Stmt list)
@compound('if')
class If(Stmt):
    def __init__(self,cond,body,elifs,else_body):
        super().__init__()
        self.cond = cond # expression
        self.body = body # Stmt list
        self.elifs = elifs # Elif list
        self.else_body = else_body # (Stmt list) | None
    @staticmethod
    def build(p):
        """
        if_stmt ::=  "if" expression ":" suite
             ("elif" expression ":" suite)*
             ["else" ":" suite]
        """
        with p.head():
            p.keyword('if')
            cond = p.expression()
        with p.body():
            body =p.stmt_list()
        def elif_fn():
            with p.head():
                if not p.or_false.keyword('elif'):
                    p.keyword('else')
                    p.keyword('if')
                elif_cond = p.expression()
            elif_body = p.simple_body()
            return Elif(elif_cond,elif_body)
        elifs = p.list(elif_fn)
        def else_fn():
            return p.simple_block('else')
        else_body = p.or_none(else_fn)
        return If(cond,body,elifs,else_body)

@compound('while')
class While(Stmt):
    def __init__(self,cond,body,elifs,else_body):
        super().__init__()
        self.cond = cond # expression
        self.body = body # Stmt list
        self.else_body = else_body # (Stmt list) | None
    @staticmethod
    def build(p):
        """
        while_stmt ::=  "while" expression ":" suite
                ["else" ":" suite]
        """
        with p.head():
            p.keyword('while')
            cond = p.expression()
        with p.body():
            body = p.stmt_list()
        def else_fn():
            return p.simple_block('else')
        else_body = p.or_none(else_fn)
        return While(cond,body,else_body)


@compound('for','async')
class For(Stmt):
    def __init__(self,targets,iter,body,else_body,is_async):
        super().__init__()
        self.targets = targets # target list
        self.iter = iter # Tuple | expression
        self.body = body # Stmt list
        self.else_body = else_body # (Stmt list) | None
        self.is_async = is_async
    @staticmethod
    def build(p):
        """
        for_stmt ::=  "for" target_list "in" expression_list ":" suite
              ["else" ":" suite]
        async_for_stmt ::=  "async" for_stmt
        """
        with p.head():
            is_async = p.or_false.keyword('async')
            p.keyword('for')
            targets = p.target_list()
            p.keyword('in')
            iter = p.expression_list()
        body = p.simple_body()
        def else_fn():
            return p.simple_block('else')
        else_body = p.or_none(else_fn)
        return For(targets,iter,body,else_body)

namedtuple('Except','exc body') # Alias, (Stmt list)
@compound('try')
class Try(Stmt):
    def __init__(self,body,excepts,else_body,finally_body)
        super().__init__()
        self.body = body # Stmt list
        self.excepts = excepts # Except list
        self.else_body = else_body # Stmt list
        self.finally_body = finally_body # Stmt list
    @staticmethod
    def build(p):
        """
        try_stmt  ::=  try1_stmt | try2_stmt
        try1_stmt ::=  "try" ":" suite
                       ("except" [expression ["as" identifier]] ":" suite)+
                       ["else" ":" suite]
                       ["finally" ":" suite]
        try2_stmt ::=  "try" ":" suite
                       "finally" ":" suite
        """
        body = p.simple_block('try')
        def except_block():
            with p.head():
                p.keyword('except')
                exc = p.or_none.alias(p.expression)
            exc_body = p.simple_body()
            return Except(exc,exc_body)
        def else_fn():
            return p.simple_block('else')
        def finally_fn():
            return p.simple_block('finally')

        def try1():
            p.list(except_block,nonempty=True)
            else_body = p.or_none(else_fn)
            finally_body = p.or_none(finally_fn)
            return Try(body,excepts,else_body,finally_body)
        def try2():
            finally_body = finally_fn()
            return Try(body,None,None,finally_body)

        return p.logical_xor(try1,try2)

@compound('with')
class With(Stmt):
    def __init__(self,aliases,body,is_async)
        super().__init__()
        self.aliases = aliases
        self.body = body
        self.is_async = is_async
    @staticmethod
    def build(p):
        """
        with_stmt ::=  "with" with_item ("," with_item)* ":" suite
        async_with_stmt ::=  "async" with_stmt
        with_item ::=  expression ["as" target]
        """
        def with_item():
            return p.alias(p.expression)
        with p.head():
            is_async = p.or_false.keyword('async')
            p.keyword('with')
            aliases = p.comma_list(with_item,nonempty=True,allow_trailing_comma=False)
        body = p.simple_body()
        return With(aliases,body,is_async)

class Decorator(AuxNode):
    def __init__(self,name_list,args):
        self.name_list = name_list # str list
        self.args = args # Arguments | None
    @staticmethod
    def build(p):
        p.token('@')
        name_list = p.dotted_name()
        args = p.or_none.argument_list()
        return Decorator(name_list,args)


@compound('def','@')
class FuncDef(Stmt):
    def __init__(self,decorators,fname,params,body,is_async)
        super().__init__()
        self.decorators = decorators # Decorator
        self.fname = fname # str
        self.params = params # Parameters
        self.body = body # Stmt list
        self.is_async = is_async # bool
    @staticmethod
    def build(p):
        """
        funcdef                 ::=  [decorators] "def" funcname "(" [parameter_list] ")"
                                     ["->" expression] ":" suite
        async_funcdef ::=  [decorators] "async" "def" funcname "(" [parameter_list] ")"
                           ["->" expression] ":" suite
        decorators              ::=  decorator+
        decorator               ::=  "@" dotted_name ["(" [argument_list [","]] ")"] NEWLINE
        dotted_name             ::=  identifier ("." identifier)*
        parameter_list          ::=  defparameter ("," defparameter)* ["," [parameter_list_starargs]]
                                     | parameter_list_starargs
        parameter_list_starargs ::=  "*" [parameter] ("," defparameter)* ["," ["**" parameter [","]]]
                                     | "**" parameter [","]
        parameter               ::=  identifier [":" expression]
        defparameter            ::=  parameter ["=" expression]
        funcname                ::=  identifier
        """

        def decorator():
            p.build_node(Decorator)
            p.next_line()
        decorators = p.list(decorator)
        with p.head():
            is_async = p.or_false.keyword('async')
            p.keyword('def')
            fname = p.identifier()
            params = p.parameter_list()
            def annotation():
                p.token('->')
                return p.expression()
            ann = p.or_none.annotation()
        body = p.simple_body()

        return FuncDef(decorators,fname,params,body,is_async).annotate(ann)

@compound('class','@')
class ClassDef(Stmt):
    def __init__(self,decorators,fname,inheritance,body)
        super().__init__()
        self.decorators = decorators # Decorator
        self.name = name # str
        self.inheritance = inheritance # Arguments
        self.body = body # Stmt list
    @staticmethod
    def build(p):
        """
        classdef    ::=  [decorators] "class" classname [inheritance] ":" suite
        inheritance ::=  "(" [argument_list] ")"
        classname   ::=  identifier
        """
        def decorator():
            p.build_node(Decorator)
            p.next_line()
        decorators = p.list(decorator)
        with p.head():
            p.keyword('class')
            name = p.identifier()
            inheritance = p.argument_list()
        body = p.simple_body()

        return ClassDef(decorators,name,inheritance,body)





class CONSTANT:
    instances = []
    def __init__(self,name,**kwargs):
        super().__init__()
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

class Elif(Stmt):
    def __init__(self,compound):
        super().__init__()
        head = compound.head
        self.dependent = True

        status,head = keyword(head,"elif",fail=FAIL)
        if status is FAIL:
            head = keyword(head,"else")
            head = keyword(head,"if")
        #if not head.keyword("elif"):
        #    head.keyword("else")
        #    head.keyword("if")
        self.cond = expr(head)
        empty(head)
        self.body = stmts(compound.body)

class Else(Stmt):
    def __init__(self,compound):
        super().__init__()
        head = compound.head
        self.dependent = True

        head = keyword(head,"else")
        empty(head)
        self.body = stmts(compound.body)


class For(Stmt):
    def __init__(self,compound):
        super().__init__()
        head = compound.head

        head = keyword(head,"for")
        self.target,head = target(head)
        head = keyword(head,"in")
        self.iter,head = expr(head)
        empty(head)
        self.body = stmts(compound.body)

class While(Stmt):
    def __init__(self,compound):
        super().__init__()
        head = compound.head

        head = keyword(head,"while")
        self.cond,head = expr(head)
        empty(head)
        self.body = stmts(compound.body)

class With(Stmt):
    def __init__(self,compound):
        super().__init__()
        head = compound.head
        self.withitems = []

        head.keyword("with")
        while not head.empty():
            contextmanager = head.expression()
            target = None
            if head.keyword('as'):
                target = head.target()
            self.withitems.append(Withitem(contextmanager,target))

        if len(self.withitems) == 0
            raise SyntaxError("empty `with` statement header")
        self.body = stmts(compound.body)

class With(Stmt):
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
class Withitem(AuxNode):
    def __init__(self,contextmanager,target):
        super().__init__()
        self.contextmanager = contextmanager
        self.target = target
class Try(Stmt):
    def __init__(self,compound):
        super().__init__()
        head = compound.head

        head = keyword(head,"try")
        empty(head)
        self.body = stmts(compound.body)



## Simple Statements

# `by` and `till` strings, tokens, or functions
# VERY IMPORTANT: if you split by commas in particular, the final sublist will be discarded if it is an empty list. This is because this is the desired behavior the vast majority of the time (for example foo(a,) is a valid function call on the argument `a` (NOT the tuple `(a,)`). Likewise `def foo(a,)` is allowed, `lambda x,:` is allowed.
# Also throws an error if there are two commas in a row only applies to `by`=',' again
"""CAREFUL. split should never be used if an expr may be somewhere inside whatever you're splitting. Because for example commas within lambda expressions are valid, or colons within lambdas -- if you split by commas or colons (e.g. when parsing slicing) it would get messed up. See examples of code like in Slicing parsing for how to properly deal with these cases"""
def unsafe_split(): # [elem] -> [[elem]]
    raise NotImplementedError



def join(): # [[elem]] -> [elem]
    raise NotImplementedError

# [elem] -> | str         if `till` is None
#           | str,[elem]  if `till` is not None
# as usual `till` can be a string/token/function
def raw_string():
    raise NotImplementedError

@simple(None)
class ExpressionStmt(Stmt):
    def __init__(self,expr):
        super().__init__()
        self.expr = expr
    @staticmethod
    def build(p):
        expr = p.starred_expression()
        return ExpressionStmt(expr)



@simple(None)
class Assignment(Stmt):
    def __init__(self,targets,val):
        super().__init__()
        self.targets = targets
        self.val = val
    @staticmethod
    def build(p):
        """
        assignment_stmt ::=  (target_list "=")+ (starred_expression | yield_expression)
        """
        def target_lists():
            p.target_list()
            p.token('=')
        targets = p.list(target_list,nonempty=True)
        val = p.logical_xor(p.starred_expression,lambda:p.build_node(YieldExpression))
        return Assignment(targets,val)

AUGOPS = ["+" , "-" , "*" , "@" , "/" , "//" , "%" , "**" , ">>" , "<<" , "&" , "^" , "|"]

@simple(None)
class AugAsn(Stmt):
    """
    augmented_assignment_stmt ::=  augtarget augop (expression_list | yield_expression)
    augtarget                 ::=  identifier | attributeref | subscription | slicing
    augop                     ::=  "+=" | "-=" | "*=" | "@=" | "/=" | "//=" | "%=" | "**="
                                   | ">>=" | "<<=" | "&=" | "^=" | "|="
    """
    def __init__(target,op,val):
        super().__init__()
        self.target = target
        self.op = op # + - etc without the equals sign
        self.val = val
    @staticmethod
    def build(p):
        target = p.augtarget()

        # dealing with operator
        op = p.curr.typ
        no_trailing_ws = (len(p.curr.verbatim) == len(p.curr.data))
        p.token(AUGOPS)
        if not no_trailing_ws: # e.g. "+ ="
            raise SyntaxError
        p.token("=")

        val = p.logical_xor(p.expression_list,lambda:p.build_node(YieldExpression))
        return Assignment(target,op,val)

@simple(None)
class AnnotatedAssignment(ExprGenerator):
    @staticmethod
    def build(p):
        """
        annotated_assignment_stmt ::=  augtarget ":" expression ["=" expression]
        Returns Assignment | Var | AttributeRef | Subscription | Slicing (note how the Assignment "=" bit is optional)
        """
        target = p.augtarget()
        p.token(':')
        target.annotate(p.expression())
        def asn():
            p.token('=')
            return p.expression()
        val = p.or_none(asn)
        if val is None:
            return target
        return Assignment([target],val)


@simple('assert')
class Assert(Stmt):
    def __init__(self,expr,msg):
        super().__init__()
        self.expr = expr
        self.msg = msg
    @staticmethod
    def build(p):
        """
        assert_stmt ::=  "assert" expression ["," expression]
        """
        p.keyword('assert')
        expr = p.expression
        def msg():
            p.token(',')
            return p.expression()
        msg = p.or_none(msg)
        return Assert(expr,msg)

@simple('pass')
class Pass(Stmt):
    @staticmethod
    def build(p):
        """
        pass_stmt ::=  "pass"
        """
        p.keyword('pass')
        return Pass()

@simple('del')
class Del(Stmt):
    def __init__(self,targets):
        super().__init__()
        self.targets = targets
    @staticmethod
    def build(p):
        """
        del_stmt ::=  "del" target_list
        """
        p.keyword('del')
        targets = p.target_list()
        return Del(targets)

@simple('return')
class Return(Stmt):
    def __init__(self,expr):
        super().__init__()
        self.expr = expr
    @staticmethod
    def build(p):
        """
        return_stmt ::=  "return" [expression_list]
        If an expression list is present, it is evaluated, else None is substituted.
        """
        p.keyword('return')
        expr = p.or_(NamedConstant(None)).expression_list()
        return Return(expr)

@simple('yield')
class Yield(Stmt):
    def __init__(self,yield_expr):
        super().__init__()
        self.yield_expr = yield_expr
    @staticmethod
    def build(p):
        """
        yield_stmt ::=  yield_expression
        """
        yield_expr = p.build_node(YieldExpression)
        return Yield(yield_expr)

@simple('raise')
class Raise(Stmt):
    def __init__(self,exc,from_exc):
        super().__init__()
        self.exc = exc # expression | None
        self.from_exc = from_exc # expression | None
    @staticmethod
    def build(p):
        """
        raise_stmt ::=  "raise" [expression ["from" expression]]
        """
        p.keyword('raise')
        exc = p.or_none.expression()
        if exc is None:
            return Raise(None,None) # deals with nested bnf brackets
        def from_exc_fn():
            p.keyword('from')
            return p.expression()
        from_exc = p.or_none(from_exc_fn)

        return Raise(exc,from_exc)

@simple('break')
class Break(Stmt):
    @staticmethod
    def build(p):
        """
        break_stmt ::=  "break"
        """
        p.keyword('break')
        return Break()

@simple('continue')
class Continue(Stmt):
    @staticmethod
    def build(p):
        """
        continue_stmt ::=  "continue"
        """
        p.keyword('continue')
        return Continue()

Alias = namedtuple('Alias','val alias')

class ModImport(InitStmt):
    def __init__(self,mods):
        self.mods = mods # Alias(str,str) list
class FromImport(InitStmt):
    def __init__(self,mod):
        self.mod = mod # str | (str list) ; it's a list if relative module
        self.imports = imports # Alias(str,str) list | '*'

@simple('import','from')
class Import(StmtGenerator):
    def __init__(self,):
        super().__init__()
        self.from_mod = 
        self.
    @staticmethod
    def build(p):
        """
        import_stmt     ::=  "import" module ["as" identifier] ("," module ["as" identifier])*
                     | "from" relative_module "import" identifier ["as" identifier]
                     ("," identifier ["as" identifier])*
                     | "from" relative_module "import" "(" identifier ["as" identifier]
                     ("," identifier ["as" identifier])* [","] ")"
                     | "from" module "import" "*"
        module          ::=  (identifier ".")* identifier
        relative_module ::=  "."* module | "."+

        """
        def module(): # returns (str list)
            """ module ::=  (identifier ".")* identifier """
            return p.dotted_name()
        def relative_module(): # returns tuple(ndots,(str|None))
            """ relative_module ::=  "."* module | "."+ """
            def dot():
                p.token('.')
            with p.no_whitespace():
                ndots = len(p.list(dot))
            mod = p.or_none(module)
            if mod is None and ndots == 0:
                raise SyntaxError
            return ndots,mod
        def as_id():
            p.keyword('as')
            return p.identifier()
        def mod_as_id(): # returns tuple((str list),str)
            """ module ["as" identifier] """
            mod = module()
            as_ident = p.or_none(as_id)
            return Alias(mod,as_ident)
        def id_as_id(): # returns tuple((str list),str)
            """ identifier ["as" identifier] """
            id1 = p.identifier()
            id2 = p.or_none(as_id)
            return Alias(id1,id2)

        def import_stmt1(): # returns (import_as list)
            """ "import" module ["as" identifier] ("," module ["as" identifier])* """
            p.keyword('import')
            return ModImport(p.comma_list(mod_as_id,allow_trailing_comma=False,nonempty=True))
        def import_stmt2(): # returns tuple(relative_module, import_as list)
            """ "from" relative_module "import" identifier ["as" identifier] ("," identifier ["as" identifier])*
            note theres no module() call this time it's identifier() instead
            """
            p.keyword('from')
            rel_mod = relative_module()
            p.keyword('import')
            imports = p.comma_list(id_as_id,allow_trailing_comma=False,nonempty=True)
            return FromImport(rel_mod,imports)
        def import_stmt3():
            """ "from" relative_module "import" "(" identifier ["as" identifier] ("," identifier ["as" identifier])* [","] ")"
            note trailing comma is allowed this time
            """
            p.keyword('from')
            rel_mod = relative_module()
            p.keyword('import')
            with p.parens():
                imports = p.comma_list(id_as_id,nonempty=True)
            return FromImport(rel_mod,imports)
        def import_stmt4():
            """ "from" module "import" "*" """
            p.keyword('from')
            mod = module()
            p.keyword('import')
            p.token('*')
            FromImport(mod,'*')
        return p.exclusive_xor(import_stmt1,import_stmt2,import_stmt3,import_stmt4)

@simple('from')
class Future(Stmt):
    def __init__(self,expr):
        super().__init__()
        self.imports = imports # Alias list
    @staticmethod
    def build(p):
        """
        future_stmt ::=  "from" "__future__" "import" feature ["as" identifier]
                 ("," feature ["as" identifier])*
                 | "from" "__future__" "import" "(" feature ["as" identifier]
                 ("," feature ["as" identifier])* [","] ")"
        feature     ::=  identifier
        """
        p.keyword('from')
        if not p.identifier() == '__future__':
            raise SyntaxError
        p.keyword('import')
        def id_as_id():
            id1 = p.identifier()
            def as_id():
                p.keyword('as')
                return p.identifier()
            id2 = p.or_none(as_id)
            return Alias(id1,id2)
        def opt1():
            return p.comma_list(id_as_id,nonempty=True,allow_trailing_comma=False)
        def opt2():
            with p.parens():
                return p.comma_list(id_as_id,nonempty=True)
        return Future(p.logical_xor(opt1,opt2))

@simple('global')
class Global(Stmt):
    def __init__(self,expr):
        super().__init__()
        self.names = names
    @staticmethod
    def build(p):
        """
        global_stmt ::=  "global" identifier ("," identifier)*
        """
        p.keyword('global')
        return Global(p.comma_list(p.identifier,nonempty=True,allow_trailing_comma=False)

@simple('nonlocal')
class Nonlocal(Stmt):
    def __init__(self,expr):
        super().__init__()
        self.names = names
    @staticmethod
    def build(p):
        """
        nonlocal_stmt ::=  "nonlocal" identifier ("," identifier)*
        """
        p.keyword('nonlocal')
        return Nonlocal(p.comma_list(p.identifier,nonempty=True,allow_trailing_comma=False)


# TODO would be ultra nice to say identifier() vs identifer?() where the latter sends an extra argument which is just like fail=BOOL!

## EXPRS

class Var(Expr):
    """
    identifier   ::=  xid_start xid_continue*
    id_start     ::=  <all characters in general categories Lu, Ll, Lt, Lm, Lo, Nl, the underscore, and characters with the Other_ID_Start property>
    id_continue  ::=  <all characters in id_start, plus characters in the categories Mn, Mc, Nd, Pc and others with the Other_ID_Continue property>
    xid_start    ::=  <all characters in id_start whose NFKC normalization is in "id_start xid_continue*">
    xid_continue ::=  <all characters in id_continue whose NFKC normalization is in "id_continue*">
    """
    def __init__(self,name):
        super().__init__()
        self.name = name
    @staticmethod
    def build(p):
        name = p.identifier()
        return Var(name)

# wraps a Node to mark it as starred
class Starred(InitExpr):
    def __init__(self,node):
        super().__init__()
        self.node = node

class DoubleStarred(InitExpr):
    def __init__(self,node):
        super().__init__()
        self.node = node
# key:val or key=val

## SYN NODES
class KVPair(AuxNode):
    def __init__(self,key,val):
        super().__init__()
        self.key = key
        self.val = val

class CompIf(AuxNode):
    def __init__(self,cond,comp_iter):
        super().__init__()
        self.cond = cond
        self.comp_iter = comp_iter

class CompFor(AuxNode):
    def __init__(self,targets,iter,comp_iter):
        super().__init__()
        self.targets = targets
        self.iter = iter
        self.comp_iter = comp_iter

class ListDisplay(Expr):
    """list_display ::=  "[" [starred_list | comprehension] "]" """
    def __init__(self,body):
        super().__init__()
        self.body = body
    @staticmethod
    def build(p):
        p = p.brackets()

        def comprehension():
            comp = p.comprehension()
            p.assert_empty()
            return ListDisplay(comp)
        def starred_list_noexpr():
            return p.starred_list(as_expr=False)

        return p.logical_xor(comprehension,starred_list_noexpr)

class SetDisplay(Expr):
    """set_display ::=  "[" [starred_list | comprehension] "]" """
    def __init__(self,body):
        super().__init__()
        self.body = body
    @staticmethod
    def build(p):

        def comprehension():
            comp = p.comprehension()
            p.assert_empty()
            return SetDisplay(comp)
        def starred_list_noexpr():
            return p.starred_list(as_expr=False)

        with p.braces():
            return p.logical_xor(comprehension,starred_list_noexpr)

class DictDisplay(Expr):
    """
    dict_display       ::=  "{" [key_datum_list | dict_comprehension] "}"
    key_datum_list     ::=  key_datum ("," key_datum)* [","]
    key_datum          ::=  expression ":" expression | "**" or_expr
    dict_comprehension ::=  expression ":" expression comp_for
    """
    def __init__(self,body):
        super().__init__()
        self.body = body # DictComprehension | ((DoubleStarred(or_expr)|KVPair) list)
    @staticmethod
    def build(p):
        def comprehension():
            comp = p.build_node(DictComprehension)
            return DictDisplay(comp)
        def kv_pair():
            key = p.expression()
            p.token(':')
            val = p.expression()
            kv = KVPair(key,val)
            return kv
        def dict_expansion():
            p.token('**')
            e = DoubleStarred(p.or_expr())
            return e
        def key_datum_list(): # empty list is allowed
            return DictDisplay(p.comma_list(kv_pair,dict_expansion))

        with p.braces():
            return p.logical_xor(comprehension,key_datum_list)



"""
The iterable expression in the leftmost for clause is evaluated directly in the enclosing scope and then passed as an argument to the implictly nested scope. Subsequent for clauses and any filter condition in the leftmost for clause cannot be evaluated in the enclosing scope as they may depend on the values obtained from the leftmost iterable. For example: [x*y for x in range(10) for y in range(x, x+10)].
"""

"""
A Comprehension does not need to be left_recursive because the .expression() that starts it can't start with Comprehension anyways.
"""

class Comprehension(Expr):
    def __init__(self,expr,comp_for):
        super().__init__()
        self.expr = expr # can be an expression or a KVPair (for dict comprehension)
        self.comp_for = comp_for
    @staticmethod
    def build(p):
        expr = self.expression()
        comp_for = p.comp_for()
        return Comprehension(expr,comp_for)

class DictComprehension(Expr):
    def __init__(self,kv_pair,comp_for):
        super().__init__()
        self.kv_pair = kv_pair # can be an expression or a KVPair (for dict comprehension)
        self.comp_for = comp_for
    @staticmethod
    def build(p):
        key = p.expression()
        p.token(':')
        val = p.expression()
        kv = KVPair(key,val)
        return DictComprehension(kv_pair,comp_for)



# a,b,c
# "a," in Tup we call expr(left=Tup) which finds "b" then sees Tup and rates precedence as EQUAL which causes it to break and return "b" to us


# TODO add "leftnodeclass=" everywhere


"""
if you see a "," and are trying to expand via left recursion and ur not a

"""

class Tuple(InitExpr):
    def init(self,vals):
        super().__init__()
        self.vals = vals # [] for empty tuple

class Ellipsis(Expr):
    def __init__(self):
        super().__init__()
    @staticmethod
    def build(p):
        p.token('...'):
        return Ellipsis()

@left_recursive('or_test')
class Ternary(Expr):
    """
    conditional_expression ::=  or_test ["if" or_test "else" expression]
    """
    def __init__(self,cond,if_val,else_val):
        super().__init__()
        self.cond = cond # or_test
        self.if_val = if_val # or_test
        self.else_val = else_val # expression
    @staticmethod
    def build(p,leftnode):
        p.keyword('if')
        cond = p.or_test()
        p.keyword('else')
        else_val = p.expression()
        return Ternary(cond,leftnode,else_val)

class Lambda(Expr):
    """
    lambda_expr        ::=  "lambda" [parameter_list] ":" expression
    lambda_expr_nocond ::=  "lambda" [parameter_list] ":" expression_nocond
    """
    def __init__(self,args,body):
        super().__init__()
        self.args = args
        self.body = body
    @staticmethod
    def build(p,nocond=False): # the trunk version is the default nocond=False
        p.keyword('lambda')
        args = p.parameter_list(no_annotations=True)
        if nocond:
            body = p.expression_nocond()
        else:
            body = p.expresssion()
        return Lambda(args,body)

# There's YieldExpression YieldAtom (also an Expr) and YieldStmt
class YieldExpression(Expr):
    """
    yield_expression ::=  "yield" [expression_list | "from" expression]
    """
    def __init__(self,val,has_from):
        super().__init__()
        self.val = val
        self.has_from = has_from # bool
    @staticmethod
    def build(p):
        p.keyword('yield')
        has_from = p.or_false.keyword('from')
        val = e.expression() if has_from else p.expression_list()
        return YieldExpression(val,has_from)

class YieldAtom(Expr):
    """
    yield_atom       ::=  "(" yield_expression ")"
    """
    def __init__(self,expr):
        super().__init__()
        self.expr = expr
    @staticmethod
    def build(p):
        with p.parens() as p:
            expr = p.build_node(YieldExpression)
        return expr



class GeneratorExpression(Expr):
    """
    generator_expression ::=  "(" expression comp_for ")"

    A generator expression yields a new generator object. Its syntax is the same as for comprehensions, except that it is enclosed in parentheses instead of brackets or curly braces.

    """
    def __init__(self,comp):
        self.comp = comp
    @staticmethod
    def build(p):
        p = p.parens()
        comp = p.comprehension()
        p.assert_empty()
        return GeneratorExpression(comp)

class ParenthForm(ExprGenerator):
    """
    parenth_form ::=  "(" [starred_expression] ")"

    A parenthesized expression list yields whatever that expression list yields: if the list contains at least one comma, it yields a tuple; otherwise, it yields the single expression that makes up the expression list.

    An empty pair of parentheses yields an empty tuple object.
    """
    @staticmethod
    def build(p):
        with p.parens() as p:
            body = p.starred_expression()
            if len(body) == 1 and p.elems[-1].typ != COMMA:
                return body[0] # eval to the single expression in the list

        return Tuple(body) # includes len==0 empty tuple case



class Lit(Expr):
    """
    literal ::=  stringliteral | bytesliteral | integer | floatnumber | imagnumber

    stringliteral   ::=  [stringprefix](shortstring | longstring)
    stringprefix    ::=  "r" | "u" | "R" | "U" | "f" | "F"
                         | "fr" | "Fr" | "fR" | "FR" | "rf" | "rF" | "Rf" | "RF"
    shortstring     ::=  "'" shortstringitem* "'" | '"' shortstringitem* '"'
    longstring      ::=  "'''" longstringitem* "'''" | '"""' longstringitem* '"""'
    shortstringitem ::=  shortstringchar | stringescapeseq
    longstringitem  ::=  longstringchar | stringescapeseq
    shortstringchar ::=  <any source character except "\" or newline or the quote>
    longstringchar  ::=  <any source character except "\">
    stringescapeseq ::=  "\" <any source character>

    bytesliteral   ::=  bytesprefix(shortbytes | longbytes)
    bytesprefix    ::=  "b" | "B" | "br" | "Br" | "bR" | "BR" | "rb" | "rB" | "Rb" | "RB"
    shortbytes     ::=  "'" shortbytesitem* "'" | '"' shortbytesitem* '"'
    longbytes      ::=  "'''" longbytesitem* "'''" | '"""' longbytesitem* '"""'
    shortbytesitem ::=  shortbyteschar | bytesescapeseq
    longbytesitem  ::=  longbyteschar | bytesescapeseq
    shortbyteschar ::=  <any ASCII character except "\" or newline or the quote>
    longbyteschar  ::=  <any ASCII character except "\">
    bytesescapeseq ::=  "\" <any ASCII character>

    integer      ::=  decinteger | bininteger | octinteger | hexinteger
    decinteger   ::=  nonzerodigit (["_"] digit)* | "0"+ (["_"] "0")*
    bininteger   ::=  "0" ("b" | "B") (["_"] bindigit)+
    octinteger   ::=  "0" ("o" | "O") (["_"] octdigit)+
    hexinteger   ::=  "0" ("x" | "X") (["_"] hexdigit)+
    nonzerodigit ::=  "1"..."9"
    digit        ::=  "0"..."9"
    bindigit     ::=  "0" | "1"
    octdigit     ::=  "0"..."7"
    hexdigit     ::=  digit | "a"..."f" | "A"..."F"

    floatnumber   ::=  pointfloat | exponentfloat
    pointfloat    ::=  [digitpart] fraction | digitpart "."
    exponentfloat ::=  (digitpart | pointfloat) exponent
    digitpart     ::=  digit (["_"] digit)*
    fraction      ::=  "." digitpart
    exponent      ::=  ("e" | "E") ["+" | "-"] digitpart

    imagnumber ::=  (floatnumber | digitpart) ("j" | "J")

    """
    def __init__(self,val):
        super().__init__()
        self.val = val
    @staticmethod
    def identify(p):
        if p.or_false.token(INTEGER):
            return Int
        if p.or_false.token(FLOAT):
            return Float
        if p.or_false.token(COMPLEX):
            return Complex
        if p.or_false.token(BYTES):
            return Bytes
        if p.or_false.keyword('True'):
            return NamedConstant
        if p.or_false.keyword('False'):
            return NamedConstant
        if p.or_false.keyword('None'):
            return NamedConstant
        if p.or_false.quote():
            return Str
        return None
class Num(Lit):pass
class Int(Num):
    @staticmethod
    def build(p):
        p.token(INTEGER)
        return Int(int(p.curr.data))
class Float(Num):
    @staticmethod
    def build(p):
        p.token(FLOAT)
        return Float(float(p.curr.data))
class Complex(Num):
    @staticmethod
    def build(p):
        p.token(COMPLEX):
        return Complex(complex(p.curr.data))
class Bytes(Lit):
    @staticmethod
    def build(p):
        p.token(BYTES):
        return Bytes(bytes(p.curr.data))
class NamedConstant(Lit):
    @staticmethod
    def build(p):
        if p.or_false.keyword('True'):
            return NamedConstant(True)
        if p.or_false.keyword('False'):
            return NamedConstant(False)
        if p.or_false.keyword('None'):
            return NamedConstant(None)
        raise SyntaxError
class Str(Lit):
    @staticmethod
    def build(p):
        p.quote():
        return Str(p.curr.data)

class UnopL(Expr):
    def __init__(self,val):
        super().__init__()
        self.val = val # no need to include `op` bc thats captured by subclass
    @staticmethod
    def identify(p):
        if p.or_false.token(UNOPSL):
            return unopl_subclass[p.curr.typ]
        raise SyntaxError
    @classmethod
    def gen_build(cls,type):
    def build(p):
        op = elems[0].typ
        cls = unopl_subclass(op)
        val,elems = expr(elems[1:],leftnodeclass=cls)
        return cls(val),elems
        raise NotImplementedError(f"Unrecongized left unary operator: {op}")
class UAdd(UnopL):pass
class USub(UnopL):pass
class Invert(UnopL):pass

class Not(UnopL):pass

@left_recursive('primary')
class AttributeRef(Expr):
    def __init__(self,val,attr):
        super().__init__()
        self.val = val
        self.attr = attr
    @staticmethod
    def build(p,leftnode):
        p.token('.')
        attr = p.identifer()
        return AttributeRef(leftnode,attr)

@left_recursive('primary')
class Subscription(InitExpr):
    def __init__(self,val,expr):
        super().__init__()
        self.val = val
        self.expr = expr

@left_recursive('primary')
class Slicing(InitExpr):
    def __init__(self,val,items):
        super().__init__()
        self.val = val
        self.items = items

@left_recursive('primary')
class SubscriptOrSlicing(ExprGenerator):
    """
    Grouping these makes sense because of the "there is ambiguity in the formal syntax" note below.

    subscript = "[" expression_list "]"
    slicing      ::=  primary "[" slice_list "]"

    slice_list   ::=  slice_item ("," slice_item)* [","]
    slice_item   ::=  expression | proper_slice
    proper_slice ::=  [lower_bound] ":" [upper_bound] [ ":" [stride] ]
    lower_bound  ::=  expression
    upper_bound  ::=  expression
    stride       ::=  expression

    There is ambiguity in the formal syntax here: anything that looks like an expression list also looks like a slice list, so any subscription can be interpreted as a slicing. Rather than further complicating the syntax, this is disambiguated by defining that in this case the interpretation as a subscription takes priority over the interpretation as a slicing (this is the case if the slice list contains no proper slice).
    """
    @staticmethod
    def build(p,leftnode):
        with p.brackets() as p:
            def proper_slice():
                # this fn must be defined after start of `with` block for proper `p` closure
                lower = p.or_none.expression()
                p.token(":")
                upper = p.or_none.expression()
                if p.or_false.token(":"):
                    stride = p.or_none.expression()
                else:
                    stride = None
                return Slice(lower,upper,stride)
            def slice_item():
                return p.logical_xor(proper_slice,p.expression)
            items = p.comma_list(slice_item,nonempty=True)

            if all([type(item) != Slice for item in exprs]):
                # a[1,2] == a[(1,2)]
                items = items[0] if (len(items) == 1 and p.prev.data != ',') else Tuple(items)
                return Subscript(leftnode,items)
        return Slicing(leftnode,items)

# A syntax node version of the python slice() object. Same kinda info but not an acual Expr. Used by Slicing and such.
class Slice(AuxNode):
    def __init__(self,start=None,stop=None,step=None):
        super().__init__()
        self.start = start
        self.stop = stop
        self.step = step

@left_recursive('primary')
class Call(Expr):
    def __init__(self,func,args):
        super().__init__()
        self.func = func
        self.args = args # Arguments | Comprehension
    @staticmethod
    def build(p,leftnode):
        with p.parens() as p:
            args = p.exclusive_xor(p.argument_list,p.comprehension)
        return AttributeRef(leftnode,args)

class AwaitExpr(Expr):
    def __init__(self,expr):
        super().__init__()
        self.expr = expr # 'primary'
    @staticmethod
    def build(p,leftnode):
        expr = p.primary()
        return AwaitExpr(expr)

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


@left_recursive(None) # forces all children to do left_recursive(actual_type)
class Binop(AbstractExpr):
    def __init__(self,lhs,rhs):
        super().__init__()
        self.lhs = lhs # no need to include `op` bc thats captured by subclass
        self.rhs = rhs
    @staticmethod
    def identify(p):
        if p.or_false.token(BINOPS):
            return binop_subclass[p.curr.typ]
        raise SyntaxError
    @classmethod
    def gen_build(cls,rhstype):
        def build(p,leftnode):
            rightnode = p.trunk_expr(rhstype)
            return cls(leftnode,rightnode)
        return build

# class decorator to easily define the types on the left and right side of the binop
def binop(left_type,right_type):
    """
    @binop('a_expr','m_expr')
    class Add(Binop): pass
    ...is equivalent to...
    @left_recursive('a_expr')
    @gen_build('m_expr')
    class Add(Binop):pass
    """
    def class_decorator(cls):
        cls = left_recursive(left_type)(cls)
        cls = gen_build(right_type)(cls)
        return cls

# power
@binop('await_or_primary','u_expr')
class Exp(Binop): pass

# m_expr
@binop('m_expr','u_expr')
class Mul(Binop): pass
@binop('m_expr','m_expr')
class MatMul(Binop): pass
@binop('m_expr','u_expr')
class FloorDiv(Binop): pass
@binop('m_expr','u_expr')
class Div(Binop): pass
@binop('m_expr','u_expr')
class Mod(Binop): pass

# a_expr
@binop('a_expr','m_expr')
class Add(Binop):pass
@binop('a_expr','m_expr')
class Sub(Binop):pass

# shift_expr
@binop('shift_expr','a_expr')
class ShiftL(Binop): pass
@binop('shift_expr','a_expr')
class ShiftR(Binop): pass

# and_expr
@binop('and_expr','shift_expr')
class BitAnd(Binop): pass

# xor_expr
@binop('xor_expr','and_expr')
class BitXor(Binop): pass

# or_expr
@binop('or_expr','xor_expr')
class BitOr(Binop): pass

# and_test
@binop('and_test','not_test')
class And(Binop):pass

# or_test
@binop('or_test','and_test')
class Or(Binop):pass

@left_recursive('or_expr')
class Compare(Expr):
    def __init__(self,ops,vals):
        super().__init__()
        self.ops = ops
        self.vals = vals
    @staticmethod
    def build(p,leftnode):
        vals = [leftnode]
        ops = []
        def extend():
            op = p.curr.typ
            p.token(CMPOPS)
            val = p.or_expr()
            ops.append(op)
            vals.append(val)
            return None
        p.list(extend,nonempty=False)
        return Compare(ops,vals)


# all these might be unnecessary
#class CompareElem(AuxNode): pass
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
_str_of_token = ['INTEGER','PERIOD','COMMA','COLON','SEMICOLON','EXCLAM','PYPAREN','SH_LBRACE','SH_LPAREN','SH','LPAREN','RPAREN','LBRACE','RBRACE','LBRACKET','RBRACKET','ADD','SUB','MUL','FLOORDIV','DIV','EXP','MOD','SHIFTRIGHT','SHIFTLEFT','BITAND','BITOR','BITXOR','INVERT','NOT','AND','OR','LEQ','LT','GEQ','GT','EQ','NEQ','IS','ISNOT','IN','NOTIN','ASN','ESCQUOTE2','ESCQUOTE1','QUOTE2','QUOTE1','HASH','PIPE','ID','UNKNOWN','SOL','EOL','NEWLINE','KEYWORD','ELLIPSIS', 'AT']
def str_of_token(t):
    return _str_of_token[t-OFFSET]
# order doesn't matter in this list, as long as it's the same order as in the preceding list
[INTEGER, PERIOD, COMMA, COLON, SEMICOLON, EXCLAM, PYPAREN, SH_LBRACE, SH_LPAREN, SH, LPAREN, RPAREN, LBRACE, RBRACE, LBRACKET, RBRACKET, ADD, SUB, MUL, FLOORDIV, DIV, EXP, MOD, SHIFTRIGHT, SHIFTLEFT, BITAND, BITOR, BITXOR, INVERT, NOT, AND, OR, LEQ, LT, GEQ, GT, EQ, NEQ, IS, ISNOT, IN, NOTIN, ASN, ESCQUOTE2, ESCQUOTE1, QUOTE2, QUOTE1, HASH, PIPE, ID, UNKNOWN, SOL, EOL, NEWLINE, KEYWORD, ELLIPSIS, AT] = [CONSTANT(_str_of_token[i]) for i in range(len(_str_of_token))] # we start at 100 so that we can still safely manually define constants in the frequently used -10..10 range of numbers with no fear of interference or overlap.

CMPOPS = [LEQ,LT,GEQ,GT,EQ,NEQ,IS,ISNOT,IN,NOTIN]
BINOPS = [ADD, SUB, MUL, DIV, FLOORDIV, MOD, EXP, SHIFTRIGHT, SHIFTLEFT, BITAND, BITOR, BITXOR, AT, AND, OR]
UNOPSL  = [ADD, SUB, NOT, INVERT] # yes, ADD and SUB can be unops
#UNOPSR  = [] # right side unary operators. There aren't any... yet! Well actually () and [] and . are UNOPSR really.


COMMON_ALLOWED_CHILDREN = set([QUOTE1,QUOTE2,LBRACKET,LBRACE,LPAREN,SH_LBRACE,SH, SH_LPAREN, HASH])
COMMON_ERROR_ON = set([RPAREN,RBRACE,RBRACKET]) # note that closers are checked first in Atomize so it's safe to include RPAREN in the .error_on field of AParen for example. Hence the COMMON_ERROR_ON can be widely used


# compile the regex and also eat any whitespace that follows it
def regex_compile(regex):
    return re.compile(fr'({regex})\s*'regex+'\s*')

stringprefix = r'"fr"|"Fr"|"fR"|"FR"|"rf"|"rF"|"Rf"|"RF"|"r"|"u"|"R"|"U"|"f"|"F"'
shortstringstart = rf'\''
#shortstringitem = rf'{shortstringchar} | {stringescapeseq}'
# not sure if \n and \r are both supposed to be there
#shortstringchar = r'[^\\\n\r\']'
#((\'(([^\\\n\r\'])|(\\.))*|)|(|))'),

# order is important in this list!
regex_of_token = {
    INTEGER   : regex_compile(r'\d+'),
    ELLIPSIS  : regex_compile(r'...'), # BEFORE PERIOD
    PERIOD    : regex_compile(r'\.'),
    COMMA     : regex_compile(r','),
    COLON     : regex_compile(r':'),
    SEMICOLON : regex_compile(r';'),
    EXCLAM    : regex_compile(r'!'),

    # GROUPINGS
    PYPAREN   : regex_compile(r'py\('), # BEFORE ID
    #SH_LBRACE : regex_compile(r'sh{'), # BEFORE ID
    SH_LPAREN : regex_compile(r'sh\('), # BEFORE ID
    SH        : regex_compile(r'sh\b'), # BEFORE ID, AFTER SH_*
    LPAREN    : regex_compile(r'\('),
    RPAREN    : regex_compile(r'\)'),
    LBRACE    : regex_compile(r'{'),
    RBRACE    : regex_compile(r'}'),
    LBRACKET  : regex_compile(r'\['),
    RBRACKET  : regex_compile(r'\]'),
    ARROW     : regex_compile(r'->'), # BEFORE SUB

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
    AT        : regex_compile(r'@'),


    # UNOPS
    INVERT    : regex_compile(r'~'),
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
    ISNOT     : regex_compile(r'is\s+not'), # BEFORE ID
    IS        : regex_compile(r'is'), # BEFORE ID
    NOTIN     : regex_compile(r'not\s+in'), # BEFORE ID
    NOT       : regex_compile(r'not'),
    IN        : regex_compile(r'in'), # BEFORE ID

    ASN       : regex_compile(r'='),
    ESCQUOTE2 : regex_compile(r'\\\"'),
    ESCQUOTE1 : regex_compile(r'\\\''),
    HEREDOC    : regex_compile(r'(?:""")|(?:\'\'\')'),
    QUOTE    : regex_compile(r'"|\''),
    HASH      : regex_compile(r'#'),
    PIPE      : regex_compile(r'\|'),
    ID        : regex_compile(r'[a-zA-z_]\w*'), # must go after keywords like sh
    UNKNOWN   : regex_compile(r'.'),
    SOL : None, # should never be matched against since 'UNKOWN' is a catch-all
    EOL : None, # should never be matched against since 'UNKOWN' is a catch-all
    NEWLINE : None, # newlines are manually inserted hence no need for a regex
    KEYWORD : None, # custom matching as a step after ID for efficiency
}
assert len(regex_of_token) == len(_str_of_token)
TOKENS = list(regex_of_token.keys())

#def closer_of_opener(opener_tok):
#    if isinstance(opener_tok,Tok):
#        opener_tok = opener_tok.typ
#
#    if opener_tok == LPAREN:
#        return Tok(RPAREN,'',')')
#    if opener_tok == PYPAREN:
#        return Tok(RPAREN,'',')')
#    if opener_tok == LBRACE:
#        return Tok(RBRACE,'','}')
#    if opener_tok == LBRACKET:
#        return Tok(RBRACKET,'',']')
#    if opener_tok == QUOTE1:
#        return Tok(QUOTE1,'','\'')
#    if opener_tok == QUOTE2:
#        return Tok(QUOTE2,'','"')
#    if opener_tok == SOL:
#        return Tok(EOL,'','')
#    raise NotImplementedError


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





LineData = namedtuple('LineData', 'linetkns, leading_whitespace, newline_tok, commenttkns')


leading_ws_regex = re.compile(r'(\s*)')
class Line:
    def __init__(self,input,loc):
        assert '\n' not in input
        self.loc = loc # Location with file and lineno
        self.leading_ws = leading_ws_regex.match(input).group()
        remaining = input[len(self.leading_ws):]
        self.tokens = []
        self.comment = []
        curr_list = self.tokens

        while remaining != '':
            for TYP,regex in regex_of_token.items():
                charno = len(line) - len(remaining)
                match = regex.match(remaining)
                if match is None:
                    continue # match failed
                # successful match
                remaining = remaining[match.end():]
                # comments
                if TYP is HASH:
                    curr_list = self.comment # switch to adding to list of comment tokens
                    continue
                # keywords
                if TYP is ID:
                    if match.group() in keywords:
                        # for efficiency rather than having every keyword in the regex
                        TYP = KEYWORD
                grps = match.groups()
                curr_list.append(Tok(
                        TYP, # typ
                        grps[0] if grps != [] else None, # data
                        match.group(), # verbatim
                        LocationRange(self.loc
                        Loc(Pos(lineno,charno),Pos(lineno,len(line)-len(remaining))) # lineno,charno TODO watch out for how \t is handled!
                    ))
                list_to_extend[-1].finish()
                break #break unless you 'continue'd before
            else: # nobreak
                raise ValueError(f"Somehow failed to match on any tokens, should be impossible because of the UNKNOWN token")
        self.empty = (len(self.tokens) == 0)

class Location:
    def __init__(self,*,file=None,lines=None,chars=None):
        assert isinstance(lines,(list,tuple))
        assert isinstance(chars,(list,tuple))
        assert len(chars) == 2
        assert isinstance(file,str)
        self.file = file
        self.lines = lines
        self.chars = chars
    @staticmethod
    def join(*locs):
        """
        Join contiguous Locations
        """
        file = locs[0].file
        lines = locs[0].lines
        chars = locs[0].chars
        for loc in locs[1:]:
            assert loc.file = file
            assert loc.lines[-1] == lines[-1] + 1
            assert loc.chars[-1] == chars[-1] + 1
        return Location(file=file,lines=lines,chars=chars)


def tokenize_line(line):
    """
    Takes a pre-stripped line as input, returns a list of Tok
    """
    assert line == line.strip()
    assert '\n' not in line
    remaining = line
    while remaining != '':
        for TOKEN,regex in regex_of_token.items():
            charno = len(line) - len(remaining)
            match = regex.match(remaining)
            if match is None:
                continue # match failed
            # successful match
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




