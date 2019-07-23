

## use this instead. It sticks with the AColonList invariant of python, and it's very minimal, and it makes a lot of sense
x =:
    x + 2


## note everything with `compiled` is just initial ideas, same with abstract data types, so think abt them more before concluding anything!
def gcd(a,b):
    return a+b

gcd(1.3,2.4) # generic with any types

gcd_compiled = compiled gcd(int,int) # note that this is NOT dynamic. Its a compile time error if it can't figure out which `gcd` you're talking about etc
gcd_compiled(10,20) # FAST (c-linked function)
gcd_compiled(1.2,39) # ERROR


# makes it turn every call with a new set of types into a `compiled` statement
@always_compiled
def gcd(a,b):
    return a+b




## constant()
# basically does what my CONSTANT does

## box():
# box() type that just holds stuff easily, sortof like a combined list/dict 
# that also has some clever stuff where it is aware of the names of things in
# code. This can actually be done at desugar time bc we have the varnames at that time!
b = box()
b.a = 3
b.b = None
assert b.a == 3
assert b.b is None
b = box(1,2,3)
assert b[1] == 2
a=1
test=2
b = box(a,test)
assert b[0] == a
assert b.a == a
t = 3
t2 = 4
b.add(t,t2)
assert b.t == t
assert b[3] == t
assert b['t'] == t

# MATCH WITH
## SEMISTABILIZED

match type(x) with:
    | int ->
        match x%2 == 0 with:
            | True -> print("even int")
            | False-> print("odd int")
    | str ->
        print("string")
        print("multiline is fine!")
    | _ ->
        print("here's a catch-all")


# ARGMATCH
## SEMISTABILIZED

argmatch cls in isinstance(x,cls):
    | Foo ->
        print("This prints if bool(isinstance(x,Foo)) is True")
    | Bar ->
        print("x is a Bar")


## desugared
fn = lambda cls: isinstance(x,cls)
if fn(Foo):
    print("This prints if bool(isinstance(x,Foo)) is True")
elif fn(Bar):
    print("x is a Bar")



argmatch x in x.is_digit():
    | foo() -> x + "goodbye" # you can use `x` in the body!!

argmatch x when x.is_digit():
    | foo() -> x + "goodbye"


y ::=
    match x%2 with:
        | 0 -> x
        | 1 -> x+1


# can be implemented easily by desugaring to an anonymous function that is executed and returns the result
y ::=
    z = [x*2 for x in range(n)] # we can use external variables like `n`
    b = len(z) # but variables we assign won't be visible outside (ie normal scoping rules)
    z



# ofc some abstract data types should be able to have arbitrary contents. Should they all?
# Id say no constraints 

## WIP

newtype Llvalue:
    |Raw(v)
    |Box(v)
    |Error

x = Raw(10)
assert x.v == 10
assert x['v'] == 10

unwrapped_val ::=
    match x with:
        | Raw(rawval) -> rawval # doesnt have to use the name `v`, can be positional
        | Box(boxedval) -> boxedval
        | Error -> None

newtype Foo:
    |Bar(a,b)

match x with:
    |Bar(b,a) -> "should this be positional (ie a==x.b and b==x.a) or name based? When you treat classes as newtypes it's name based and you need to use `as` to rename -- should we force that here too? or should we make an exception where if you use a name that's actually part of the newtype it will use that field. Well, if a newtype has 3 fields we should be able to grab only the first and thrid if we want so really it should be name based like classes i think. Oh i guess Baz(first,_,third) would do that"


unwrapped_val ::=
    match x with:
        | Raw(v) | Box(v) -> v
        | Error -> None
        | 3 -> 3

target "::=" [Stmt]

"match" x "with" ":"
    (("|" match_option)+ "->" [Stmt])+

match_option = newtype_destructure | newtype | Expr # a newtype destructure looks like a function call except the identifier is a newtype

newtype_destructure = ID ["("(ID",")*")"]


# one line unpacking in an expression
y = (match x with Raw(v) -> v)
# That's the same as x.v except it becomes None (or smtg else?) if the match fails

# Often we do want methods associated w a data type -- ie we do want a full `class` like Atom or Expr, but we also want to match on them easily


# renaming variable
match foo(x) as y with:
    | 10 -> print(y)

# destructuring based
def codegen_stmt(stmt):
    match stmt with:
        |If -> print("this was an if")
        |Try (body,excepts) ->
            print(body)
            print(excepts)
        |Try (body as b, excepts) -> # renaming, common for clarity
            print(b)
            print(excepts)
        |Try ->
            print(stmt.body)
            print(stmt.excepts)
        |Try (body) ->
            print(body)
            print(stmt.excepts)


class Bar:
    def __init__(self):
        self.field1 = 1
        self.field2 = 2

match foo(x).y as z with:
    | Bar -> print(z.field1)
    | Bar(field1) -> print(field1)
    | Bar(field1 as baz) -> print(baz)
    | Bar(field1 as baz, field2) -> print(field2)
    | Bar(field2) -> print(field2)
    #| Bar(x) -> pass # runtime error
    | < 3 -> print("it's less than 3! Any comparison function works, tho we default to `==`")
    | 3 -> "it equals 3"
    | 5+3 -> "any expression works"
    | foo(z) -> "true if z == foo(z). You can use your own variable in it"
    | is None -> print("it's None!")
    | None -> print("it's equal to None!")
    | < 3 -> "matches when bool(z<3) is true"
    | (f,g) -> print(f"it's a 2-tuple/list (or any destructurable `target`-like thing) with elements {f} and {g}")
    | (f,g,*) -> print(f"it's a tuple/list that starts with {f} and {g}")
    | (first,*,last) -> "this is interesting, not sure if it works in normal python"
    | (first,_,_,fourth) -> "this too"
    | (head,*tail) -> "head is the first element and tail is a list holding the rest"
    | head,*tail -> "no need for the parens, as with normal tuples"
    | _ -> print(f"didn't recognize {z}")


# Debate
match foo(a) as z with:
    | < 3 -> "matches when bool(z<3) is true" # not composable
    | z < 3 -> "matches when bool(z<3) is true" # ambiguous
    | z when z < 3 -> "matches when bool(z<3) is true"
    | z when z < 3 -> pass # 
    | if z < 3 -> pass # not great
    ? z < 3 -> pass # unnecessary and not composable
    ? < 3 -> pass #BAD


match foo(a) as result with:
    | Bar when isinstance(Bar,x)


# proposing this as a readable shorthand, with less composability and only works with comparison operators and can't chain with `and` or anything like that. For anything beyond a comparison operator that takes one other expression, you need to use `when`. Note that `when` has a huge amount of composability.


# shorthand only allowed with the basic comparisons LEQ,LT,GEQ,GT,EQ,NEQ,IS,ISNOT,IN,NOTIN
match x with:
    | 3 -> pass
    | > 3 -> pass
    | < 3 -> pass

# examples of composability of `when` with other syntax stuff
match x as y with:
    |Raw(v) when v is not None -> pass
    |Raw(v) when v > 10 -> pass
    |Raw(v) when x > 10 -> pass
    |Raw(v) when x.age == 10 -> pass # using `x` since its in the namespace. we're assuming Raw stuff has a .age field too that we've chosen not to unpack.
    |Raw(v) when foo(y) -> pass # using anything else in the namespace too
    |Raw(v) as z when z.age == 10 -> pass # combining `as` with it
    |Raw(v as inner) as outer when outer.age == 10 -> pass


## the possible things:
match x [as y] with:
    | a comparison operator e.g. `is` or `!=`
    | a target e.g. (x,y,*rest)
    | class or newtype destructure e.g. Bar or Bar(x) where Bar can be a newtype or a class
    | Expr (but not one of the above) in which case we make it `==` Expr
    | _




#`in` keyword
match x with:
    | 4 -> pass
    | in [2,3,4] -> pass
    | in keywords -> pass # some list `keywords`
    | is y -> pass # `is` too
    | > 2 -> pass # any Compare works actually!
    | 2 -> pass # defaults to `==`



# ofc this isn't terribly useful since we already have None in python
newtype Maybe:
    |Some of any
    |None


y ::=
    match x with:
        |Some(v) -> v
        |None -> print("fuck")

longvariablename ::=
    match x with:
        |Some(v) -> v
        |None -> print("fuck")







"""
The match statement. It contains statement within an expression so it's quite unusual.
    -makes use of ':' and indentation in place of '|' characters
    -RHS can only be an expression
    -considered ':' instead of '->' but it's pretty noisy and less clear
    -on a match the series of statement(s) in the block are evaluated and if the last one is an exprstmt then the total expression returns that value else it returns None
    -bars are required '|' because they make it so much more readable, esp when nesting etc
    - ';;' is needed to put multiple matches on a single line. Note a simple '|' could be interpreted as a bitwise OR etc.

"""

x = (match x+1 with:
    0 -> print("hi"); 1
    _ ->        # just as with semicolons you can't have a stmt on this line unless all stmts are on this line chained w ';'
         print("yo")
         3
    )


match x with:
    0 ->
        print("hi")
    1 ->
         print("hi") # this is fine, its ok if indent levels dont line up. Thats a clean code thing, we shouldn't be torturing people until their indent levels are perfect (plus sometimes you want a semicolon-separated thing for just one of the branches etc)

#"match e with:" is really shorthand for "match e==opt with opt as:"

# as usual newlines can be replaced with semicolons, however in this case double semicolons are needed to separate branches
x = (match x+1 with: 0 -> print("hi");; _ -> print("yo"))







