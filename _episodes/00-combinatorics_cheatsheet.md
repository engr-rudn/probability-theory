---
title: "Combinatorics CheatSheet"
questions:
- ""
objectives:
- ""
---

# Basics of generating functions

-   Introduction \[Wilf 1--3\]:

    -   how to define a sequence: exact formula, recurrent relation
        (Fibonacci), algorithm (the sequence of primes); there are
        uncomputable sequences (programs that do not stop)

    -   a new way: power series (members of the sequence as coefficients
        in the series)

    -   advantages: many advanced tools from analytical theory of
        functions

    -   very powerful: works on many sequences where nothing else is
        known to work

    -   allows to get asymptotic formulas and statistical properties

    -   powerful way to prove combinatorial identities

    -   "Konečne vidím, že je tá matalýza aj na niečo dobrá. Keby mi to
        bol niekto predtým povedal..."

-   Two examples \[Wilf 3--7\]:

    -   $$a_{n+1} = 2a_n + 1$$ for $$n\ge 0$$, $$a_0 = 0$$

    -   write few members, guess $$a_n = 2^n-1$$, provable by induction

    -   multiply by $$x^n$$, sum over all $$n$$, assign gf:
        $$\displaystyle \qquad{A(x)\over x}=2A(x)+{1\over 1-x}$$

    -   partial fraction expansion:
        $$\displaystyle \qquad A(x)={x\over (1-x)(1-2x)}={1\over 1-2x}-{1\over 1-x}$$

    -   the method stays basically the same for harder problems

    -   $$a_{n+1}=2a_n+n$$ for $$n\ge 0$$, $$a_0=1$$

    -   exact formula not obvious; no unqualified variables in the
        recurrence

    -   obstacle: $$\sum_{n\ge 0} nx^n = x/(1-x)^2$$; solution:
        differentiation

    -   concern: is differentiation allowed? discussed later, but in
        principle yes: in formal power series (as an algebraic ring) or
        via convergence (if we care about analytical properties)

    -   $$\displaystyle A(x) = {1-2x+2x^2\over (1-x)^2(1-2x)} = {A\over (1-x)^2} + {B\over 1-x} + {C\over 1-2x} = {-1\over (1-x)^2} + {2\over 1-2x}$$

    -   $$1/(1-x)^2$$ is just $$x/(1-x)^2$$ (see above) shifted by $$1$$

    -   $$a_n=2^{n+1}-n-1$$

-   The method \[Wilf 8\]:

    -   1\. make sure variables in the recurrence are qualified (e.g.
        range for $$n$$)

    -   2\. name and define the gf

    -   3\. multiply by $$x^n$$, sum over all $$n$$ in the range

    -   4\. express both sides in terms of the gf

    -   5\. solve the equation for gf

    -   6\. calculate coefficients of gf power series expansion

    -   useful notation: $$[x^n]f(x)$$; e.g.
        $$[x^n]e^x=1/n!\qquad [t^r]{1\over 1-3t}=3^r\qquad [v^m](1+v)^s={s\choose m}$$

-   Solve $$a_n=5a_{n-1}-6a_{n-2}$$ for $$n\ge 2$$, $$a_0 = 0$$, $$a_1=1$$.

-   Fibonacci \[Wilf 8--10\]:

    -   three-term recurrence: $$F_{n+1}=F_n+F_{n-1}$$ for $$n\ge 1$$,
        $$F_0=0$$, $$F_1=1$$.

    -   apply the method ($$r_\pm = (1\pm \sqrt 5)/2$$):
        $$\displaystyle F(x) = {x\over 1-x-x^2} = {x\over (1-xr_{+})(1-xr_{-})}={1\over r_{+}-r_{-}}\left({1\over 1-xr_{+}}-{1\over 1-xr_{-}}\right)$$

    -   $$F_n={1\over \sqrt 5}(r_{+}^n-r_{-}^n)$$

    -   the second term is $${} < 1$$ and goes to zero, so the first term
        $${1\over \sqrt 5}({1+\sqrt 5\over 2})^n$$ gives a good
        approximation

-   Find ogf for the following sequences (always $$n\ge 0$$) \[W1.1\]:

      ------- ------------------------------------------------- --
       \(a\)  $$a_n = n$$                                         
       \(b\)  $$a_n = \alpha n + \beta$$                          
       \(c\)  $$a_n = n^2$$                                       
       \(d\)  $$a_n = n^3$$                                       
       \(e\)  $$a_n = P(n)$$; $$P$$ is a polynomial of degree $$m$$   
       \(f\)  $$a_n = 3^n$$                                       
       \(g\)  $$a_n = 5\cdot 7^n-3\cdot 4^n$$                     
       \(h\)  $$a_n = (-1)^n$$                                    
      ------- ------------------------------------------------- --

-   Find the following coefficients \[W1.5\]:

      ------- -------------------------------------- --
       \(a\)  $$[x^n]\, e^{2x}$$                       
       \(b\)  $$[x^n/n!]\, e^{\alpha x}$$              
       \(c\)  $$[x^n/n!]\, \sin x$$                    
       \(d\)  $$[x^n]\, 1/(1-ax)(1-bx)$$ ($$a\neq b$$)   
       \(e\)  $$[x^n]\, (1+x^2)^m$$                    
      ------- -------------------------------------- --
 

-   Compute $$\square_n = \sum_{k=1}^n k^2$$.

    -   assign ogf to the sequence $$1^2, 2^2, \dots, n^2$$:
        $$f(x) = \sum_{k=1}^n{k^2x^k}$$

    -   $$(x{\rm D})^2 [(x^{n+1}-1)/(x-1)] = x {-2 n^2 x^{n + 1} + n^2 x^{n + 2} + n^2 x^n - 2 n x^{n + 1} + x^{n + 1} + 2 n x^n + x^n - x - 1)\over (x - 1)^3}$$

    -   note that
        $$\square_n = f(1) = \lim_{x\to 1} (xD)^2 [(x^{n+1}-1)/(x-1)]=n(n+1)(2n+1)/6$$

-   Find the sequence with gf $$1/(1-x)^3$$.

-   Find a linear recurrence going back two sequence members that has a
    solution that contains $$n\cdot 3^n$$ (possibly plus some linear
    combination of other exponential or polynomial factors).

-   Find explicit formulas for the following sequences \[W1.6, R2, R3,
    R7\]:

      ------- -------------------------------------------------------------------------------- --
       \(a\)  $$a_{n+1} = 3a_n+2$$ for $$n\ge 0$$; $$a_0=0$$                                         
       \(b\)  $$a_{n+1} = \alpha a_n + \beta$$ for $$n\ge 0$$; $$a_0=0$$                             
       \(c\)  $$a_{n+1} = a_n/3  +1$$ for $$n\ge 0$$; $$a_0=1$$                                      
       \(d\)  $$a_{n+2} = 2a_{n+1}-a_n$$ for $$n\ge 0$$, $$a_0=0$$, $$a_1=1$$                          
       \(e\)  $$a_{n+2} = 3a_{n+1}-2a_n+3$$ for $$n\ge 0$$; $$a_0=1$$, $$a_1=2$$                       
       \(f\)  $$a_n = 2a_{n-1}-a_{n-2}+(-1)^n$$ for $$n>1$$; $$a_0=a_1=1$$                           
       \(g\)  $$a_n = 2a_{n-1}-n\cdot(-1)^n$$ for $$n\ge 1$$; $$a_0=0$$                              
       \(h\)  $$a_n = 3a_{n-1} + {n\choose 2}$$ for $$n\ge 1$$; $$a_0=2$$                            
       \(i\)  $$a_n = 2a_{n-1}-a_{n-2}-2$$ for $$n\ge 2$$; $$a_0=a_{10}=0$$                          
       \(j\)  $$a_n = 4(a_{n-1}-a_{n-2})+(-1)^n$$ for $$n\ge 2$$; $$a_0=1$$, $$a_1=4$$                 
       \(k\)  $$a_n = -3a_{n-1}+a_{n-2}+3a_{n-3}$$ for $$n\ge 3$$; $$a_0=20$$, $$a_1=-36$$, $$a_2=60$$   
      ------- -------------------------------------------------------------------------------- --

# Ordinary generating functions

-   From the homework: solve $$a_n = 2a_{n-1}-a_{n-2}-2$$ for $$n\ge 1$$;
    $$a_0=a_{10}=0$$.\
    Applying the standard method, while keeping $$a_1$$ as a parameter, we
    get
    $$A(x)={a_1x-a_1x^2-2x^2\over (1-x)^3}={a_1x\over (1-x)^2}+{x(1-x)\over (1-x)^3}-{x^2+x\over (1-x)^3},$$
    so $$a_n=(a_1+1)n-n^2$$. From $$a_{10}=0$$ we get $$a_1=9$$, thus
    $$a_n=n(10-n)$$.

-   Another way for boundary problems (this particular example is
    motivated by splines, Wilf 10--11):

    -   consider $$au_{n+1}+bu_n+cu_{n-1}=d_n$$ for $$1\le n\le N-1$$;
        $$u_0=u_N=0$$.

    -   similar to Fibonacci with two given non-consecutive terms (but
        more general)

    -   define $$U(x)= \sum_{j=0}^N u_jx^j$$ (unknown);
        $$D(x)=\sum_{j=1}^{N-1} d_jx^j$$ (known)

    -   derive
        $$\displaystyle a\cdot {U(x)-u_1x\over x}+bU(x)+cx(U(x)-u_{N-1}x^{N-1}) = D(x)$$

    -   $$(a+bx+cx^2) U(x) = x D(x)  +au_1x + cu_{N-1}x^N$$ (\*)

    -   plug in suitable values of $$x$$ (roots $$r_{+}$$ and $$r_{-}$$ of the
        quadratic polynomial on the LHS)

    -   solve the system of two linear equations and two uknowns $$u_1$$,
        $$u_{N-1}$$

    -   if the roots are equal, differentiate (\*) to obtain the second
        equation

-   Mutually recursive sequences \[Knuth 343, Example 3\]

    -   consider the number $$u_n$$ of tilings of $$3\times n$$ board with
        $$2\times 1$$ dominoes

    -   define $$v_n$$ as the number of tilings of $$3\times n$$ board
        without a corner

    -   $$u_n = 2v_{n-1} + u_{n-2}$$; $$u_0 = 1$$; $$u_1 = 0$$

    -   $$v_n = v_{n-2} + u_{n-1}$$; $$v_0 = 0$$; $$v_1 = 1$$

    -   derive
        $$U(x) = {1-x^2\over 1-4x^2+x^4},\qquad V(x) = {x\over 1-4x^2+x^4}$$

    -   consider $$W(z) = 1/(1-4z+z^2)$$; $$U(x) = (1-x^2)W(x^2)$$, so
        $$u_{2n} = w_n - w_{n-1}$$

    -   hence
        $$u_{2n} = {(2+\sqrt 3)^n\over 3-\sqrt 3} + {(2-\sqrt 3)^n\over 3+\sqrt 3} = \left\lceil {(2+\sqrt 3)^n\over 3-\sqrt 3}\right\rceil$$
        (derivation as a homework)

-   Given $$f(x)\overset{{\rm ogf}}{\longleftrightarrow}(a_n)_{n\ge 0}$$,
    express ogf for the following sequences in terms of $$f$$ \[W1.3\]:\

      ------- --------------------------------------- -----------------------------------------------------------------------
       \(a\)  $$(a_n+c)_{n\ge 0}$$                      
       \(b\)  $$(na_n)_{n\ge 0}$$                       ; napísať im $$(P(n)a_n)_{n\ge 0} \longleftrightarrow P(x{\rm D})f(x)$$
       \(c\)  $$0, a_1, a_2, a_3, \dots$$               
       \(d\)  $$0, 0, 1, a_3, a_4, a_5,\dots$$          
       \(e\)  $$(a_{n+2}+3a_{n+1}+a_n)_{n\ge 0}$$       
       \(f\)  $$a_0, 0, a_2, 0, a_4, 0, a_6, 0\dots$$   
       \(g\)  $$a_0, 0, a_1, 0, a_2, 0, a_3, 0\dots$$   
       \(h\)  $$a_1, a_2, a_3, a_4,\dots$$              
       \(i\)  $$a_0, a_2, a_4, \dots$$                  
      ------- --------------------------------------- -----------------------------------------------------------------------

```{=html}
<!-- -->
```
-   introducing a new variable and changing the order of summation can
    help $$\begin{aligned}
            \sum_{n\ge 0} {n\choose k}x^n &=& [y^k]\sum_{m\ge 0} \left(\sum_{n\ge 0} {n\choose m}x^n\right)y^m = [y^k]\sum_{n\ge 0} (1+y)^nx^n\nonumber\\
            &=& [y^k] {1\over 1-x(1+y)} = {1\over 1-x}[y^k] {1\over 1-{x\over 1-x}y} = {x^k\over (1-x)^{k+1}} \label{binomial}
        
    \end{aligned}$$

-   alternatively, one can use binomial theorem (Knuth 199/5.56 and
    5.57): $$\begin{aligned}
            {1\over (1-z)^{n+1}} &=& (1-z)^{-n-1} =\sum_{k\ge 0} {-n-1\choose k}(-z)^k\\
                                 &=& \sum_{k\ge 0} {(-n-1)(-n-2)\dots(-n-k)\over k!}(-z)^k = \sum_{k\ge 0} {n+k\choose n}z^k
        
    \end{aligned}$$

```{=html}
<!-- -->
```
-   a ring with addition and multiplication
    $$\sum_n a_nx^n\sum_n b_nx^n = \sum_n \sum_k (a_k b_{n-k})x^n$$

-   if $$f(0)\neq 0$$, then $$f$$ has a unique reciprocal $$1/f$$ such that
    $$f\cdot 1/f = 1$$

-   composition $$f(g(x))$$ defined iff $$g(0) = 0$$ or $$f$$ is a polynomial
    (cf. $$e^{e^x-1}$$ vs. $$e^{e^x}$$)

-   formal derivative $${\rm D}$$:
    $${\rm D}\sum_n a_nx^n = \sum na_nx^{n-1}$$; usual rules for sum,
    product etc.

-   exercise: find all $$f$$ such that $${\rm D}f = f$$

```{=html}
<!-- -->
```
-   **Rule 1**: for a positive integer $$h$$,
    $$(a_{n+h})\overset{{\rm ogf}}{\longleftrightarrow}(f-a_0-\dots-a_{h-1}x^{h-1})/x^h$$

-   **Rule 2**: if $$P$$ is a polynomial, then
    $$P(x{\rm D})f\overset{{\rm ogf}}{\longleftrightarrow}(P(n)a_n)_{n\ge 0}$$

    -   example: $$(n+1)a_{n+1} = 3a_n+1$$ for $$n\ge 0$$, $$a_0 = 1$$; thus
        $$f' = 3f + 1/(1-x)$$

    -   example: $$\sum_{n\ge 0} {n^2+4n+5\over n!}$$; thus
        $$f=\sum_{n\ge 0} (n^2+4n+5){x^n\over n!} = ((x{\rm D})^2+4x{\rm D}+5)e^x = (x^2+5x+5)e^x$$\
        we need $$f(1)=11e$$; works because the resulting $$f$$ is analytic
        in a disk\
        containing $$1$$ in the complex plane (that is, it converges to
        its Taylor series)

-   **Rule 3**: if $$g\overset{{\rm ogf}}{\longleftrightarrow}(b_n)$$,
    then
    $$fg\overset{{\rm ogf}}{\longleftrightarrow}(\sum_{k=0}^n a_kb_{n-k})_{n\ge 0}$$
    $$\sum_{k=0}^n (-1)^kk = (-1)^n\sum_{k=0}^n k\cdot (-1)^{n-k} = (-1)^n[x^n]{x\over (1-x)^2}\cdot{1\over 1+x} = {(-1)^n\over 4}\left(2n+1-(-1)^n\right)$$

-   **Rule 4**: for a positive integer $$k$$, we have
    $$\displaystyle f^k\overset{{\rm ogf}}{\longleftrightarrow}\left(\sum_{n_1+n_2+\dots+n_k=n} a_{n_1}a_{n_2}\dots a_{n_k}\right)_{n\ge 0}$$

    -   example: let $$p(n,k)$$ be the number of ways $$n$$ can be written
        as an ordered sum of $$k$$ nonnegative integers

    -   according to R4,
        $$(p(n,k))_{n\ge 0}\overset{{\rm ogf}}{\longleftrightarrow}1/(1-x)^k$$,
        so $$p(n,k) = {n+k-1\choose n}$$ thanks to
        [\[binomial\]](#binomial){reference-type="eqref"
        reference="binomial"}

-   **Rule 5**:
    $$\displaystyle {f\over (1-x)}\overset{{\rm ogf}}{\longleftrightarrow}\left(\sum_{k=0}^n a_k\right)_{n\ge 0}$$\

    -   example:
        $$\displaystyle (\square_n)_{n\ge 0}\overset{{\rm ogf}}{\longleftrightarrow}{1\over 1-x}\cdot (x{\rm D})^2 {1\over 1-x} = {x(1+x)\over (1-x)^4}$$,
        so by [\[binomial\]](#binomial){reference-type="eqref"
        reference="binomial"},
        $$\square_n = {n+2\choose 3}+{n+1\choose 3}$$

1.  Using Rule 5, prove that $$F_0+F_1+\dots+F_n=F_{n+2}-1$$ for $$n\ge 0$$
    \[Wilf 38, example 6\].\

2.  Solve $$g_n=g_{n-1}+g_{n-2}$$ for $$n\ge 2$$, $$g_0 = 0$$, $$g_{10} = 10$$.\

3.  Solve $$a_n = \sum_{k=0}^{n-1}a_k$$ for $$n > 0$$; $$a_0 = 1$$. \[R16\]\

4.  Solve $$f_n=2f_{n-1}+f_{n-2}+f_{n-3}+\dots+f_1+1$$ for $$n\ge 1$$,
    $$f_0 = 0$$ \[Knuth 349/(7.41)\]\

5.  Solve $$g_n = g_{n-1} + 2g_{n-2}+\dots +ng_0$$ for $$n> 0$$, $$g_0 = 1$$.
    \[K7.7\]\

6.  Solve $$g_n = \sum_{k=1}^{n-1} {g_k + g_{n-k} + k\over 2}$$ for
    $$n\ge 2$$, $$g_1 = 1$$.

7.  Solve $$g_n=g_{n-1}+2g_{n-2}+(-1)^n$$ for $$n\ge 2$$, $$g_0 = g_1 = 1$$.
    \[Knuth 341, example 2\]\

8.  Solve $$a_{n+2}=3a_{n+1}-2a_n+n+1$$ for $$n\ge 0$$; $$a_0 = a_1 = 1$$.
    \[R24\]\

9.  Prove that
    $$\displaystyle \ln {1\over 1-x} = \sum_{n\ge 1} {1\over n} x^n$$.

# Skipping sequence elements, Catalan numbers

-   $$(1+x)^r = \sum_{k\ge 0} {r\choose k}x^k$$; consider
    $$(1+x)^r(1+x)^s = (1+x)^{r+s}$$

-   comparison of coefficients yields
    $$\sum_{k\ge 0}^n {r\choose k}{s\choose n-k}={r+s\choose n}$$ ---
    Vandermonde

-   by considering $$(1-x)^r(1+x)^r = (1-x^2)^r$$, we obtain
    $$\sum_{k=0}^n {r\choose k}{r\choose n-k}(-1)^k = (-1)^{n/2}{r\choose n/2}[2\mid n]$$

```{=html}
<!-- -->
```
-   why
    $${1\over 2}(A(x)+A(-x))\overset{{\rm ogf}}{\longleftrightarrow}a_0, 0, a_2, 0, a_4, \dots$$
    works: $${1\over 2}(1^n + (-1)^n) = [2\mid n]$$

-   in general, for $$\omega$$ being $$r$$-th root of unity,
    $${1\over r}\sum_{j=0}^{r-1} (\omega^j)^n = {1\over r}\sum_{j=0}^{r-1} e^{2\pi ijn/r} = [r\mid n]$$\
    --- just a geometric progression, or a consequence of
    $$t^r-1=(t-1)(t^{r-1}+\dots+t+1)$$

-   problem: find $$S_n = \sum_k (-1)^k{n\choose 3k}$$

-   if we knew $$f(x) = \sum_k {n\choose 3k}x^{3k}$$, we would have
    $$S_n = f(-1)$$

-   for $$A(x) = (1+x)^n$$, we have
    $$f(x) = {1\over 3}\big(A(x) + A(x\omega^1) + A(x\omega^2)\big)$$ for
    $$\omega=e^{2\pi i/3}$$

