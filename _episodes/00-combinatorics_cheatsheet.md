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

-   Given $$f(x)\overset{\text{ogf}}{\longleftrightarrow}(a_n)_{n\ge 0}$$,
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
    help 
    $$
    \begin{aligned}
      \sum_{n\ge 0} {n\choose k}x^n &=& [y^k]\sum_{m\ge 0} \left(\sum_{n\ge 0} {n\choose m}x^n\right)y^m = [y^k]\sum_{n\ge 0} (1+y)^nx^n\nonumber\\
            &=& [y^k] {1\over 1-x(1+y)} = {1\over 1-x}[y^k] {1\over 1-{x\over 1-x}y} = {x^k\over (1-x)^{k+1}} \label{binomial}
     \end{aligned}
     $$

-   alternatively, one can use binomial theorem (Knuth 199/5.56 and
    5.57): 
$$
    \begin{aligned}
            {1\over (1-z)^{n+1}} &=& (1-z)^{-n-1} =\sum_{k\ge 0} {-n-1\choose k}(-z)^k\\
                                 &=& \sum_{k\ge 0} {(-n-1)(-n-2)\dots(-n-k)\over k!}(-z)^k = \sum_{k\ge 0} {n+k\choose n}z^k
    \end{aligned}
    $$

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
    $$(a_{n+h})\overset{\text{ogf}}{\longleftrightarrow}(f-a_0-\dots-a_{h-1}x^{h-1})/x^h$$

-   **Rule 2**: if $$P$$ is a polynomial, then
    $$P(x{\rm D})f\overset{\text{ogf}}{\longleftrightarrow}(P(n)a_n)_{n\ge 0}$$

    -   example: $$(n+1)a_{n+1} = 3a_n+1$$ for $$n\ge 0$$, $$a_0 = 1$$; thus
        $$f' = 3f + 1/(1-x)$$

    -   example: $$\sum_{n\ge 0} {n^2+4n+5\over n!}$$; thus
        $$f=\sum_{n\ge 0} (n^2+4n+5){x^n\over n!} = ((x{\rm D})^2+4x{\rm D}+5)e^x = (x^2+5x+5)e^x$$\
        we need $$f(1)=11e$$; works because the resulting $$f$$ is analytic
        in a disk\
        containing $$1$$ in the complex plane (that is, it converges to
        its Taylor series)

-   **Rule 3**: if $$g\overset{\text{ogf}}{\longleftrightarrow}(b_n)$$,
    then
    $$fg\overset{\text{ogf}}{\longleftrightarrow}(\sum_{k=0}^n a_kb_{n-k})_{n\ge 0}$$
    $$\sum_{k=0}^n (-1)^kk = (-1)^n\sum_{k=0}^n k\cdot (-1)^{n-k} = (-1)^n[x^n]{x\over (1-x)^2}\cdot{1\over 1+x} = {(-1)^n\over 4}\left(2n+1-(-1)^n\right)$$

-   **Rule 4**: for a positive integer $$k$$, we have
    $$
    \displaystyle f^k\overset{\text{ogf}}{\longleftrightarrow}\left(\sum_{n_1+n_2+\dots+n_k=n} a_{n_1}a_{n_2}\dots a_{n_k}\right)_{n\ge 0}
    $$

    -   example: let $$p(n,k)$$ be the number of ways $$n$$ can be written
        as an ordered sum of $$k$$ nonnegative integers

    -   according to R4,
    -   $$(p(n,k))_{n\ge 0}\overset{\text{ogf}}{\longleftrightarrow}1/(1-x)^k$$,
        so $$p(n,k) = {n+k-1\choose n}$$ thanks to
        [\[binomial\]](#binomial)

-   **Rule 5**:
    $$
    \displaystyle {f\over (1-x)}\overset{\text{ogf}}{\longleftrightarrow}\left(\sum_{k=0}^n a_k\right)_{n\ge 0}
    $$\

    -   example:
        $$\displaystyle (\square_n)_{n\ge 0}\overset{\text{ogf}}{\longleftrightarrow}{1\over 1-x}\cdot (x{\mathrm D})^2 {1\over 1-x} = {x(1+x)\over (1-x)^4}$$,
        so by [\[binomial\]](#binomial),
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
    $${1\over 2}(A(x)+A(-x))\overset{\text{ogf}}{\longleftrightarrow}a_0, 0, a_2, 0, a_4, \dots$$
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
 and so $$S_n=f(-1) ={1\over 3}[(1-\omega)^n + (1-\omega^2)^n)] =$$
 
   $$
    = {1\over 3}\left[\left({3-\sqrt3 i\over 2}\right)^n+\left({3+\sqrt3 i\over 2}\right)^n\right] = 2\cdot 3^{\frac{n}{2}-1}\cos\left({\pi n\over 6}\right)
    $$
 
```{=html}
<!-- -->
```
-   consider the number of possibilities $$c_n$$ of how to specify the
    multiplication order of $$A_0A_1\dots A_n$$ by parentheses; let
    $$C(x)=\sum_{n\ge 0} c_nx^n$$

-   divide possibilities by the place of last multiplication;
    $$c_n = \sum\limits_{k=0}^{n-1} c_kc_{n-1-k}$$ for $$n > 0$$; $$c_0=1$$

-   many ways to deal with the recurrence:

    -   shift the recurrence to $$c_{n+1} = \sum_{k=0}^n c_kc_{n-k}$$ and
        use Rules 1 and 3; $${C(x)-1\over x} = C(x)^2$$

    -   RHS as a convolution of $$c_n$$ with $$c_{n-1}$$, i.e.
        $$C(x)\cdot xC(x)$$

    -   RHS as a convolution of $$c_n$$ with $$c_n$$ shifted by Rule 1, i.e.
        $$x\cdot C(x)^2$$

    -   rewriting through sums and changing the order of summation:
        $$\sum_{n\ge 1}x^n\sum_{k=0}^{n-1}c_kc_{n-1-k}=\sum_{k=0}^\infty x^kc_k\sum_{n\ge k+1} c_{n-1-k}x^{n-k}=
                                \sum_{k=0}^\infty x^kc_k xC(x)=xC(x)\cdot C(x)$$

-   consequently, $$C(x) - 1 = xC(x)^2$$ and thus
    $$C(x) = {1\pm \sqrt{1-4x}\over 2x}=\displaystyle {1\over 2x}\left(1 - \sqrt{1-4x}\right)$$

-   we want $$C$$ continuous and $$C(0) = 1$$, so we choose the minus sign
    (note that the resulting function below is analytical since
    $${2n\choose n}/(n+1) < 2^{2n}$$; it would be analytical also if we
    chose the plus sign)

-   binomial theorem yields $$\begin{aligned}
            \sqrt{1-4x} = (1-4x)^{1/2} = \sum_{k\ge 0} {1/2\choose k}(-4x)^k &=& 1+\sum_{k\ge 1}{1\over 2k\cdot (-4)^{k-1}}{2k-2\choose k-1}(-4)^kx^k\\
            &=& 1 - \sum_{k\ge 1}{2\over k}{2k-2\choose k-1}x^k 
        \end{aligned}$$

-   we used
    $${1/2\choose k}={1/2\over k}{-1/2\choose k-1} = {1\over 2k(-4)^{k-1}}{2k-2\choose k-1}$$
    because $${-1/2\choose m}={1\over (-4)^m}{2m\choose m}$$

-   therefore,
    $$C(x)={1\over 2x}\sum_{k\ge 1}{2\over k}{2k-2\choose k-1}x^k = \sum_{n\ge 0}{1\over n+1}{2n\choose n}x^n$$

1.  Assume that $$A(x)\overset{\text{ogf}}{\longleftrightarrow}(a_n)$$.
    Express the generating function for $$\sum_{n\ge 0} a_{3n}x^n$$ in
    terms of $$A(x)$$.\

2.  Compute $$S_n=\sum_{n\ge 0} F_{3n}\cdot 10^{-n}$$ (by plugging a
    suitable value into the generating function for $$F_{3n}$$).\

3.  Compute $$\sum_k {n\choose 4k}$$.

4.  Compute $$\sum_k {6m\choose 3k+1}$$.

5.  Evaluate $$S_n = \sum_{k=0}^n (-1)^k k^2$$.

6.  Find ogf for $$H_n = 1 + 1/2 + 1/3 + \dots$$.

7.  Find the number of ways of cutting a convex $$n$$-gon with labelled
    vertices into triangles.\

# Snake Oil

The Snake Oil method \[Wilf 118, chapter 4.3\] -- external method vs.
internal manipulations within a sum.

1.  identify the free variable and give the name to the sum, e.g. $$f(n)$$

2.  let $$F(x) = \sum f(n)x^n$$

3.  interchange the order of summation; solve the inner sum in closed
    form

4.  find coefficients of $$F(x)$$

-   Example 0

    -   let's evaluate $$f(n) = \sum_k {n\choose k}$$; after Step 2,
        $$F(x) = \sum_{n\ge 0} x^n \sum_k {n\choose k}$$

    -   $$\displaystyle F(x) = \sum_k \sum_n {n\choose k}x^n = \sum_{k\ge 0} {x^k\over (1-x)^{k+1}}={1\over 1-x}\cdot {1\over 1-{x\over 1-x}}={1\over 1-2x}$$

-   Example 1 \[Wilf 121\]

    -   let's evaluate $$f(n) = \sum_{k\ge 0} {k\choose n-k}$$; after Step
        2, $$F(x) = \sum_n x^n \sum_{k\ge 0} {k\choose n-k}$$

    -   $$\displaystyle F(x) = \sum_{k\ge 0} \sum_n {k\choose n-k}x^n = \sum_{k\ge 0}x^k\sum_n {k\choose n-k}x^{n-k} = \sum_{k\ge 0}x^k (1+x)^k = {1\over 1-x-x^2}$$

    -   so $$f(n) = F_{n+1}$$

-   Example 2 \[Wilf 122\]

    -   let's evaluate
        $$f(n) = \sum_{k} {n+k\choose m+2k}{2k\choose k}{(-1)^k\over k+1}$$,
        where $$m$$, $$n$$ are nonnegative integers $$\begin{aligned}
F(x) &=& \sum_{n\ge 0} x^n \sum_{k} {n+k\choose m+2k}{2k\choose k}{(-1)^k\over k+1} \\
     &=& \sum_k {2k\choose k}{(-1)^k\over k+1}x^{-k}\sum_{n\ge 0}{n+k\choose m+2k}x^{n+k}\\
     &=& \sum_k {2k\choose k}{(-1)^k\over k+1}x^{-k}{x^{m+2k}\over (1-x)^{m+2k+1}}\\
     &=& {x^m\over (1-x)^{m+1}}\sum_k {2k\choose k}{1\over k+1}\left({-x\over (1-x)^2}\right)^k \\
     &=& {-x^{m-1}\over 2(1-x)^{m-1}}\left(1-\sqrt{1+{4x\over (1-x)^2}}\right) = {x^m\over (1-x)^m}
\end{aligned}$$


    -   so $$f(n) = {n-1\choose m-1}$$

-   Example 6 \[Wilf 127\]

    -   prove that
        $$\sum_{k} {m\choose k}{n+k\choose m} = \sum_k {m\choose k}{n\choose k}2^k$$,
        where $$m$$, $$n$$ are nonnegative integers

    -   the ogf of the left-hand side is
        $$L(x) = \sum_{k} {m\choose k} x^{-k}\sum_{n\ge 0}{n+k\choose m}x^{n+k} ={(1+x)^m\over (1-x)^{m+1}}$$

    -   we get the same for the right-hand side

1.  Prove that $$\sum_k k{n\choose k} = n2^{n-1}$$ via the snake oil
    method.

2.  Evaluate $$\displaystyle f(n)=\sum_k k^2{n\choose k}3^k$$.\

3.  Find a closed form for
    $$\displaystyle \sum_{k\ge 0} {k\choose n-k}t^k$$. \[W4.11(a)\]\

4.  Evaluate $$\displaystyle f(n)=\sum_k {n+k\choose 2k}2^{n-k}$$,
    $$n\ge 0$$. \[Wilf 125, Example 4\]\

5.  Evaluate
    $$\displaystyle f(n)=\sum_{k\le n/2} (-1)^k{n-k\choose k}y^{n-2k}$$.
    \[Wilf 122, Example 3\]\

6.  Evaluate
    $$\displaystyle f(n)=\sum_{k} {2n+1\choose 2p+2k+1}{p+k\choose k}$$.
    \[W4.11(c)\]\

7.  Try to prove that $$\sum_k {n\choose k}{2n\choose n+k}={3n\choose n}$$
    via the snake oil method in three different ways: consider the sum
    $$\sum_k {n\choose k}{m\choose r-k}$$ and the free variable being
    one of $$n$$, $$m$$, $$r$$.

# Asymptotic estimates

-   Purpose of asymptotics \[Knuth 439\]

    -   sometimes we do not have a closed form or it is hard to compare
        it to other quantities

    -   $$\displaystyle S_n = \sum_{k=0}^n {3n\choose k}\sim 2{3n\choose n}$$;
        $$\displaystyle S_n = {3n\choose n}\left(2-{4\over n} + O\left({1\over n^2}\right)\right)$$

    -   how to compare it with $$F_{4n}$$? we need to approximate the
        binomial coefficient

    -   purpose is to find *accurate* and *concise* estimates:\
        $$H_n$$ is
        $$\sum_{k\ge 1}^n 1/k$$vs.$$O(\log n)$$vs.$$\ln n + \gamma + O(n^{-1})$$

-   Hierarchy of log-exp functions \[Hardy, see Knuth 442\]

    -   the class $$\cal L$$ of logarithmico-exponential functions: the
        smallest class that contains constants, identity function
        $$f(n) = n$$, difference of any two functions from $$\cal L$$, $$e^f$$
        for every $$f\in {\cal L}$$, $$\ln f$$ for every $$f\in {\cal L}$$
        that is "eventually positive"

    -   every such function is identically zero, eventually positive or
        eventually negative

    -   functions in $$\cal L$$ form a hierarchy (every two of them are
        comparable by $$\prec$$ or $$\asymp$$)

-   Notations

    -   $$f(n) = O(g(n))$$ iff $$\exists c: |f(n)|\le c|g(n)|$$
        (alternatively, for $$n\ge n_0$$ for some $$n_0$$)

    -   $$f(n) = o(g(n))$$ iff $$\lim_{n\to\infty} f(n)/g(n) = 0$$

    -   $$f(n) = \Omega(g(n))$$ iff $$\exists c: |f(n)|\ge c|g(n)|$$
        (alternatively, ...)

    -   $$f(n) = \Theta(g(n))$$ iff $$f(n) = O(g(n))$$ and
        $$f(n) = \Omega(g(n))$$

    -   basic manipulation: $$O(f)+O(g) = O(|f|+|g|)$$,
        $$O(f)O(g)=O(fg)=fO(g)$$ etc.

    -   meaning of $$O$$ in sums

    -   *relative* vs. *absolute* error

-   Warm-ups

    1.  Prove or disprove: $$O(f+g)=f + O(g)$$ if $$f$$ and $$g$$ are
        positive. \[K9.5\]

    2.  Multiply $$\ln n + \gamma + O(1/n)$$ by $$n + O(\sqrt n)$$. \[K9.6\]

    3.  Compare $$n^{\ln n}$$ with $$(\ln n)^n$$.

    4.  Compare $$n^{\ln\ln\ln n}$$ with $$(\ln n)!$$.

    5.  Prove or disprove: $$O(x+y)^2 = O(x^2) + O(y^2)$$. \[K9.11\]

-   Common tricks

    -   cut off series expansion (works for convergent series, Knuth
        451)

    -   substitution, e.g. $$\ln(1+2/n^2)$$ with precision of $$O(n^{-5})$$

    -   factoring (pulling the large part out), e.g.
        $${1\over n^2+n} = {1\over n^2}{1\over 1+{1\over n}}={1\over n^2}-{1\over n^3}+O(n^{-4})$$

    -   division, e.g.
        $$\displaystyle {H_n\over \ln (n + 1)}= {\ln n + \gamma + O(n^{-1})\over (\ln n)(1+O(n^{-1}))}=1 + {\gamma\over \ln n} + O(n^{-1})$$

    -   exp-log, i.e. $$f(x) = e^{\ln f(x)}$$

-   Typical situations for approximation

    -   Stirling formula:
        $$\displaystyle n! = \sqrt{2\pi n}\left({n\over e}\right)^n\left(1+{1\over 12n}+{1\over 288n^2}+O(n^{-3})\right)$$

    -   harmonic numbers:
        $$H_n = \ln n + \gamma + {1\over 2n} - {1\over 12n^2} + O(n^{-4})$$

    -   rational functions, e.g.
        $${n\over n+2} = {1\over 1+{2\over n}} = 1-{2\over n}+{4\over n^2}+O(n^{-3})$$

    -   exponentials:
        $$e^{H_n}=ne^\gamma e^{O(1/n)}=ne^\gamma (1+O(1/n))=ne^\gamma + O(1)$$

    -   rational function powered to $$n$$, e.g.
        $$\left(1-{1\over n}\right)^n=e^{n\ln \left(1-{1\over n}\right)}= \exp\left(n\left({-1\over n}+O\left(n^{-2}\right)\right)\right) = e^{-1 + O(n^{-1}))} = {1\over e} + O(n^{-1})$$

    -   binomial coefficient, e.g. $$2n\choose n$$: factorials and
        Stirling formula $$\begin{aligned}
                        {2n\choose n}={\sqrt{4\pi n}\left({2n\over e}\right)^{2n}(1+O(n^{-1}))\over 2\pi n\left({n\over e}\right)^{2n}(1+O(n^{-1}))^2}=
                                {{2^{2n}}\over \sqrt{\pi n}}(1+O(n^{-1}))
                    
        \end{aligned}$$

-   Exercises

    1.  Estimate $$\ln(1+1/n)+ \ln(1-1/n)$$ with abs. error $$O(n^{-3})$$

    2.  Estimate $$\ln(2+1/n)- \ln(3-1/n)$$ with abs. error $$O(n^{-2})$$

    3.  Estimate $$\lg (n-2)$$, abs. error $$O(n^{-2})$$

    4.  Evaluate $$H_n^2$$ with abs. error $$O(n^{-1})$$.

    5.  Estimate $$n^3/(2+n+n^2)$$ with abs. error $$O(n^{-3})$$

    6.  Prove or disprove: \[K9.20\] (b) $$e^{(1+O(1/n))^2} = e + O(1/n)$$
        cm (c) $$n! = O\left(((1-1/n)^nn)^n\right)$$

    7.  Evaluate $$(n+2+O(n^{-1}))^n$$ with rel. error $$O(n^{-1})$$.
        \[K9.13\]

    8.  Compare $$H_{F_n}$$ with $$F_{\lceil H_n\rceil}^2$$ \[K9.2\]

    9.  Estimate $$\sum_{k\ge 0} e^{-k/n}$$ with abs. error $$O(n^{-1})$$.
        \[K9.7\]

    10. Estimate $$H_n^5/\ln (n + 5)$$ with abs. error $$O(n^{-2})$$.

    11. Estimate $$2n\choose n$$ with relative error $$O(n^{-2})$$. \[A1\]

    12. Estimate $$2n+1\choose n$$ with relative error $$O(n^{-2})$$. \[A2\]

    13. Compare $$(n!)!$$ with $$((n-1)!)!\cdot (n-1)!^{n!}$$. \[K9.2c\]
        (Homework if not enough time is left.)

# Estimates of sums and products

-   Warm-ups

    1.  Let $$f(n) = \sum_{k=1}^n \sqrt k$$. Show that
        $$f(n) = \Theta(n^{3/2})$$. Find $$g(n)$$ such that
        $$f(n) = g(n) + O(\sqrt n)$$.

    2.  Estimate $$(n-2)!/(n-1)$$ with abs. error $$O(n^{-2})$$.

    3.  For a constant integer $$k$$, estimate $$n^{\underline{k}}/n^k$$
        with abs. error $$O(n^{-3})$$. \[A5\]\

-   Find a good estimate of $$P_n = {(2n-1)!!\over n!}$$.

    -   obviously
        $$\displaystyle 1.5^{n-1}\le {1\over 1}\cdot {3\over 2}\cdot {5\over 3}\cdot \dots \cdot {(2n-1)\over n}\le 2^{n-1}$$

    -   we split the product into a "small" part (first $$k$$ terms, each
        at least $$3/2$$ except the first one) and a "large" part
        (remaining $$n-k$$ terms); then\
        $$P_n\ge \left({2k+1\over k+1}\right)^{n-k}\cdot 1.5^{k-1} = Q_n\cdot 1.5^{k-1}$$;
        we estimate $$Q_n$$

    -   if we try $$k = \alpha n$$, then
        $$Q_n = 2^{n-\alpha n} \exp \left((n-\alpha n)\ln \left(1-{1\over 2(\alpha n + 1)}\right)\right)=2^{n(1-\alpha)}e^{{\alpha-1\over 2\alpha}}(1+O(n^{-1})),$$
        so $$P_n \ge (2^{1-\alpha}\cdot 1.5^\alpha)^n \Theta(1)$$

    -   if we try $$k = \ln n$$, then
        $$Q_n = \exp\left((n-\ln n)\left[\ln 2 + \ln \left(1-{1\over 2(1+\ln n)}\right)\right]\right);$$
        if we expand $$\ln$$ into Taylor series, the error will be
        $$1/\ln^k n = \omega(n^{-1})$$, so we can get relative error
        $$O(1)$$ at best;\
        anyway, if we carry it through, we get
        $$P_n = \Omega(2^n n^{-c} e^{-0.5n/\ln n})$$

    -   If we try $$k = \sqrt n$$, then
$$\begin{aligned}
    Q_n &= \exp\left((n-\sqrt n)\left[\ln 2 + \ln \left(1-{1\over 2(1+\sqrt n)}\right)\right]\right)\\
        &= 2^{n-\sqrt n}\exp\left((n-\sqrt n)\left[{-1\over 2\sqrt n} + {3\over 8n}-{7\over 24n^{3/2}}+O(n^{-2})\right]\right)\\
        &= 2^{n-\sqrt n}\exp\left(-{\sqrt n\over 2} + {7\over 8}-{2\over 3\sqrt n}+O(n^{-1})\right),
 \end{aligned}$$
thus
$$P_n \ge 2^n \cdot 0.75^{\sqrt n}\cdot e^{\frac{-\sqrt n}{2}+\frac{7}{8}-\frac{2}{3\sqrt n}} (1+O(n^{-1})) = \Omega\left(2^n c^{\sqrt n}\right)$$
for $$c\in (0, 1)$$. 


    -   TODO compare with previous estimate from $$k=\ln n$$; which is
        better?

    -   another approach:
        $$P_n = {(2n)!\over n! 2^n n!} = {2n\choose n}/2^n = {2^n\over \sqrt{\pi n}}(1+O(n^{-1}))$$

-   Estimate $$S_n = \sum_{k=1}^n {1\over n^2+k}$$ with absolute error (a)
    $$O(n^{-3})$$, (b) $$O(n^{-7})$$. \[Knuth 458/Problem 4\] First
    approach: $${1\over n^2+k}={1\over n^2(1+k/n^2)}$$ etc.; second
    approach: $$S_n = H_{n^2+n}-H_n$$. (DU)

-   Sums --- gross bound on the tail:
    $$S_n = \sum_{0\le k\le n} k! = n!\left(1+{1\over n}+{1\over n(n-1)}+ \dots\right)$$,
    all the terms except the first two are at most $$1/n(n+1)$$, so
    $$S_n = n!(1+{1\over n}+n{1\over n(n-1)}) = n!(1+O(n^{-1}))$$

-   Sums --- make the tail infinite: $$\begin{aligned}
    n!\sum_{k=0}^n{(-1)^k\over k!} &= n!\left(\sum_{k=0}^\infty{(-1)^k\over k!}-\sum_{k\ge n+1}{(-1)^k\over k!}\right)\\
                                   &= n!\left(e^{-1}-O\left({1\over (n+1)!}\right)\right)= {n!\over e}+O(n^{-1})
    \end{aligned}$$

-   Estimate $$S_n=\sum_{k=0}^n {3n\choose k}$$ with relative error
    $$O(n^{-2})$$. We split the sum into a "small" and a "large" part at
    $$b$$ (which is yet to be determined). 
    $$
    \begin{aligned}
\sum_{k=0}^{n} \binom {3n}k &= \sum_{k=0}^{n} \binom {3n}{n-k} = \sum_{0\leq k<b} \binom {3n}{n-k}+\sum_{b\le k\le n} \binom {3n}{n-k}.\\
\binom{3n}{n-k} &= \binom{3n}{n} \frac{n(n-1)\cdot\ldots \cdot 1}{(2n+1)(2n+2)\ldots(2n+k)} =\\
                 &= \binom{3n}{n}\cdot\frac{n^k}{(2n)^k}\frac{\prod_{j=0}^{k-1}\left(1-\frac jn\right)}{\prod_{j=1}^k \left(1+\frac j{2n}\right)}=\binom{3n}{n}\cdot\frac{1}{2^k}\cdot\left[1-\frac{3k^2-k}{4n}+O\left(\frac{k^4}{n^2}\right)\right].\\
\sum_{b\le k\le n} \binom {3n}{n-k} &\le n\cdot \binom{3n}{n-b}=\binom{3n}{n}\cdot \frac{1}{2^b} O(n)=\binom{3n}{n}\cdot O\left(n^{-2}\right) \text{if } \sqrt{n} \succ b \geq 3\lg n.\\
\sum_{0\leq k<3\lg n}\frac{1}{2^k} &= 2-\frac{1}{2^{3\lg n}}=2+O(n^{-3}).\\
-\frac{3}{4n}\sum_{0\leq k<3\lg n}\frac{k^2}{2^k} &= \frac{-9}{2n}+O(n^{-3}).\\
+\frac{1}{4n}\sum_{0\leq k<3\lg n}\frac{k}{2^k} &= \frac{1}{2n}+O(n^{-3}).\\
O(n^{-2})\cdot\sum_{0\leq k<3\lg n}\frac{k^4}{2^k} &= O(n^{-2})
\end{aligned}
 $$

-   Estimate $$S_n=\sum_{k=0}^n \binom{4n+1}{k+1}$$ with relative error
    $$O(n^{-2})$$. $$\binom{4n+1}{k+1}=\binom{4n}{k+1}+\binom{4n}{k};$$
    $$S_n=\sum_{k=0}^n \binom{4n+1}{k+1}=\sum_{k=0}^n\binom{4n}{k}+ \sum_{k=0}^n\binom{4n}{k+1}=\sum_{k=0}^n\binom{4n}{k}+\sum_{k=1}^{n+1}\binom{4n}{k};$$
    $$S_n=2\sum_{k=0}^n\binom{4n}{k}+\binom{4n}{n+1}-\binom{4n}{0}.$$
    $$Q_n=\sum_{k=0}^n\binom{4n}{k}=\sum_{k=0}^n\binom{4n}{n-k};$$
    $$\binom{4n}{n-k}=\binom{4n}{n}\cdot\frac{\prod_{j=0}^{k-1}(n-j)}{\prod_{j=1}^{k}(3n+j)}=\binom{4n}{n}\cdot\left(\frac 13\right)^3\cdot\frac{\prod_{j=0}^{k-1}(1-j/n)}{\prod_{j=1}^{k}(1+j/3n)}$$
    $$Q_n=\sum_{0\leq k\leq 2\log_3 n}\binom{4n}{n-k}+\sum_{2\log_3 n\leq k<n}\binom{4n}{n-k}$$
    $$\sum_{2\log_3 n\leq k<n}\binom{4n}{n-k}=O\left(n\cdot\binom{4n}{n-\lceil 2\log_3 n\rceil} \right)=O\left(\binom{4n}{n}\cdot\frac 1n \right).$$
    $$\frac{\prod_{j=0}^{k-1}(1-j/n)}{\prod_{j=1}^{k}(1+j/3n)}=\frac{1-\frac 1n\cdot\sum_{0\leq j<k}j+O\left(\frac{k^4}{n^2}\right)}{1+\frac{1}{3n}\cdot\sum_{1< j\leq k}j+O\left(\frac{k^4}{n^2}\right)} = 1+\frac{2k^2+k}{3n}+O\left(\frac{\log^n}{n^2}\right),$$
    $$\sum_{0\leq k\leq 2\log_3 n}\binom{4n}{n-k}=\binom{4n}{n}\cdot\sum_{0\leq k\leq 2\log_3 n}\left( \frac 13\right)^k\cdot[1+\frac{2k^2+k}{3n}+O\left(\frac{\log^n}{n^2}\right)]=$$
    $$=\frac 32\cdot \binom{4n}{n} (1+O(n^{-1})).$$
    $$\binom{4n}{n+1}= \binom{4n}{n}\cdot\frac{3n}{n+1}=3\cdot\binom{4n}{n}(1+O(n^{-1}));$$
    $$S_n=6\cdot\binom{4n}{n}(1+O(n^{-1})).$$

-   How many bits are needed to represent a binary tree with $$n$$
    internal nodes?

    -   we need just the internal vertices to capture the structure;
        what is the relation between the number of internal vertices and
        total number of vertices?

    -   imagine labeling the vertices by $$1,2,\dots,n$$ in such a way
        that we get a binary search tree (descendants in the left
        subtree are smaller, in the right subtree are larger); by
        summing over possible roots of the tree we get
        $$t_n = \sum_{i=1}^n t_{i-1} t_{n-i}$$; $$t_0 = 1$$

    -   this is the same as for Catalan numbers, so
        $$t_n = {2n\choose n}{1\over n+1}$$
 
---

> ## Find explicit formulas for the following sequences:
> ### 1) $$a_{n+1} = 3a_n+2$$ for $$n\ge 0$$, $$a_0=0$$
> > 
> > ## Solution
> > $$3x/(1-x)(1-3x)$$\
> > $$3^n-1$$
> >
>{: .solution}
{: .challenge}
> ### 2. $$a_{n+1} = \alpha a_n + \beta$$ for $$n\ge 0$$, $$a_0=0$$
> > 
> > ## Solution
> > $$\beta x/(1-x)(1-\alpha x)$$\
> > $${\alpha^n-1\over \alpha-1}\beta$$
> >
>{: .solution}
{: .challenge}
> ### 3. $$a_{n+1} = a_n/3  +1$$ for $$n\ge 0$$, $$a_0=1$$
> > 
> > ## Solution
> >
> > $${3/2\over 1-x}-{1/2\over 1-x/3}$$\
> > $${3^{n+1}-1\over 2\cdot 3^n}$$
> >
>{: .solution}
{: .challenge}
> ### 4. $$a_{n+2} = 2a_{n+1}-a_n$$ for $$n\ge 0$$, $$a_0=0$$, $$a_1=1$$
> > 
> > ## Solution
> >
> > $$x/(1-x)^2$$\
> > $$n$$
> >
>{: .solution}
{: .challenge}
> ### 5. $$a_{n+2} = 3a_{n+1}-2a_n+3$$ for $$n>0$$, $$a_0=1$$, $$a_1=2$$
> > 
> > ## Solution
> >
> > $${4\over 1-2x}-{3\over (1-x)^2}$$\
> > $$2^{n+2}-3n-3$$
> >
>{: .solution}
{: .challenge}
> ### 6. $$a_n = 2a_{n-1}-a_{n-2}+(-1)^n$$ for $$n>1$$, $$a_0=a_1=1$$
> > 
> > ## Solution
> >
> > $${1/2\over (1-x)^2}-{1/4\over 1-x}+{1/4\over 1+x}$$\
> > $${2n+3+(-1)^n\over 4}$$
> >
>{: .solution}
{: .challenge}
> ### 7. $$a_n = 2a_{n-1}-n\cdot(-1)^n$$ for $$n\ge 1$$, $$a_0=0$$
> > 
> > ## Solution
> >
> > $${x/9-2/9\over (1+x)^2}+{2/9\over 1-2x}$$\
> > $${2^{n+1}-(3n+2)(-1)^n\over 9}$$
> >
>{: .solution}
{: .challenge}
> ### 8. $$a_n = 3a_{n-1} + {n\choose 2}$$ for $$n\ge 1$$, $$a_0=2$$
> > 
> > ## Solution
> >
> > $${1\over 8}(19\cdot 3^n-2n(n+2)-3)$$
> >
>{: .solution}
{: .challenge}
> ### 9. $$a_n = 2a_{n-1}-a_{n-2}-2$$ for $$n > 1$$, $$a_0=a_{10}=0$$
> > 
> > ## Solution
> >
> > $$n(a_1+1-n)$$, so with $$a_{10}$$, $$a_n=n(10-n)$$
> >
>{: .solution}
{: .challenge}
> ### 10. $$a_n = 4(a_{n-1}-a_{n-2})+(-1)^n$$ for $$n \ge 2$$, $$a_0=1$$, $$a_1=4$$
> > 
> > ## Solution
> >
> > $${1+x+x^2\over (1+x)(1-2x)^2} = {1\over 9}{1\over 1+x} +\left({-5\over 18}\right) {1\over 1-2x} + \left({7\over 6}\right){1\over (1-2x)^2}$$\
> > $${1\over 9}(-1)^n-{5\over 18}\cdot 2^n+{7\over 6}(n+1)\cdot 2^n$$
> >
>{: .solution}
{: .challenge}
> ### 11. $$a_n = -3a_{n-1}+a_{n-2}+3a_{n-3}$$ for $$n\ge 3$$, $$a_0=20$$, $$a_1=-36$$, $$a_2=60$$
> > 
> > ## Solution
> >
> > $$5(-3)^n+18(-1)^n-3$$
> >
>{: .solution}
{: .challenge}
> ### 12. $$a_n = -3a_{n-1}+a_{n-2}+3a_{n-3}+128n$$ for $$n\ge 3$$, $$a_0=0$$, $$a_1=0$$, $$a_2=0$$
> > 
> > ## Solution
> >
> > $$8n^2+28n-29-11(-3)^n+40(-1)^n$$
> >
>{: .solution}
{: .challenge}

