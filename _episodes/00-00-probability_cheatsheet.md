---
title: "Probability CheatSheet"
questions:
- ""
objectives:
- ""
---

# Counting
---

## Multiplication Rule

![icecream](figures/icecream.pdf)

Let's say we have a compound experiment (an experiment with multiple components). If the 1st component has $n_1$ possible outcomes, the 2nd component has $n_2$ possible outcomes, ..., and the $r$th component has $n_r$ possible outcomes, then overall there are $n_1n_2 \dots n_r$ possibilities for the whole experiment.

## Sampling Table

![jar](figures/jar.pdf)

The sampling table gives the number of possible samples of size $k$ out of a population of size $n$, under various assumptions about how the sample is collected.

|                      | Order Matters          | Not Matter                    |
|----------------------|------------------------|-------------------------------|
| With Replacement     | $n^k$                  | ${n+k-1 \choose k}$           |
| Without Replacement  | $\frac{n!}{(n - k)!}$  | ${n \choose k}$               |

## Naive Definition of Probability

If all outcomes are equally likely, the probability of an event $A$ happening is:

\[P_{\textrm{naive}}(A) = \frac{\textnormal{number of outcomes favorable to $A$}}{\textnormal{number of outcomes}}\]

### Independence

- Independent Events: $A$ and $B$ are independent if knowing whether $A$ occurred gives no information about whether $B$ occurred. More formally, $A$ and $B$ (which have nonzero probability) are independent if and only if one of the following equivalent statements holds:
    - $P(A \cap B) = P(A)P(B)$
    - $P(A|B) = P(A)$
    - $P(B|A) = P(B)$

- Conditional Independence: $A$ and $B$ are conditionally independent given $C$ if $P(A \cap B|C) = P(A|C)P(B|C)$. Conditional independence does not imply independence, and independence does not imply conditional independence.

### Unions, Intersections, and Complements

- De Morgan's Laws: A useful identity that can make calculating probabilities of unions easier by relating them to intersections, and vice versa. Analogous results hold with more than two sets.
    - $(A \cup B)^c = A^c \cap B^c$
    - $(A \cap B)^c = A^c \cup B^c`
### Joint, Marginal, and Conditional

- Joint Probability: $P(A \cap B)$ or $P(A, B)$ -- Probability of $A$ and $B$.
- Marginal (Unconditional) Probability: $P(A)$ -- Probability of $A$.
- Conditional Probability: $P(A|B) = \frac{P(A, B)}{P(B)}$ -- Probability of $A$, given that $B$ occurred.
- Conditional Probability *is* Probability: $P(A|B)$ is a probability function for any fixed $B$. Any theorem that holds for probability also holds for conditional probability.

### Probability of an Intersection or Union

**Intersections via Conditioning**

\begin{align*} 
    P(A,B) &= P(A)P(B|A) \\
    P(A,B,C) &= P(A)P(B|A)P(C|A,B)
\end{align*}

**Unions via Inclusion-Exclusion**

\begin{align*} 
    P(A \cup B) &= P(A) + P(B) - P(A \cap B) \\
    P(A \cup B \cup C) &= P(A) + P(B) + P(C) \\
    &\quad - P(A \cap B) - P(A \cap C) - P(B \cap C) \\
    &\quad + P(A \cap B \cap C)
\end{align*}

### Simpson's Paradox

![Simpson's Paradox](figures/SimpsonsParadox.pdf)

It is possible to have
\[P(A\mid B,C) < P(A\mid B^c, C) \text{ and } P(A\mid B, C^c) < P(A \mid B^c, C^c)\]
\[ \text{yet also } P(A\mid B) > P(A \mid B^c).\]

### Law of Total Probability (LOTP)

Let ${ B}_1, { B}_2, { B}_3, ... { B}_n$ be a *partition* of the sample space (i.e., they are disjoint and their union is the entire sample space).

\begin{align*} 
    P({ A}) &= P({ A} | { B}_1)P({ B}_1) + P({ A} | { B}_2)P({ B}_2) + \dots + P({ A} | { B}_n)P({ B}_n)\\
    P({ A}) &= P({ A} \cap { B}_1)+ P({ A} \cap { B}_2)+ \dots + P({ A} \cap { B}_n)
\end{align*} 

For **LOTP with extra conditioning**, just add in another event $C$!

\begin{align*} 
    P({ A}| { C}) &= P({ A} | { B}_1, { C})P({ B}_1 | { C}) + \dots +  P({ A} | { B}_n, { C})P({ B}_n | { C})\\
    P({ A}| { C}) &= P({ A} \cap { B}_1 | { C})+ P({ A} \cap { B}_2 | { C})+ \dots +  P({ A} \cap { B}_n | { C})
\end{align*} 

Special case of LOTP with ${ B}$ and ${ B^c}$ as partition:

\begin{align*} 
P({ A}) &= P({ A} | { B})P({ B}) + P({ A} | { B^c})P({ B^c}) \\
P({ A}) &= P({ A} \cap { B})+ P({ A} \cap { B^c})
\end{align*} 

### Bayes' Rule**Bayes' Rule, and with extra conditioning (just add in $C$!)**

\[P({ A}|{ B})  = \frac{P({ B}|{ A})P({ A})}{P({ B})}\]

\[P({ A}|{ B}, { C}) = \frac{P({ B}|{ A}, { C})P({ A} | { C})}{P({ B} | { C})}\]

We can also write

\[P(A|B,C) = \frac{P(A,B,C)}{P(B,C)} = \frac{P(B,C|A)P(A)}{P(B,C)}\]

**Odds Form of Bayes' Rule**\[\frac{P({ A}| { B})}{P({ A^c}| { B})} = \frac{P({ B}|{ A})}{P({ B}| { A^c})}\frac{P({ A})}{P({ A^c})}\]

The *posterior odds* of $A$ are the *likelihood ratio* times the *prior odds*.

# Random Variables and their Distributions
---

## PMF, CDF, and Independence

**Probability Mass Function (PMF)**  
Gives the probability that a *discrete* random variable takes on the value *x*.

\[ p_X(x) = P(X=x) \]

The PMF satisfies:  
\[p_X(x) \geq 0 \quad \textrm{and} \quad \sum_x p_X(x) = 1\]

**Cumulative Distribution Function (CDF)**  
Gives the probability that a random variable is less than or equal to *x*.

\[F_X(x) = P(X \leq x)\]

The CDF is an increasing, right-continuous function with:  
\[F_X(x) \to 0 \quad \textrm{as} \quad x \to -\infty \quad \textrm{and} \quad F_X(x) \to 1 \quad \textrm{as} \quad x \to \infty\]

**Independence**  
Intuitively, two random variables are independent if knowing the value of one gives no information about the other. Discrete random variables *X* and *Y* are independent if for all values of *x* and *y*:

\[P(X=x, Y=y) = P(X = x)P(Y = y)\]

## Expected Value and Indicators
---

### Expected Value and Linearity

**Expected Value**  
(a.k.a. *mean*, *expectation*, or *average*) is a weighted average of the possible outcomes of our random variable. Mathematically, if *x<sub>1</sub>*, *x<sub>2</sub>*, *x<sub>3</sub>*, ... are all of the distinct possible values that *X* can take, the expected value of *X* is:

\[E(X) = \sum\limits_{i}x_iP(X=x_i)\]

**Linearity**  
For any random variables *X* and *Y*, and constants *a*, *b*, *c*:

\[E(aX + bY + c) = aE(X) + bE(Y) + c\]

**Same distribution implies same mean**  
If *X* and *Y* have the same distribution, then *E(X) = E(Y)* and, more generally:

\[E(g(X)) = E(g(Y))\]

**Conditional Expected Value**  
Conditional expected value is defined like expectation, only conditioned on any event *A*:

\[E(X | A) = \sum\limits_{x}xP(X=x | A)\]

### Indicator Random Variables

**Indicator Random Variable**  
An indicator random variable is a random variable that takes on the value 1 or 0. It is always an indicator of some event: if the event occurs, the indicator is 1; otherwise, it is 0. They are useful for many problems about counting how many events of some kind occur. Write:

\[I_A =
 \begin{cases}
   1 & \text{if $A$ occurs,} \\
   0 & \text{if $A$ does not occur.}
  \end{cases}
\]

Note that:  
- \(I_A^2 = I_A\)
- \(I_A I_B = I_{A \cap B}\)
- \(I_{A \cup B} = I_A + I_B - I_A I_B\)

**Distribution**  
\(I_A \sim \Bern(p)\) where \(p = P(A)\).

**Fundamental Bridge**  
The expectation of the indicator for event *A* is the probability of event *A*: \(E(I_A) = P(A)\).

### Variance and Standard Deviation

\[\var(X) = E \left(X - E(X)\right)^2 = E(X^2) - (E(X))^2\]

\[\textrm{SD}(X) = \sqrt{\var(X)}\]

# Continuous RVs, LOTUS, UoU
---

## Continuous Random Variables (CRVs)

### What is a Continuous Random Variable (CRV)?

A continuous random variable can take on any possible value within a certain interval (for example, [0, 1]), whereas a discrete random variable can only take on variables in a list of countable values (for example, all the integers, or the values 1, 1/2, 1/4, 1/8, etc.)

### Do Continuous Random Variables have PMFs?

No. The probability that a continuous random variable takes on any specific value is 0.

### What's the probability that a CRV is in an interval?

Take the difference in CDF values (or use the PDF as described later).

\[P(a \leq X \leq b) = P(X \leq b) - P(X \leq a) = F_X(b) - F_X(a)\]

For X ~ N(μ, σ^2), this becomes

\[P(a \leq X \leq b) = \Phi\left(\frac{b-\mu}{\sigma}\right) - \Phi\left(\frac{a-\mu}{\sigma}\right)\]

### What is the Probability Density Function (PDF)?

The PDF f is the derivative of the CDF F.

\[F'(x) = f(x)\]

A PDF is nonnegative and integrates to 1. By the fundamental theorem of calculus, to get from PDF back to CDF we can integrate:

\[F(x) = \int_{-\infty}^x f(t)dt\]

To find the probability that a CRV takes on a value in an interval, integrate the PDF over that interval.

\[F(b) - F(a) = \int_a^b f(x)dx\]

### How do I find the expected value of a CRV?

Analogous to the discrete case, where you sum x times the PMF, for CRVs you integrate x times the PDF.

\[E(X) = \int_{-\infty}^\infty xf(x)dx\]

## LOTUS

### Expected value of a function of an r.v.

The expected value of X is defined this way:

\[E(X) = \sum_x xP(X=x) \textnormal{ (for discrete X)}\]
\[E(X) = \int_{-\infty}^\infty xf(x)dx \textnormal{ (for continuous X)}\]

The Law of the Unconscious Statistician (LOTUS) states that you can find the expected value of a function of a random variable, g(X), in a similar way, by replacing the x in front of the PMF/PDF by g(x) but still working with the PMF/PDF of X:

\[E(g(X)) = \sum_x g(x)P(X=x) \textnormal{ (for discrete X)}\]
\[E(g(X)) = \int_{-\infty}^\infty g(x)f(x)dx \textnormal{ (for continuous X)}\]

### What's a function of a random variable?

A function of a random variable is also a random variable. For example, if X is the number of bikes you see in an hour, then g(X) = 2X is the number of bike wheels you see in that hour, and h(X) = ${X \choose 2} = \frac{X(X-1)}{2}$ is the number of pairs of bikes such that you see both of those bikes in that hour.

### What's the point?

You don't need to know the PMF/PDF of g(X) to find its expected value. All you need is the PMF/PDF of X.

## Universality of Uniform (UoU)

When you plug any CRV into its own CDF, you get a Uniform(0,1) random variable. When you plug a Uniform(0,1) r.v. into an inverse CDF, you get an r.v. with that CDF. For example, let's say that a random variable X has CDF

\[F(x) = 1 - e^{-x}, \textrm{ for } x>0\]

By UoU, if we plug X into this function then we get a uniformly distributed random variable.

\[F(X) = 1 - e^{-X} \sim \textrm{Unif}(0,1)\]

Similarly, if U ~ Unif(0,1) then F^{-1}(U) has CDF F. The key point is that for any continuous random variable X, we can transform it into a Uniform random variable and back by using its CDF.

---

### Moments

Moments describe the shape of a distribution. Let X have mean μ and standard deviation σ, and Z=(X-μ)/σ be the *standardized* version of X. The kth moment of X is μₖ = E(Xᵏ), and the kth standardized moment of X is mₖ = E(Zᵏ). The mean, variance, skewness, and kurtosis are important summaries of the shape of a distribution.

- Mean: E(X) = μ₁
- Variance: var(X) = μ₂ - μ₁²
- Skewness: skew(X) = m₃
- Kurtosis: kurt(X) = m₄ - 3

### Moment Generating Functions

- MGF (Moment Generating Function): For any random variable X, the function Mₓ(t) = E(e^(tX)) is the moment generating function (MGF) of X, if it exists for all t in some open interval containing 0. The variable t could just as well have been called u or v. It's a bookkeeping device that lets us work with the function Mₓ rather than the sequence of moments.

- Why is it called the Moment Generating Function? Because the kth derivative of the moment generating function, evaluated at 0, is the kth moment of X. 
   μₖ = E(Xᵏ) = Mₓ⁽ᵏ⁾(0)
   This is true by Taylor expansion of e^(tX) since
   Mₓ(t) = E(e^(tX)) = ∑(k=0)^(∞) [E(Xᵏ)tᵏ/k!] = ∑(k=0)^(∞) [μₖtᵏ/k!]

- MGF of linear functions: If Y = aX + b, then Mₙ(t) = E(e^(t(aX + b))) = e^(bt)Mₓ(at)

- Uniqueness: If it exists, the MGF uniquely determines the distribution. This means that for any two random variables X and Y, they are distributed the same (their PMFs/PDFs are equal) if and only if their MGFs are equal.

- Summing Independent RVs by Multiplying MGFs: If X and Y are independent, then
  Mₓ₊ᵧ(t) = E(e^(t(X + Y))) = E(e^(tX))E(e^(tY)) = Mₓ(t) ⋅ Mᵧ(t)
  The MGF of the sum of two random variables is the product of the MGFs of those two random variables.

### Joint PDFs and CDFs

#### Joint Distributions
- Joint CDF: The joint cumulative distribution function (CDF) of X and Y is F(x,y) = P(X ≤ x, Y ≤ y).
- Joint PMF: In the discrete case, X and Y have a joint probability mass function (PMF) pₓᵧ(x,y) = P(X=x, Y=y).
- Joint PDF: In the continuous case, X and Y have a joint probability density function (PDF) fₓᵧ(x,y) = (∂²/∂x∂y)Fₓᵧ(x,y). The joint PMF/PDF must be nonnegative and sum/integrate to 1.

#### Conditional Distributions
- Conditioning and Bayes' rule for discrete random variables:
  P(Y=y|X=x) = P(X=x,Y=y) / P(X=x)
             = P(X=x|Y=y)P(Y=y) / ∑ᵧ P(X=x|Y=ᵧ)P(Y=ᵧ)
- Conditioning and Bayes' rule for continuous random variables:
  fᵧ|ₓ(y|x) = fₓᵧ(x, y) / fₓ(x)
             = (fₓ|ᵧ(x|y)fᵧ(y)) / fₓ(x)
- Hybrid Bayes' rule:
  fₓ(x|A) = (P(A | X = x)fₓ(x)) / P(A)

#### Marginal Distributions
To find the distribution of one (or more) random variables from a joint PMF/PDF, sum/integrate over the unwanted random variables.

- Marginal PMF from joint PMF:
  P(X = x) = ∑ₓ P(X=x, Y=y)
- Marginal PDF from joint PDF:
  fₓ(x) = ∫[∞, -∞] fₓᵧ(x, y) dy

#### Independence of Random Variables
Random variables X and Y are independent if and only if any of the following conditions holds:
- Joint CDF is the product of the marginal CDFs.
- Joint PMF/PDF is the product of the marginal PMFs/PDFs.
- Conditional distribution of Y given X is the marginal distribution of Y.
Write X ⫫ Y to denote that X and Y are independent.

#### Multivariate LOTUS
Law of the unconscious statistician (LOTUS) in more than one dimension is analogous to the 1D LOTUS.
For discrete random variables:
E(g(X, Y)) = ∑ₓ∑y g(x, y)P(X=x, Y=y)
For continuous random variables:
E(g(X, Y)) = ∫[-∞, ∞]∫[-∞, ∞] g(x, y)fₓᵧ(x, y)dxdy

\section{Covariance and Transformations}\smallskip \hrule height 2pt \smallskip

\subsection{Covariance and Correlation}
\begin{description}
\item [Covariance] is the analog of variance for two random variables.
    $$ \text{cov}(X, Y) = \mathbb{E}\left((X - \mathbb{E}(X))(Y - \mathbb{E}(Y))\right) = \mathbb{E}(XY) - \mathbb{E}(X)\mathbb{E}(Y) $$
    Note that 
    $$ \text{cov}(X, X) = \mathbb{E}(X^2) - (\mathbb{E}(X))^2 =  \text{var}(X) $$
\item [Correlation] is a standardized version of covariance that is always between $-1$ and $1$.
    $$ \text{corr}(X, Y) = \frac{\text{cov}(X, Y)}{\sqrt{\text{var}(X)\text{var}(Y)}} $$
\item [Covariance and Independence] If two random variables are independent, then they are uncorrelated. The converse is not necessarily true (e.g., consider $X \sim \mathcal{N}(0,1)$ and $Y=X^2$).
    \begin{align*}
    	X \independent Y &\longrightarrow \text{cov}(X, Y) = 0 \longrightarrow \mathbb{E}(XY) = \mathbb{E}(X)\mathbb{E}(Y)
    \end{align*}
%, except in the case of Multivariate Normal, where uncorrelated \emph{does} imply independence.
\item [Covariance and Variance]  The variance of a sum can be found by
    \begin{align*}
        %\text{cov}(X, X) &= \text{var}(X) \\
        \text{var}(X + Y) &= \text{var}(X) + \text{var}(Y) + 2\text{cov}(X, Y) \\
        \text{var}(X_1 + X_2 + \dots + X_n ) &= \sum_{i = 1}^{n}\text{var}(X_i) + 2\sum_{i < j} \text{cov}(X_i, X_j)
    \end{align*}
    If $X$ and $Y$ are independent then they have covariance $0$, so
    $$X \independent Y \Longrightarrow \text{var}(X + Y) = \text{var}(X) + \text{var}(Y)$$
    If $X_1, X_2, \dots, X_n$ are identically distributed and have the same covariance relationships (often by \textbf{symmetry}), then 
    $$\text{var}(X_1 + X_2 + \dots + X_n ) = n\text{var}(X_1) + 2{n \choose 2}\text{cov}(X_1, X_2)$$
\item [Covariance Properties]  For random variables $W, X, Y, Z$ and constants $a, b$:
    \begin{align*}
    	\text{cov}(X, Y) &= \text{cov}(Y, X) \\
        \text{cov}(X + a, Y + b) &= \text{cov}(X, Y) \\
        \text{cov}(aX, bY) &= ab\text{cov}(X, Y) \\
        \text{cov}(W + X, Y + Z) &= \text{cov}(W, Y) + \text{cov}(W, Z) + \text{cov}(X, Y)\\
        &\quad + \text{cov}(X, Z)
    \end{align*}
\item [Correlation is location-invariant and scale-invariant] For any constants $a,b,c,d$ with $a$ and $c$ nonzero,
    \begin{align*}
        \text{corr}(aX + b, cY + d) &= \text{corr}(X, Y) 
    \end{align*}
\end{description}



