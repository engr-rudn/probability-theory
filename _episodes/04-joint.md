---
title: "Joint, Marginal, and Conditional Probabilities"
teaching: 
exercises:
questions:
objectives:

keypoints:

---
# Joint, Marginal, and Conditional Probabilities


## Diagnostic accuracy

What if I told you that you just tested positive on a diagnostic test
with 95% accuracy, but that there's only a 16% chance that you have
the disease in question?  This running example will explain how this
can happen.^[The example diagnostic accuracies and disease prevalences
we will use for simulation are not unrealistic---this kind of thing
happens all the time in practice, which is why there are often
batteries of follow-up tests after preliminary positive test results.]

Suppose the random variable $$Z \in 0:1$$ represents whether a
particular subject has a specific disease.^[$$Z = 1$$ if the subject has
the disease and $$Z = 0$$ if not.]  Now suppose there is a diagnostic
test for whether a subject has the disease.  Further suppose the
random variable $$Y \in 0:1$$ represents the result of applying the test
to the same subject.^[$$Y = 1$$ if the test result is positive and $$Y =
0$$ if it's negative.]

Now suppose the test is 95% accurate in the sense that

* if the test is administered to a subject with the disease (i.e., $$Z
  = 1$$), there's a 95% chance the test result will be positive (i.e.,
  $$Y = 1$$), and
* if the test is administered to a subject without the disease (i.e.,
  $$Z = 0$$), there's a 95% chance the test result will be negative
  (i.e., $$Y = 0)$$.

We're going to pause the example while we introduce the probability
notation required to talk about it more precisely.
Conditional probability notation expresses the diagnostic test's
accuracy for people have the disease ($$Z = 1$$) as

$$
\mbox{Pr}[Y = 1 \mid Z = 1] = 0.95
$$\

and for people who don't have the disease ($$Z = 0$$) as

$$
\mbox{Pr}[Y = 0 \mid Z = 0] = 0.95.
$$\

We read $$\mbox{Pr}[\mathrm{A} \mid \mathrm{B}]$$ as the *conditional
probability* of event $$\mathrm{A}$$ given that we know event
$$\mathrm{B}$$ occurred.  Knowing that event $$\mathrm{B}$$ occurs often,
but not always, gives us information about the probability of
$$\mathrm{A}$$ occurring.

The conditional probability function $$\mbox{Pr}[\, \cdot \mid
\mathrm{B}]$$ behaves just like the unconditional probability function
$$\mbox{Pr}[\, \cdot \,]$$.  That is, it satisfies all of the laws of
probability we have introduced for event probabilities.  The
difference is in the semantics---conditional probabilities are
restricted to selecting a way the world can be that is consistent with
the event $$\mathrm{B}$$ occurring.^[Formally, an event $$\mathrm{B}$$ is
modeled as a set of ways the world can be, i.e., a subset $$\mathrm{B}
\subseteq \Omega$$ of the sample space $$\Omega$$.  The conditional
probability function $$\mbox{Pr}[\, \cdot \mid \mathrm{B}]$$ can be
interpreted as an ordinary probability distribution with sample space
$$\mathrm{B}$$ instead of the original $$\Omega$$.]

For example, the sum of exclusive and exhaustive probabilities must be
one, and thus the diagnostic error probabilities are one minus the
correct diagnosis probabilities,^[These error rates are often called the
false positive rate and false negative rate, the positive and negative
in this case being the test result and the false coming from not
matching the true disease status.]

$$
\mbox{Pr}[Y = 0 \mid Z = 1] = 0.05
$$\

and

$$
\mbox{Pr}[Y = 1 \mid Z = 0] = 0.05.
$$\

## Conditional probability

**Definition:** Events A, B are independent if $$P(A\cap B) = P(A)P(B)$$ 

Note: completely different form disjointness

$$A, B, C$$ are independent, if $$P(A, B) = P(A)P(B)$$, $$P(A,C) = P(A)P(C)$$, $$P(B,C) = P(B)P(C)$$ and $$P(A, B, C) = P(A)P(B)P(C)$$

Similarly for events A1,…An

---
> ## Newton-Pepys Problem(1693)
> 
>  The **Newton–Pepys problem** is a [probability](https://en.wikipedia.org/wiki/Probability) problem concerning the probability of throwing sixes from a certain number of dice.
> 
> In 1693 [Samuel Pepys](https://en.wikipedia.org/wiki/Samuel_Pepys) and [Isaac Newton](https://en.wikipedia.org/wiki/Isaac_Newton) corresponded over a problem posed by Pepys in relation to a [wager](https://en.wikipedia.org/wiki/Gambling) he planned to make. The problem was:
>
> 
> *A. Six fair dice are tossed independently and at least one “6” appears.*
>
> *B. Twelve fair dice are tossed independently and at least two “6”s appear.*
>
> *C. Eighteen fair dice are tossed independently and at least three “6”s appear.*
> >
> > ## Solution
> > Pepys initially thought that outcome C had the highest probability, but Newton correctly concluded that outcome A actually has the highest probability.
> > Quoted from Wikipedia : [Newton–Pepys problem](https://en.wikipedia.org/wiki/Newton%E2%80%93Pepys_problem)\
> > $$P(A) = 1 - (5/6)^6 \approx 0.665$$\
> ><br> 
> > $$P(B) = 1 - (5/6)^{12} - 12 *(1/6)(5/6)^{11}  \approx  0.619$$\
> ><br> 
> > $$P(C) = 1 - \sum_{k=0}^2 \binom{18}{k} (1/6)^k (5/6)^{(18-k)}  \approx 0.597$$
>{: .solution}
{: .challenge} 

> ## Challenge - — How should you update probability/beliefs/uncertainty based on new evidence?
> "Conditioning is the soul of statistic" 
{: .challenge}

### Conditional Probability
#### Definition:

$$
P(A|B) = \frac{P(A\cap B)} {P(B)}
$$, if $$
P(B) > 0
$$

#### Intuition:

1. Pebble world , there are finite possible outcomes, each one is represented as a pebble. For example, 9 outcomes, that is 9 pebbles , total mass is 1. B: four pebbles, $$
P(A|B)
$$:get rid of pebbles in $$B^c$$ , renormalize to make total mass again

2. Frequentist world: repeat experiment many times

   (100101101) 001001011 11111111 

    circle repeatitions where B occurred ; among those , what fraction of time did A also occur?

### Theorem

1. $$
P(A\cap B) = P(B)P(A|B) = P(A)P(B|A)
$$\
2. $$
P(A_1…A_n) = P(A_1)P(A_2|A_1)P(A_3|A_1,A_2)…P(A_n|A_1,A_2…A_{n-1})
$$\
3. $$
P(A|B) = \frac{P(B|A)P(A)}{P(B)}
$$\ 

**Thinking** conditionally is a condition for thinking

**How to solve a problem?**

1. Try simple and extreme cases

2. Break up problem into simple pieces 

   $$P(B) = P(B|A_1)P(A_1) + P(B|A_2)P(A_2) +…P(B|A_n)P(A_n)$$

   law of total probability 

> ### Example 1- Suppose we have 2 random cards from standard deck
> Find $$P(both\ aces|have\ ace), P(both\ aces|have\ ace\ of\ spade)$$
> >
> > ## Solution
> >
> > $$P
(\text{both aces}|\text{have ace}) = \frac{P(\text{both aces}, \text{have ace})}{P(\text{have ace})} = \frac{\binom{4}{2}/\binom{52}{2}}{1-\binom{48}{2}/\binom{52}{2}} = \frac{1}{33}
$$\
> > $$P
(both\ aces|have\ ace\ of\ spade) = 3/51 = 1/17
$$
> {: .solution}
{: .challenge}
> ### Example 2 - Patient get tested for disease afflicts 1% of population, tests positive (has disease)
> Suppose the test is advertised as "95% accurate" , suppose this means 
> $$D$$: has disease, $$T$$: test positive
> 
> > ## Solution
> >
> > Trade-off: It's rare that the test is wrong, it's also rare the disease is rare
> > $$$$\
> > $$P(T|D) = 0.95 = P(T^c |D^c)$$
> > $$$$\
> > $$P(D|T) = \frac{P(T|D)P(D)}{P(T)} = \frac{P(T|D)P(D)}{P(T|D)P(D) + P(T|D^c)P(D^c}{}$$
> {: .solution}
{: .challenge}



### Biohazards

1. confusing $$
P(A|B)
$$, $$
P(B|A)
$$  ([procecutor's fallacy](https://en.wikipedia.org/wiki/Prosecutor%27s_fallacy)) 

<u>Ex</u> [Sally Clark](https://en.wikipedia.org/wiki/Sally_Clark) case, SIDS

want $$
P(innocence|evidence)
$$ 

1. confusing $$
P(A) - prior 
$$ with $$
P(A|B)
$$ - posterior

$$P(A|A) = 1$$

1. confusing independent with conditional independent 



**Definition**:

 Events $$A,B$$ are conditionally independent given $$C$$ if 
 	$$P(A\cap B|C)$$ = $$P(A|C)P(B|C)$$

> ### Does conditional independence given C imply independence ? 
> 
> > ## Solution
> > No
> > $$$$\
> > Example - Chess opponent of unknown strength may be that game outcomes are conditionally independent given strength
> {: .solution}
{: .challenge}

 

> ### Does independent imply conditional independent given C?
> 
> > ## Solution
> > No
> > $$$$\
> > Example - A: Fire alarm goes off, cause by : F:fire; C:popcorn. suppose F, C independent But 
> > $$$$\
> > $$P(F|A, C^c^) = 1$$ not conditionally independent given A
> {: .solution}
{: .challenge}

## Joint probability

Joint probability notation expresses the probability of multiple
events happening.  For instance, $$\mbox{Pr}[Y = 1, Z = 1]$$ is the
probability of both events, $$Y = 1$$ and $$Z = 1$$, occurring.

In general, if $$\mathrm{A}$$ and $$\mathrm{B}$$ are events,^[We use
capital roman letters for event variables to distinguish them from
random variables for which we use capital italic letters.] we write
$$\mbox{Pr}[\mathrm{A}, \mathrm{B}]$$ for the event of both $$\mathrm{A}$$
and $$\mathrm{B}$$ occurring.  Because joint probability is defined in
terms of conjunction ("and" in English), the definition is symmetric,
$$\mbox{Pr}[\mathrm{A}, \mathrm{B}] = \mbox{Pr}[\mathrm{B},
\mathrm{A}].$$

In the context of a joint probability $$\mbox{Pr}[\mathrm{A}, \mathrm{B}]$$, the single
event $$\mbox{Pr}[\mathrm{A}]$$ is called a *marginal probability*.^[Because of
symmetry, $$\mbox{Pr}[\mathrm{B}]$$ is also a marginal probability.]

Joint probability is defined relative to conditional and marginal
probabilities by

$$
\mbox{Pr}[\mathrm{A}, \mathrm{B}]
\ = \
\mbox{Pr}[\mathrm{A} \mid \mathrm{B}] \times \mbox{Pr}[\mathrm{B}].
$$

In words, the probability of events $$\mathrm{A}$$ and $$\mathrm{B}$$ occurring is the same
as the probability of event $$\mathrm{B}$$ occurring times the probability of $$\mathrm{A}$$
occurring given that $$\mathrm{B}$$ occurs.

The relation between joint and conditional probability involves simple
multiplication, which may be rearranged by dividing both sides by
$$\mbox{Pr}[\mathrm{B}]$$ to yield

$$
\mbox{Pr}[\mathrm{A} \mid \mathrm{B}]
\ = \
\frac{\displaystyle \mbox{Pr}[\mathrm{A}, \mathrm{B}]}
     {\displaystyle \mbox{Pr}[\mathrm{B}]}.
$$

Theoretically, it is simplest to take joint probability as the
primitive so that this becomes the definition of conditional
probability.  In practice, all that matters is the relation between
conditional and joint probability.

## Dividing by zero: not-a-number or infinity

Conditional probabilities are not well defined in the situation where
$$\mbox{Pr}[B] = 0$$.  In such a situation, $$\mbox{Pr}[\mathrm{A},
\mathrm{B}] = 0$$, because $$B$$ has probability zero.

In practice, if we try to evaluate such a condition using standard
computer floating-point arithmetic, we wind up dividing zero by zero.

```
Pr_A_and_B = 0.0
Pr_B = 0.0
Pr_A_given_B = Pr_A_and_B / Pr_B
print('Pr[A | B] = ', Pr_A_given_B)
```
{: .language-python}

Running that, we get

```
ZeroDivisionError: float division by zero
```
{: .output}

The value `NaN` indicates what is known as the *not-a-number*
value.^[There are technically two types of not-a-number values in the
IEEE 754 standard, of which we will only consider the non-signaling
version.]  This is a special floating-point value that is different
from all other values and used to indicate a domain error on an
operation.  Other examples that will return not-a-number values
include `log(-1)`.

If we instead divide a positive or negative number by zero,

```
print('1.0 / 0.0 = ', 1.0 / 0.0)
print('-3.2 / 0.0 = ', -3.2 / 0.0)
```
{: .language-r}
we get

```
1.0 / 0.0 = Inf
-3.2 / 0.0 = -Inf
```
{: .output}

These values denote positive infinity ($$\infty$$) and negative infinity
($$-\infty$$), respectively.  Like not-a-number, these are special
reserved floating-point values with the expected interactions with
other numbers.^[For example, adding a finite number and an infinity
yields the infinity and subtracting two infinities produces a
not-a-number value.  For details, see [754-2008---IEEE standard for floating-point
arithmetic](https://ieeexplore.ieee.org/document/4610935).]

## Simulating the diagnostic example

What we don't know so far in the running example is what the
probability is of a subject having the disease or not.  This
probability, known as the *prevalence* of the disease, is often the
main quantity of interest in an epidemiological study.

Let's suppose in our case that 1% of the population has the disease in
question.  That is, we assume

$$
\mbox{Pr}[Z = 1] = 0.01
$$

Now we have enough information to run a simulation of all
of the joint probabilities.  We just follow the marginal and
conditional probabilities.  First, we generate $$Z$$, whether or not the
subject has the disease, then we generate $$Y$$, the result of the test,
conditional on the disease status $$Z$$.

```
import numpy as np

M = 1000000
z = np.zeros(M)
y = np.zeros(M)

for m in range(M):
    z[m] = np.random.binomial(1, 0.01)
    if z[m] == 1:
        y[m] = np.random.binomial(1, 0.95)
    else:
        y[m] = np.random.binomial(1, 0.05)

print('estimated Pr[Y = 1] = ', np.sum(y) / M)
print('estimated Pr[Z = 1] = ', np.sum(z) / M)

```
{: .language-python}

```
estimated Pr[Y = 1] =  0.058812
estimated Pr[Z = 1] =  0.009844
```
{: .output}

The program computes the marginals for $$Y$$ and $$Z$$ directly.  This is
straightforward because both $$Y$$ and $$Z$$ are simulated in every
iteration (as `y[m]` and `z[m]` in the code).  Marginalization using
simulation requires no work whatsoever.^[Marginalization can be
tedious, impractical, or impossible to carry out analytically.]

Let's run that with $$M = 100,000$$ and see what we get.

```
import numpy as np

np.random.seed(1234)
M = 100000
z = np.random.binomial(1, 0.01, M)
y = np.where(z == 1, np.random.binomial(1, 0.95, M), np.random.binomial(1, 0.05, M))
print('estimated Pr[Y = 1] = ', np.sum(y) / M)
print('estimated Pr[Z = 1] = ', np.sum(z) / M)
```
{: .language-python}

```
estimated Pr[Y = 1] =  0.05755
estimated Pr[Z = 1] =  0.01008
```
{: .output}

We know that the marginal $$\mbox{Pr}[Z = 1]$$ is 0.01, so the estimate is
close to the true value for $$Z$$;  we'll see below that it's also close
to the true value of $$\mbox{Pr}[Y = 1]$$.

We can also use the simulated values to estimate conditional
probabilities.  To estimate, we just follow the formula for the
conditional distribution,

$$
\mbox{Pr}[A \mid B]
\ = \
\frac{\displaystyle \mbox{Pr}[A, B]}
     {\displaystyle \mbox{Pr}[B]}.
$$

Specifically, we count the number of draws in which both A and B
occur, then divide by the number of draws in which the event B occurs.

```
import numpy as np

np.random.seed(1234)
M = 100000
z = np.random.binomial(1, 0.01, M)
y = np.where(z == 1, np.random.binomial(1, 0.95, M), np.random.binomial(1, 0.05, M))
y1z1 = np.logical_and(y == 1, z == 1)
y1z0 = np.logical_and(y == 1, z == 0)

print('estimated Pr[Y = 1 | Z = 1] = ', np.sum(y1z1) / np.sum(z == 1))
print('estimated Pr[Y = 1 | Z = 0] = ', np.sum(y1z0) / np.sum(z == 0))
```
{: .language-python}

We set the indicator variable `y1z1[m]` to 1 if $$Y = 1$$ and $$Z = 1$$ in
the $$m$$-th simulation; `y1z0` behaves similarly.  The operator `&` is
used for conjuncton (logical and) in the usual way.^[The logical and
operation is often written as `&&`  in programming languages to
distinguish it from bitwise and, which is conventionally written `&`.]

Recall that `z == 0`
is an array where entry $$m$$ is 1 if the condition holds, here $$z^{(m)}
= 0$$.

The resulting estimates with $$M = 100,000$$ draws are pretty close to
the true values,

```
estimated Pr[Y = 1 | Z = 1] =  0.9454365079365079
estimated Pr[Y = 1 | Z = 0] =  0.048508970421852274
```
{: .output}

The true values were stipulated as part of the example to be
$$\mbox{Pr}[Y = 1 \mid Z = 1] = 0.95$$ and $$\mbox{Pr}[Y = 1 \mid Z = 0]
= 0.05$$.  Not surprisingly, knowing the value of $$Z$$ (the disease
status) provides quite a bit of information about the value of $$Y$$
(the diagnostic test result).

Now let's do the same thing the other way around,and look at the
probability of having the disease based on the test result, i.e.,
$$\mbox{Pr}[Z = 1 \mid Y = 1]$$ and $$\mbox{Pr}[Z = 1 \mid Y = 0]$$.^[The
program is just like the last one.]

```
print('estimated Pr[Z = 1 | Y = 1] = {:.3f}'.format(sum(y * z) / sum(y)))
print('estimated Pr[Z = 1 | Y = 0] = {:.3f}'.format(sum(z * (1 - y)) / sum(1 - y)))

```
{: .language-python}


```
estimated Pr[Z = 1 | Y = 1] = 0.166
estimated Pr[Z = 1 | Y = 0] = 0.001
```
{: .output}

Did we make a mistake in coding up our simulation?  We estimated
$$\mbox{Pr}[Z = 1 \mid Y = 1]$$ at around 16%, which says that if the
subject has a positive test result ($$Y = 1$$), there is only a 16%
chance they have the disease ($$Z = 1$$).  How can that be when the test
is 95% accurate and simulated as such?

The answer hinges on the prevalence of the disease.  We assumed only
1% of the population suffered from the disease, so that 99% of the
people being tested were disease free.  Among the disease free ($$Z =
0$$) that are tested, 5% of them have false positives ($$Y = 1$$).
That's a lot of patients, around 5% times 99%, which is nearly 5% of
the population.  On the other hand, among the 1% of the population
with the disease, almost all of them test positive.  This means
roughly five times as many disease-free subjects test positive for the
disease as disease-carrying subjects test positive.


## Analyzing the diagnostic example

The same thing can be done with algebra as we did in the previous
section with simulations.^[Computationally, precomputed algebra is a
big win over simulations in terms of both compute time and accuracy.
It may not be a win when the derivations get tricky and human time is
taken into consideration.]  Now we can evaluate joint probabilities,
e.g.,

$$
\begin{array}{rcl}
\mbox{Pr}[Y = 1, Z = 1]
& = & \mbox{Pr}[Y = 1 \mid Z = 1] \times \mbox{Pr}[Z = 1]
\\[4pt]
& = & 0.95 \times 0.01
\\
& = & 0.0095.
\end{array}
$$

Similarly, we can work out the probability the remaining probabilities
in the same way, for example, the probability of a subject having the
disease and getting a negative test result,

$$
\begin{array}{rcl}
\mbox{Pr}[Y = 0, Z = 1]
& = & \mbox{Pr}[Y = 0 \mid Z = 1] \times \mbox{Pr}[Z = 1]
\\[4pt]
& = & 0.05 \times 0.01
\\
& = & 0.0005.
\end{array}
$$

Doing the same thing for the disease-free patients completes a
four-by-four table of probabilities,

$$
\begin{array}{|r|r|r|}
\hline
\mbox{Probability} & Y = 1  & Y = 0
\\ \hline
Z = 1 & 0.0095 & 0.0005
\\ \hline
Z = 0 & 0.0450 & 0.8500
\\ \hline
\end{array}
$$

For example, the top-left entry records the fact that $$\mbox{Pr}[Y =
1, Z = 1] = 0.0095.$$ The next entry to the right indicates that
$$\mbox{Pr}[Y = 0, Z = 1] = 0.0005.$$

The marginal probabilities (e.g., $$\mbox{Pr}[Y = 1]$$) can be computed
by summing the probabilities of all alternatives that lead to $$Y = 1$$,
here $$Z = 1$$ and $$Z = 0$$

$$
\mbox{Pr}[Y = 1]
\ = \
\mbox{Pr}[Y = 1, Z = 1]
+ \mbox{Pr}[Y = 1, Z = 0]
$$

We can extend our two-by-two table by writing the sums in what
would've been the margins of the original table above.

$$
\begin{array}{|r|r|r|r|}
\hline
\mbox{Probability} & Y = 1  & Y = 0 & Y = 1 \ \mbox{or} \ Y = 0
\\ \hline
Z = 1 & 0.0095 & 0.0005 & \mathit{0.0100}
\\ \hline
Z = 0 & 0.0450 & 0.8500 & \mathit{0.9900}
\\ \hline
Z = 1 \ \mbox{or} \ Z = 0 & \mathit{0.0545} & \mathit{0.8505}
& \mathbf{1.0000}
\\ \hline
\end{array}
$$

Here's the same table with the symbolic values.

$$
\begin{array}{|r|r|r|r|}
\hline
\mbox{Probability} & Y = 1  & Y = 0 & Y = 1 \ \mbox{or} \ Y = 0
\\ \hline
Z = 1 & \mbox{Pr}[Y = 1, Z = 1] & \mbox{Pr}[Y = 0, Z = 1]
& \mbox{Pr}[Z = 1]
\\ \hline
Z = 0 & \mbox{Pr}[Y = 1, Z = 0] & \mbox{Pr}[Y = 0, Z = 0]
& \mbox{Pr}[Z = 0]
\\ \hline
Z = 1 \ \mbox{or} \ Z = 0
& \mbox{Pr}[Y = 1] & \mbox{Pr}[Y = 0]
& \mbox{Pr}[ \ ]
\\ \hline
\end{array}
$$

For example, that $$\mbox{Pr}[Z = 1] = 0.01$$ can be read off the top of
the right margin column---it is the sum of the two table entries in
the top row, $$\mbox{Pr}[Y = 1, Z = 1]$$ and $$\mbox{Pr}[Y = 0, Z = 1]$$.

In the same way, $$\mbox{Pr}[Y = 0] = 0.8505$$ can be read off the right
of the bottom margin row, being the sum of the entries in the right
column, $$\mbox{Pr}[Y = 0, Z = 1]$$ and $$\mbox{Pr}[Y = 0, Z = 0]$$.

The extra headings define the table so that
each entry is the probability of the event on the top row and on the
left column.  This is why it makes sense to record the grand sum of
1.00 in the bottom right of the table.


## Joint and conditional distribution notation

Recall that the probability mass function $$p_Y(y)$$ for a discrete
random variable $$Y$$ is defined by

$$
p_Y(y) = \mbox{Pr}[Y = y].
$$

As before, capital $$Y$$ is the random variable, $$y$$ is a potential
value for $$Y$$, and $$Y = y$$ is the event that the random variable $$Y$$
takes on value $$y$$.

The *joint probability mass function* for two discrete random variables
$$Y$$ and $$Z$$ is defined by the joint probability,

$$
p_{Y,Z}(y, z) = \mbox{Pr}[Y = y, Z = z].
$$

The notation follows the previous notation with $$Y, Z$$ indicating that
the first argument is the value of $$Y$$ and the second that of $$Z$$.


Similarly, the conditional probablity mass function is defined by

$$
p_{Y \mid Z}(y \mid z) = \mbox{Pr}[Y = y \mid Z = z].
$$

It can equivalently be defined as

$$
p_{Y \mid Z}(y \mid z)
\ = \
\frac{\displaystyle p_{Y, Z}(y, z)}
     {\displaystyle p_{Z}(z)}.
$$

The notation again follows that of the conditional probability
function through which the conditional probability mass function is
defined.


## Bayes's Rule

Bayes's rule relates the conditional probability $$\mbox{Pr}[\mathrm{A} \mid \mathrm{B}]$$
for events $$\mathrm{A}$$ and $$\mathrm{B}$$ to the inverse conditional probability
$$\mbox{Pr}[\mathrm{A} \mid \mathrm{B}]$$ and the marginal probability $$\mbox{Pr}[\mathrm{B}]$$.
The rule requires a partition of the event $$\mathrm{A}$$ into events $$\mathrm{A}_1,
\ldots, \mathrm{A}_K$$, which are mutually exclusive and exhaust $$\mathrm{A}$$.  That is,

$$
\mbox{Pr}[\mathrm{A}_k, \mathrm{A}_{k'}] = 0
\ \ \mbox{if} \ \ k \neq k',
$$

and

$$
\begin{array}{rcl}
\mbox{Pr}[\mathrm{A}]
& = & \mbox{Pr}[\mathrm{A}_1] + \cdots + \mbox{Pr}[\mathrm{A}_K].
\\[3pt]
& = & \sum_{k \in 1:K} \mbox{Pr}[\mathrm{A}_k].
\end{array}
$$

The basic rule of probability used to derive each line is noted to the
right.

$$
\begin{array}{rcll}
\mbox{Pr}[\mathrm{A} \mid \mathrm{B}]
& = &
\frac{\displaystyle \mbox{Pr}[\mathrm{A}, \mathrm{B}]}
     {\displaystyle \mbox{Pr}[\mathrm{B}]}
& \ \ \ \ \mbox{[conditional definition]}
\\[6pt]
& = &
\frac{\displaystyle \mbox{Pr}[\mathrm{B} \mid \mathrm{A}] \times \mbox{Pr}[\mathrm{A}]}
     {\displaystyle \mbox{Pr}[\mathrm{B}]}
& \ \ \ \ \mbox{[joint definition]}
\\[6pt]
& = &
\frac{\displaystyle \mbox{Pr}[\mathrm{B} \mid \mathrm{A}] \times \mbox{Pr}[\mathrm{A}]}
     {\displaystyle \sum_{k \in 1:K} \displaystyle \mbox{Pr}[\mathrm{B}, \mathrm{A}_k]}
& \ \ \ \ \mbox{[marginalization]}
\\[6pt]
& = &
\frac{\displaystyle \mbox{Pr}[\mathrm{B} \mid \mathrm{A}] \times \mbox{Pr}[\mathrm{A}]}
     {\displaystyle \sum_{k \in 1:K} \mbox{Pr}[\mathrm{B} \mid \mathrm{A}_k] \times \mbox{Pr}[\mathrm{A}_k]}.
& \ \ \ \ \mbox{[joint definition]}
\end{array}
$$

Letting $$\mathrm{A}$$ be the event $$Y = y$$, $$\mathrm{B}$$ be the event
$$Z = z$$, and $$A_k$$ be the event $$Y = k$$, Bayes's rule can be
instantiated to

$$
\mbox{Pr}[Y = y \mid Z = z]
\ = \
\frac{\displaystyle
        \mbox{Pr}[Z = z \mid Y = y] \times \mbox{Pr}[Y = y]}
     {\displaystyle
       \sum_{y' \in 1:K} \mbox{Pr}[Z = z \mid Y = y'] \times \mbox{Pr}[Y = y']}.
$$

This allows us to express Bayes's rule in terms of probability mass
functions as

$$
p_{Y \mid Z}(y \mid z)
\ = \
\frac{\displaystyle p_{Z \mid Y}(z \mid y) \times p_{Y}(y)}
     {\displaystyle \sum_{y' \in 1:K} p_{Z \mid Y}(z \mid y') \times p_Y(y')}.
$$

Bayes's rule can be extended to infinite partitions of the event $$B$$,
or in the probability mass function case, a variable $$Y$$ taking on
infinitely many possible values.


## Fermat and the problem of points

Blaise Pascal and Pierre de Fermat studied the problem of how to
divide the pot^[The *pot* is the total amount bet by both players.] of
a game of chance that was interrupted before it was finished.  As a
simple example, Pascal and Fermat considered a game in which each turn
a fair coin was flipped, and the first player would score a win a
point if the result was heads and the second player if the result was
tails.  The first player to score ten points wins the game.

Now suppose a game is interrupted after 15 flips, at a point where the
first player has 8 points and the second player only 7.  What is the
probability of the first player winning the match were it to continue?

We can put that into probability notation by letting $$Y_n$$ be the
number of points for the first player after $$n$$ turns.

$$Y_{n, 1}$$ be the
number of heads for player 1 after $$n$$ flips, $$Y_{n, 2}$$ be the same
for player 2.  Let $$Z$$ be a binary random variable taking value 1 if
the first player wins and 0 if the other player wins.  Fermat managed
to evaluate a formula like Fermat evaluated $$\mbox{Pr}[Z = 1 \mid Y_{n,
1} = 8, Y_{n, 2} = 7]$$ by enumerating the possible game continuations
and adding up the probabilities of the ones in which the first player
wins.

We can solve Fermat and Pascal's problem by simulation.  As usual, our
estimate is just the proportion of the simulations in which the first
player wins.  The value of `pts` must be given as input---that is the
starting point for simulating the completion of the game, assuming
neither player yet has ten points.^[For illustrative purposes only!
In robust code, validation should produce diagnostic error messages
for invalid inputs.]

```
import numpy as np

np.random.seed(1234)
M = 100000
win = np.zeros(M)
for m in range(M):
    pts = [0, 0]
    while pts[0] < 10 and pts[1] < 10:
        toss = np.random.uniform()
        if toss < 0.5:
            pts[0] += 1
        else:
            pts[1] += 1
    if pts[0] == 10:
        win[m] = 1
    else:
        win[m] = 0

print('est. Pr[player 1 wins] =', np.mean(win))

```
{: .language-python}

```
est. Pr[player 1 wins] = 0.50181
```
{: .output}

If the while-loop terminates because one player has ten points, then
`wins[m]` must have been set in the previous value of the loop.^[In
general, programs should be double-checked (ideally by a third party)
to make sure *invariants* like this one (i.e., `win[m]` is always set)
actually hold.  Test code goes a long way to ensuring robustness.]

Let's run that a few times with $$M = 100,000$$, starting with the
`pts` set to `(8, 7)`, to simulate Fermat and Pascal's problem.

```
import numpy as np

np.random.seed(1234)

for k in range(1, 6):
    M = 100000
    game_wins = 0
    
    for m in range(M):
        wins = [8, 7]
        
        while wins[0] < 10 and wins[1] < 10:
            toss = np.random.binomial(1, 0.5)
            winner = 1 if toss == 1 else 2
            wins[winner - 1] += 1
            
            if wins[0] == 10:
                game_wins += 1
    
    print(f'est. Pr[player 1 wins] = {game_wins / M:.3f}')

```
{: .language-python}

```
est. Pr[player 1 wins] = 0.686
est. Pr[player 1 wins] = 0.687
est. Pr[player 1 wins] = 0.688
est. Pr[player 1 wins] = 0.687
est. Pr[player 1 wins] = 0.686
```
{: .output}


This is very much in line with the result Fermat derived by brute
force, namely $$\frac{11}{16} \approx 0.688.$$^[There are at most four
more turns required, which have a total of $$2^4 = 16$$ possible
outcomes, HHHH, HHHT, HHTH, $$\ldots,$$ TTTH, TTTT, of which 11 produce
wins for the first player.]


## Independence of random variables

Informally, we say that a pair of random variables is independent if
knowing about one variable does not provide any information about the
other.  If $$X$$ and $$Y$$ are the variables in question, this property
can be stated directly in terms of their probability mass functions as

$$
p_{X}(x) = p_{X|Y}(x \mid y).
$$

In practice, we use an equivalent definition.  Random variables $$X$$
and $$Y$$ are said to be *independent* if

$$
p_{X,Y}(x, y) = p_X(x) \times p_Y(y).
$$

for all $$x$$ and $$y$$.^[This is equivalent to requiring the events $$X
\leq x$$ and $$Y \leq y$$ to be independent for every $$x$$ and $$y$$.
Events A and B are said to be *independent* if $$\mbox{Pr}[\mathrm{A},
\mathrm{B}] \ = \ \mbox{Pr}[\mathrm{A}] \times
\mbox{Pr}[\mathrm{B}]$$.]

By way of example, we have been assuming that a fair dice throw
involves the throw of two independent and fair dice.  That is, if
$$Y_1$$ is the first die and $$Y_2$$ is the second die, then $$Y_1$$ is
independent of $$Y_2$$.

In the diagnostic testing example, the disease state $$Z$$ and the test
result $$Y$$ are not independent.^[That would be a very poor test,
indeed!].  This can easily be verified because $$p_{Y|Z}(y \mid z) \neq
p_Y(y)$$.



## Independence of multiple random variables

It would be nice to be able to say that a set of random $$Y_1, \ldots,
Y_N$$ was independent if each of its pairs of random variables was
independent.  We'd settle for being able to say that the joint
probability factors into the product of marginals,

$$
p_{Y_1, \ldots, Y_N}(y_1, \ldots, y_N)
\ = \
p_{Y_1}(y1) \times \cdots \times p_{Y_N}(y_N).
$$

But neither of these is enough.^[Analysis in general and probability
theory in particular defeat simple definitions with nefarious edge
cases.]  For a set of random variables to be *independent*, the
probability of each of its subsets must factor into the product of its
marginals.^[More precisely, $$Y_1, \ldots, Y_N$$ are *independent* if
for every $$M \leq N$$ and permutation $$\pi$$ of $$1:N$$ (i.e., a bijection
between $$1:N$$ and itself), we have $$\begin{array}{l}
\displaystyle p_{Y_{\pi(1)},
\ldots, Y_{\pi(M)}}(u_1, \ldots, u_M)
\\ \displaystyle \mbox{ } \ \ \ = \
p_{Y_{\pi(1)}}(u_1) \times \cdots \times p_{Y_{\pi(M)}}(u_M)
\end{array}$$
for all $$u_1, \ldots, u_M.$$]

## Conditional independence

Often, a pair of variables are not independent only because they both
depend on a third variable.  The random variables $$Y_1$$ and $$Y_2$$ are
said to be *conditionally independent* given the variable $$Z$$ if they
are independent after conditioning,

$$
p_{Y_1, Y_2 \mid Z}(y_1, y_2 \mid z)
\ = \
p_{Y_1 \mid Z}(y_1 \mid z) \times p_{Y_2 \mid Z}(y_2 \mid z).
$$

## Conditional expectations

The expectation $$\mathbb{E}[Y]$$ of a random variable $$Y$$ is its
average value (weighted by density or mass, depending on whether it is
continuous or discrete).  The conditional expectation $$\mathbb{E}[Y
\mid A]$$ given some event $$A$$ is defined to be the average value of $$Y$$
conditioned on the event $$A$$,

$$
\mathbb{E}[Y \mid A]
\ = \
\int_Y y \times p_{Y \mid A}(y \mid A) \, \mathrm{d} y,
$$

where $$p_{Y \mid A}$$ is the density of $$Y$$ conditioned on event $$A$$
occurring.  This conditional density $$p_{Y \mid A}$$ is defined just
like the ordinary density $$p_Y$$ only with the conditional cumulative
distribution function $$F_{Y \mid A}$$ instead of the standard
cumulative distribution function $$F_Y$$,

$$
p_{Y \mid A}(y \mid A)
\ = \
\frac{\mathrm{d}}{\mathrm{d} y} F_{Y \mid A}(y \mid A).
$$

The conditional cumulative distribution function $$F_{Y \mid A}$$
is, in turn, defined by the conditioning on the event probability,

$$
F_{Y \mid A}(y \mid A)
\ = \
\mbox{Pr}[Y < y \mid A].
$$

This also works to condition on zero probability events, such as
$$\Theta = \theta$$, by taking the usual definition of conditional
density,

$$
\mathbb{E}[Y \mid \Theta = \theta]
\ = \
\int_Y y \times p_{Y \mid \Theta}(y \mid \theta) \, \mathrm{d}y.
$$

When using discrete variables, integrals are replaced with sums.


## Independent and identically distributed variables

If the variables $$Y_1, \ldots, Y_N$$ are not only independent, but also
have the same probability mass functions (i.e., $$p_{Y_n} = p_{Y_{m}}$$
for all $$m, n \in 1:N$$), we say that they are *independent and
identically distributed*, or "iid" for short.  Many of our statistical
models, such as linear regression, will make the assumption that
observations are conditionally independent and identically
distributed.
