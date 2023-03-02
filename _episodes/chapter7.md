---
title: "Questions and Answers-7"
teaching: 0
exercises: 0
questions:
objectives:
keypoints:

---

# Markov Chains

## Important/Useful Theorems

### 

For a Markox transition matrix, the limiting probabilities of being in a
certain state as $$n \rightarrow \infty$$ are given by the solution to the
following set of linear equations.
$$p_j^* = \sum_{i=1}^{\infty} p_i^* p_{ij}$$

## Answers to Problems

### 

Since state $$i$$ signifies that the highest number chosen so far is $$i$$,
what is the probability the next number is lower than $$i$$ and you stay
in state $$i$$? Well, there are $$m$$ possible numbers that could get picked
and $$i$$ of them are less than or equal to $$i$$ giving:

$$p_{ii} = \frac{i}{m}$$

What now if we're in $$i$$ and we want to know what the probability of
being in state $$j$$ is next. Well, if $$j<i$$ then it's zero because you
can't go to a lower number in this game. If, however, $$j$$ is not lower
then there are $$m$$ possible numbers that could get called and only one
of them is $$j$$, giving: $$p_{ij} = \frac{1}{m}, j>i$$
$$p_{ij} = 0, j<i$$

**Answer verified**

### 

Since the chain has an inevitable endpoint that you cannot get out of,
there is a persistent state at $$i=m$$ while all other states are
transient. Transient, in that you will only see them a few times but $$m$$
is bound to show up sooner or later and once you're in $$m$$ you can't get
out no matter what.

**Answer not verified**

### 

To do this properly we need to first construct the matrix $$P$$.
$$P = \left(
\begin{array}{ccccc}
 0 & \frac{1}{4} & \frac{1}{4} & \frac{1}{4} & \frac{1}{4} \\
 0 & \frac{1}{4} & \frac{1}{4} & \frac{1}{4} & \frac{1}{4} \\
 0 & 0 & \frac{1}{2} & \frac{1}{4} & \frac{1}{4} \\
 0 & 0 & 0 & \frac{3}{4} & \frac{1}{4} \\
 0 & 0 & 0 & 0 & 1
\end{array}
\right)$$ Then we square it: $$P^2 = 
\left(
\begin{array}{ccccc}
 0 & \frac{1}{16} & \frac{3}{16} & \frac{5}{16} & \frac{7}{16} \\
 0 & \frac{1}{16} & \frac{3}{16} & \frac{5}{16} & \frac{7}{16} \\
 0 & 0 & \frac{1}{4} & \frac{5}{16} & \frac{7}{16} \\
 0 & 0 & 0 & \frac{9}{16} & \frac{7}{16} \\
 0 & 0 & 0 & 0 & 1
\end{array}
\right)$$ **Answer not verified**

### 

So, if we start with $$i$$ balls in the urn, what is the probability that
we have $$j$$ after drawing $$m$$ and discarding all the white balls. The
obvious first simplification we can make is that you can't end up with
fewer that the $$N-m$$ white balls after drawing:
$$p_{ij} = 0, j > N - m$$ You also can't gain white balls
$$p_{ij} = 0, j > i$$ OK! now for the interesting one. There are
$$\binom{N}{m}$$ ways to draw $$m$$ balls from the urn. In any given step,
you are going to draw $$i-j$$ white balls from a total of $$i$$ and
$$m - i +j$$ black balls from a total of $$N-i$$. Thus there are
$$\binom{i}{i-j}\binom{N-i}{m - i +j}$$ ways to make that draw.
$$p_{ij} = \frac{\binom{i}{i-j}\binom{N-i}{m - i +j}}{\binom{N}{m}}, \text{otherwise}$$
**Answer verified**

### 

Once again, we start building the transition matrix. $$P = \left(
\begin{array}{ccccccccc}
 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
 \frac{1}{2} & \frac{1}{2} & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
 \frac{3}{14} & \frac{4}{7} & \frac{3}{14} & 0 & 0 & 0 & 0 & 0 & 0 \\
 \frac{1}{14} & \frac{3}{7} & \frac{3}{7} & \frac{1}{14} & 0 & 0 & 0 & 0 & 0 \\
 \frac{1}{70} & \frac{8}{35} & \frac{18}{35} & \frac{8}{35} & \frac{1}{70} & 0 & 0 & 0 & 0 \\
 0 & \frac{1}{14} & \frac{3}{7} & \frac{3}{7} & \frac{1}{14} & 0 & 0 & 0 & 0 \\
 0 & 0 & \frac{3}{14} & \frac{4}{7} & \frac{3}{14} & 0 & 0 & 0 & 0 \\
 0 & 0 & 0 & \frac{1}{2} & \frac{1}{2} & 0 & 0 & 0 & 0 \\
 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0
\end{array}
\right)$$ And then just square it! $$P^2 = \left(
\begin{array}{ccccccccc}
 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
 \frac{3}{4} & \frac{1}{4} & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
 \frac{107}{196} & \frac{20}{49} & \frac{9}{196} & 0 & 0 & 0 & 0 & 0 & 0 \\
 \frac{75}{196} & \frac{24}{49} & \frac{6}{49} & \frac{1}{196} & 0 & 0 & 0 & 0 & 0 \\
 \frac{1251}{4900} & \frac{624}{1225} & \frac{264}{1225} & \frac{24}{1225} & \frac{1}{4900} & 0 & 0 & 0 & 0 \\
 \frac{39}{245} & \frac{471}{980} & \frac{153}{490} & \frac{23}{490} & \frac{1}{980} & 0 & 0 & 0 & 0 \\
 \frac{22}{245} & \frac{102}{245} & \frac{393}{980} & \frac{22}{245} & \frac{3}{980} & 0 & 0 & 0 & 0 \\
 \frac{3}{70} & \frac{23}{70} & \frac{33}{70} & \frac{3}{20} & \frac{1}{140} & 0 & 0 & 0 & 0 \\
 \frac{1}{70} & \frac{8}{35} & \frac{18}{35} & \frac{8}{35} & \frac{1}{70} & 0 & 0 & 0 & 0
\end{array}
\right)$$ Telling us there is a 39 in 245 chance that if we start with 5
balls that we'll be at zero after two steps. **Answer verified**

### 

Since our chain is finite-dimensional and each state is accessible from
every other state, all states are persistent by the corollary to theorem
7.3.

**Answer not verified**

### 

As the problem alludes to, if you start off on at any given point, there
is zero probability of being at that point during the next step. Hence,
the elements of the transfer matrix oscillate between being 0 and
non-zero so the limit of 7.20 is always zero and not greater than zero
as implied.

**Answer not verified**

### 

Because you can now conceivably stay at the edge point, that means that
every point is now accessible from every other point at every time step
after a certain period of time has elapsed. Since we're also finite
dimensional and now accessible, 7.20 is now satisfied and we have proven
the existence of the stationary probabilities.

**Answer not verified**

### 

We now want to solve:

$$p_j^* = \sum_{i=1}^{m} p_i^* p_{ij}$$ So let's get to it!

$$\begin{aligned}
    p_1^* = \sum_{i=1}^{m} p_i^* p_{i1} = q p_1^* + q p_2^* \\
    p_2^* = \sum_{i=1}^{m} p_i^* p_{i2} = p p_1^* + q p_3^* \\
    p_j^* = p p_{j-1}^* + q p_{j+1}^* \\
    p_m^* = \sum_{i=1}^{m} p_i^* p_{im} = p p_{m-1}^* + p p_{m}^*\end{aligned}$$
You can easily solve the first equation to get:
$$p_2^* =\frac{p}{q} p_1^*$$ Then if we look to solve the second
equation for $$p_2^*$$, $$\begin{aligned}
    p_2^*  = p p_1^* + q p_3^* \\
    \frac{p}{q} p_1^* = p p_1^* + q p_3^* \\
    \left( \frac{p}{q^2} - \frac{p}{q} \right) p_1^* = p_3^* \\
    p_3^* = \left( \frac{p}{q} \right)^2 p_1^*\end{aligned}$$ So clearly
if we were to put that back into the third equation, we'd get another
factor of $$p/q$$, ergo: $$\begin{aligned}
    p_j^* = \left( \frac{p}{q} \right)^{j-1}p_1^*\end{aligned}$$ This
needs to be normalized, however: $$\begin{aligned}
    1 = \sum_{j = 1}^m p_j^* = \sum_{j = 1}^m \left( \frac{p}{q} \right)^{j-1}p_1^* \\
    1 = \frac{q \left(\left(\frac{p}{q}\right)^m-1\right)}{p-q} p_1^* \\
    p_1^* = \frac{p-q}{q \left(\left(\frac{p}{q}\right)^m-1\right)}\end{aligned}$$
But that's only if $$q \neq p$$. Clearly, if they are equal, each term of
the sum will just be equal to 1. giving: $$\begin{aligned}
    1 = \sum_{j = 1}^m p_j^* = \sum_{j = 1}^m \left( \frac{p}{q} \right)^{j-1}p_1^* \\
    1 = m p_1^* \\
    p_1^* = \frac{1}{m} \end{aligned}$$ **Answer verified**

### 

Let's turn $$A$$ into 1 and $$B$$ into 2. So, if 1 is shooting, there is a
$$\alpha$$ probability that 1 will go next whereas there is a $$1 - \alpha$$
probability that 2 goes next. Similarly, if 2 is shooting, there is a
$$1 - \beta$$ chance he shoots again and a $$\beta$$ chance that 1 goes
next. That gives us a transfer matrix of: $$P = \left(
\begin{array}{cc}
 \alpha  & 1-\alpha  \\
 \beta  & 1 - \beta  \\
\end{array}
\right)$$ We want to solve the equation: $$\begin{aligned}
    \pi^T P = \pi^T \\
    \pi^T (P - I) = 0 \\
    \pi_1 +\pi_2 = 0\end{aligned}$$ I got lazy so I plugged the last two
equations into my choice of symbolic manipulation program and got:
$$\begin{aligned}
    p_1^* = \frac{\beta }{1 - \alpha +\beta} \\
    p_2^* = \frac{1 - \alpha }{1 - \alpha +\beta}\end{aligned}$$ Since
we want to know the probability the target eventually gets hit, the
first limiting probability is our choice since it represents the limit
that $$A$$ is firing as the number of shots goes towards infinity.

**Answer not verified** NOTE: this answer is different from the book's
answer\... I tried like a dozen times and kept getting this so I think
it may be wrong although I'd also believe that I am wrong so let me
know!

### 

$$p_j^* = \sum_{i=1}^{m} p_i^* p_{ij}$$ But we're also told:
$$1 = \sum_{i=1}^{m} p_{ij} = \sum_{j=1}^{m} p_{ij}$$ Let's expand
things a bit: $$\begin{aligned}
    p^*_1 = p_{11}p^*_1 + p_{21}p^*_2 + p_{31}p^*_3 + \cdots    p_{m1}p^*_m \\
    p^*_2 = p_{12}p^*_1 + p_{22}p^*_2 + p_{32}p^*_3 + \cdots    p_{m2}p^*_m \\
    \cdots \\
    p^*_m = p_{1m}p^*_1 + p_{2m}p^*_2 + p_{3m}p^*_3 + \cdots    p_{mm}p^*_m\end{aligned}$$
You will notice that a clear solution is every $$p^*$$ being unity and
since it is a solution, that's all we care for. Then, since the solution
must be normalized, they all turn out to actually be
$$p^*_i = \frac{1}{m}$$. **Answer not verified**

### 

**Solution practically in book**

### 

$$p_j^* = \sum_{i=1}^{\infty} p_i^* p_{ij}$$ So let's solve\...
$$\begin{aligned}
    p_1^* = \sum_{i=1}^{\infty} p_i^* p_{i1} = \sum_{i=1}^{m} p_i^* \frac{i}{i+1} \\
    p_j^* = \sum_{i=1}^{\infty} p_i^* p_{ij} = \frac{1}{j} p^*_{j-1} \\
    p_j^* = \frac{1}{j!} p_1^* \\\end{aligned}$$ Normalize this
$$\begin{aligned}
    \sum_{j=1}^{\infty} p_j^* = 1 = \sum_{j=1}^{\infty} \frac{1}{j!} p_1^* \\
    1 = (1-e)p_1^* \\
    p_1^* = \frac{1}{1-e} \\
    p_j^* = \frac{1}{j!(1-e)} \end{aligned}$$

**Answer verified**

### 

**Solution practically in book**

### 

If the stakes are doubled, it's like playing the game without doubled
stakes but half the capital on both sides in which case it is clear that
$$\hat{p_j}$$ gets bigger. **Answer verified**

### 

Play with the two possible limits in equation 7.34. **Sorry\... maybe
some other day**
