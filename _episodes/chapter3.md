# Combination of Events

## Important/Useful Theorems

### 

$$P(A|B) = \frac{P(AB)}{P(B)}$$

### 

If $A = \bigcup{}_k A_k$ where $A_k$ are all mutually exclusive
$$P(A|B) = \sum_k P(A_k|B)$$

### 

The events $A_1, A_2, A_3, ..., A_n$ are said to be statistically
independent if $$\begin{aligned}
    P(A_i A_j) = P(A_i)P(A_j) \\
    P(A_i A_j A_k) = P(A_i)P(A_j)P(A_k) \\
    ... \\
    P(A_1 A_2 A_3...A_n) = P(A_1)P(A_2)P(A_3)...P(A_n)\end{aligned}$$
For all possible combinations of the indeces.

### Second Borel-Cantelli lemma

If the events $A_1, A_2, A_3, ...$ are all statistically independent and
$p_k = P(A_k$ $$\sum_{k=1}^\infty p_k = \infty$$

## Answers to Problems

### 

We need to examine $A(\bar{A}B)$ , $A \overline{A \cup B}$ and
$(\bar{A}B) \overline{A \cup B}$.

$$A(\bar{A}B) = A\bar{A}B = 0$$ since $A$ will not overlap with its
complement at all $$A \overline{A \cup B} = A \bar{A}\bar{B} = 0$$ for
the same reason
$$(\bar{A}B) \overline{A \cup B} = \bar{A}B \bar{A}\bar{B} = 0$$ again
for the same reason.

Clearly, since none of the three proposed events intersect at all with
any of the others, these events are mutually exclusive and cannot happen
in concert. **Answer not verified**

### 

To get $C$ to be mutually exclusive, its overlap with both $B$ and $A$
must be zero, giving a logical conclusion of:

$$C = \bar{A}\bar{B}
\label{answer3.2}

To put it in English, the mutually exclusive outcomes of a chess match are, white wins, black wins and neither wins (a draw).$$
**Answer not verified**

### 

$$P(A|B) = \frac{P(AB)}{P(B)} = \frac{P(BA)}{P(B)} \frac{P(A)}{P(A)} = P(B|A) \frac{P(A)}{P(B)} > P(A)$$

$$P(B|A)  >  P(B)
\label{answer3.3}$$

**Answer not verified**

### 

$$P(A|B) = \frac{P(AB)}{P(B)} = \frac{3}{2} P(AB)$$ We can come up for
an expression for $P(AB)$ in terms of $P(A\cup B)$ with a little
algebra. $$P(A \cup B) = P(A) +P(B) - P(AB) = \frac{4}{3} - P(AB)$$
clearly $P(A \cup B)$ must be between 0 and 1.
$$1 \geq \frac{4}{3} - P(AB) \geq 0$$
$$-\frac{1}{3} \geq - P(AB) \geq -\frac{4}{3}$$
$$\frac{1}{3} \leq  P(AB) \leq \frac{4}{3}$$
$$\frac{1}{3} \leq  \frac{2}{3} P(A|B) \leq \frac{4}{3}$$
$$\frac{1}{2} \leq  P(A|B) \leq 2$$ $$\frac{1}{2} \leq  P(A|B)
\label{answer3.4}$$ **Answer not verified**

### 

$$P(A)P(B|A)P(C|AB) = P(A)\frac{P(BA)}{P(A)} \frac{P(ABC)}{P(AB)} = P(ABC)$$
Simple enough, but now we need a general theorem
$$P\left(  \bigcap_{k=1}^n A_k  \right) = P(A_1)P(A_2|A_1)P(A_3|A_1A_2)P(A_4|A_1A_2A_3) ... P\left(A_n| \bigcap_{k=1}^{n-1} A_k  \right)$$

$$P\left(  \bigcap_{k=1}^n A_k  \right) = P(A_1) \prod_{z=2}^n} P\left(A_z| \bigcap_{k=1}^{z-1} A_k  \right)
\label{answer3.5}$$ **Answer not verified**

### 

$$P(A) = P(A|B) +  P(A|\bar{B}) = \frac{P(AB)}{P(B)} + \frac{P(A\bar{B})}{P(\bar{B})}$$

#### $A=0$

$$P(0) = \frac{P(0)}{P(B)} + \frac{P(0)}{P(\bar{B})} = 0$$ Because the
intersection with nothing is nothing.

#### $B=0$

$$P(A) = \frac{P(0)}{P(0)} + \frac{P(A\Omega)}{P(\Omega)} = P(A)$$

#### $B=\Omega$

$$P(A) = \frac{P(A\Omega)}{P(\Omega)} + \frac{P(0)}{P(0)} = P(A)$$

#### $B=A$

$$P(A) = \frac{P(AA)}{P(A)} + \frac{P(A\bar{A})}{P(\bar{A})} = 1 ???$$ I
cannot prove this to be true

#### $B=\bar{A}$

$$P(A) = \frac{P(A\bar{A})}{P(\bar{A})} + \frac{P(AA)}{P(A)} = 1 ???$$ I
cannot prove this to be true

**Answer not verified**

### 

We start knowing: $$P(AB) = P(A)P(B)$$ But we want to prove:
$$P(\bar{A}\bar{B}) = P(\bar{A})P(\bar{B})$$ as the problem suggests
$$P(B|A) + P(\bar{B}|A) = 1$$
$$\frac{P(BA)}{P(A)} + \frac{P(A\bar{B})}{P(A)} = 1$$
$$P(B) + \frac{P(A\bar{B})}{P(A)} = 1$$
$$\frac{P(A\bar{B})}{P(A)} = 1 - P(B)$$
$$P(A\bar{B}) = \left(1 - P(B)\right)P(A)$$
$$P(A\bar{B}) = P(\bar{B})P(A)$$ so clearly $A \text{ and } \bar{B}$ are
independent. Since we have just proved that for two events that are
independent, we can show that one of the events is independent of the
others complement, without loss of generality, we can apply this logic
recursively, thus proving that if two events are independent, so are
their complements.

**Answer not verified**

### 

Given two mutually exclusive events, we wonder if they are dependent.
$$P(AB) = 0$$ but in order for them to be independent, we need to be
able to say $$P(AB) = P(A)P(B)$$ but we were told that $P(A)$ and $P(B)$
are positive, therefore they are **NOT** independent! **Answer not
verified**

### 

Let $A_i$ be the event of getting a white ball at the $i^{th}$ urn.
Since there are 2 possibilities for each step, there are $2^n$ different
ways to do this "experiment". Clearly $$P(A_1) = \frac{w}{w+b}$$ but
things get more complicated as $i>1$. Here there are two mutually
exclusive possibilities: we got a white one or we got a black one from
the first urn.
$$P(A_2) = P(A_2|A_1)P(A_1) + P(A_2|\bar{A_1})P(\bar{A_1})$$
$$P(A_2) = P(A_2A_1) + P(A_2\bar{A_1})$$ For both denominators, there
were $w+b$ possibilities to begin with and $w+b+1$ for the second. Then,
there were $w$ ways to get white first and $b$ ways to get black first.
Then, respectively, there would be $w+1$ ways and $w$ ways to get white
for the second urn.

$$P(A_2) = \frac{1}{(w+b)(w+b+1)} \left( w(w+1) + bw \right)$$
$$P(A_2) = \frac{w(w+b+1)}{(w+b)(w+b+1)} = \frac{w}{w+b}$$

Since we started off with $\frac{w}{w+b}$ and then got the same answer
for the second step, if we were to do it for a third step, a fourth
step, etc. we would be starting with the same initial conditions and
would get the same answer therefore this is true in general for $n$
urns.

**Answer not verified**

### 

Labelling the points starting from the left, we know that
$$\begin{aligned}
    P(C_1|B_1) = \frac{1}{3} = P(C_2|B_1) \\
    P(C_3|B_2) = \frac{1}{2} \\
    P(C_4|B_4) = \frac{1}{5} = P(C_5|B_4) = P(C_6|B_4)\end{aligned}$$
Since there's only one way to get to each of these specified end-points,
the total probabilities are just the conditional probabilities times the
probabilities of the conditions. $$\begin{aligned}
    P(C_1) = P(C_1|B_1)P(B_1) = \frac{1}{12} = P(C_2) \\
    P(C_3) = P(C_3|B_2)P(B_2) = \frac{1}{8} \\
    P(C_4) = P(C_4|B_4)P(B_4) = \frac{1}{20} = P(C_5) = P(C_6)\end{aligned}$$
So when we combine all these possibilities with the probability of
getting to $A$:
$$\frac{67}{120} + \frac{2}{12} + \frac{1}{8} + \frac{3}{20} = 1
\label{answer3.10}$$ **Answer not verified**

### 

Let's go from 1 dollar stakes to $q$ dollar stakes; all the other
variables stay the same.

$$p(x) = \frac{1}{2} \left[  p(x+q) + p(x-q)  \right], q \leq x \leq m - q$$
but the boundary conditions do not change. Therefore, there is no change
here between the original linear equation and the new linear equation.
Thus: $$p(x) = 1 - \frac{x}{m}
\label{answer3.11}$$ **Answer not verified**

### 

$$P(B|A) = P(B | \bar{A})$$ but let's work on the assumption they are
not independent $$\begin{aligned}
    \frac{P(AB)}{P(A)} = \frac{P(\bar{A}B)}{\bar{A}} \\
    \frac{P(AB)}{P(\bar{A}B)} = \frac{P(A)}{P(\bar{A})}\end{aligned}$$
but now let's factor out the expression for the intersection, leaving
some error behind $$\begin{aligned}
    \frac{P(A)P(B)\epsilon}{P(\bar{A})P(B)\epsilon{}'} = \frac{P(A)}{P(\bar{A})} \\ 
    \frac{P(A)\epsilon}{P(\bar{A})\epsilon{}'} = \frac{P(A)}{P(\bar{A})}\end{aligned}$$
The only way for this to always be true is for both error terms to be
the same and the only way for it to be true for arbitrary events is for
them to both be one such that $A$ and $B$ are independent. **Answer not
verified**

### 

There are four mutually exclusive outcomes for the first step: getting
two whites, one white and zero whites($A_2, A_1, A_0$). We'll call the
last event we want to turn out white, $B$.
$$P(B) = P(B|A_2)P(A_2) + P(B|A_1)P(A_1) + P(B|A_0)P(A_0)$$

$$\begin{aligned}
    P(B|A_2)P(A_2) =  1 \frac{w_1 w_2}{(w_1 + b_1)(w_2 + b_2)}\\
    P(B|A_1)P(A_1) = \frac{1}{2} \frac{w_1 b_2 + w_2 b_1}{(w_1 + b_1)(w_2 + b_2)} \\
    P(B|A_0)P(A_0) = 0\end{aligned}$$ putting it together
$$\begin{aligned}
    P(B) = \frac{1}{2} \frac{w_1 b_2 + w_2 b_1 + 2 w_1 w_2}{(w_1 + b_1)(w_2 + b_2)}  \\
    P(B) = \frac{1}{2} \frac{(w_1 +b_1 )w_2 + (w_2 + b_2) w_1 }{(w_1 + b_1)(w_2 + b_2)}  \\
    P(B) = \frac{1}{2} \left( \frac{w_1}{w_1+b_1} + \frac{w_2}{w_2+b_2} \right)\end{aligned}$$

**Answer not verified**

### 

As the hint suggests, we'll use Bayes' Rule. Our set of mutually
exclusive events $\{ B_i\}$ are the getting of the ball from the
$i^{th}$ urn and $A$ is the event of getting a white ball. So we want
the probability of choosing the odd urn out (we'll call it $B_k$) given
that we got a white ball.

$$P(B_k|A) = \frac{P(B_k)P(A|B_k)}{\sum_{i=1}^{10} P(B_i)P(A|B_i}$$ For
all $i$, $P(B_i) = \frac{1}{10}$ and for $i \neq k$
$P(A|B_i) = \frac{1}{2}$ but for $i=k$, $P(A|B_k) = \frac{5}{6}$
$$P(B_k|A) = \frac{\frac{5}{6 \cdot 10}}{9\frac{1}{20} + \frac{5}{6 \cdot 10}} = \frac{5}{32}$$

**Answer verified**

### 

To do this problem, let's figure out what the chance is that we picked
the all-white urn, the event $B_1$. $A$ is the event we pick a white
ball and $B_@$ is the event of picking the other urn with $\frac{3}{4}$
white balls.

$$P(B_1|A) = \frac{P(B_1)P(A|B_1)}{P(B_1)P(A|B_1) + P(B_2)P(A|B_2)} = \frac{\frac{1}{2}}{\frac{1}{2} + \frac{1}{2} \frac{3}{4}} = \frac{4}{7}$$
Thus, there is a $\frac{3}{7}$ chance we chose the urn that actually has
black balls in it and a $\frac{1}{4}$ chance that from that urn once
chooses a black ball giving an overall probability of picking a black
ball given the information in the problem of $\frac{3}{28}$. **Answer
verified**

### 

**SKIPPED**: unsure of meaning of problem.

### 

Clearly $P(A) = \frac{1}{2} = P(B) = P(C)$ and
$P(AB) = \frac{1}{4} = P(BC) = P(AC)$ because there is only one place
out of four for the die to hit both letters. There is, however, still
only one way to hit all three letters at once such that
$P(ABC)=\frac{1}{4}$ therefore the events in question are pairwise
independent since for all sets of letters $P(AB)=P(A)P(B)$ but not
completely independent since $P(ABC) \neq P(A)P(B)P(C)$. **Answer not
verified**

### 

**SKIPPED**: didn't feel like doing the problem\... so there.
