# Combination of Events

## Important/Useful Theorems

### 

If $A_1 \subset A_2$ then $\bar{A_1} \supset \bar{A_2}$

### 

If $A = A_1 \cup A_2$ then $\bar{A} = \bar{A_1} \cap \bar{A_2}$

### 

If $A = A_1 \cap A_2$ then $\bar{A} = \bar{A_1} \cup \bar{A_2}$

### 

$0\leq P(A) \leq 1$

### 

$P(A_1 \cup A_2) = P(A_1) + P(A_2) - P(A_1 \cap A_2 )$

### 

$P(A_1) \leq P(A_2)$ if $A_1 \subset A_2$

### 

Given any $n$ events $A_1, A_2, ... A_n$, let
$$P_1 = \sum_{i=1}^{n} P(A_i)$$
$$P_2 = \sum_{1 \leq i < j \leq n} P(A_iA_j)$$
$$P_3 = \sum_{1 \leq i < j < k \leq n} P(A_i A_j A_k)$$ etc. for
integers up to $n$. Then
$$P\left(\bigcup_{k=1}^n A_k \right) = P_1 -P_2 + P_3 + ... \pm P_n$$

### 

If $A_1, A_2, ...$ is an increasing sequence of events such that
$A_1 \subset A_2 \subset ...$
$$P\left(\bigcup_{k=1}^n A_k \right) = \lim_{n \rightarrow \infty} P(A_n)$$

### 

If $A_1, A_2, ...$ is an decreasing sequence of events such that
$A_1 \supset A_2 \supset ...$
$$P\left(\bigcap_{k=1}^n A_k \right) = \lim_{n \rightarrow \infty} P(A_n)$$

### 

For any collection of arbitrary events
$$P\left(\bigcup_{k} A_k \right) \leq \sum_k P(A_k)$$

### 

For any sequence of events $A_1, A_2, ...$ with porbabilities
$P(A_k)=p_k$ $$\sum_k p_k < \infty$$

## Answers to Problems

### 

#### a--$AB=A$

$AB=A$ is shorthand for $A\cap B=A$ or, in English, that the
intersection of the events $A$ and $B$ is equivalent to the event $A$.
Clearly, this means that $A$ is equivalent to $B$ such that their
overlap is complete--wither because they are exactly the same or because
all outcomes in B are contained in A

#### b--$ABC=A$

The intersection $A$, $B$ and $C$ is equivalent to A. This implies that
both $B$ and $C$ are either the same as $A$ or contained within $A$

#### c--$A\cup B\cup C=A$

The intersection of $A$, $B$ and $C$ is equivalent to $A$. This
statement implies that the overlap of $B$ and $C$ is $A$.

**Answer not verified**

### 

#### a--$A \cup B = \bar{A}$

The union of $A$ and $B$ is the complement of $A$. This statement seems
to imply that the sum of the events in both $A$ and $B$ are the same as
the events NOT in A. This seeming contradiction can only be true if $A$
as a set contains no events and $B$ contains all events.

#### b--$AB=\bar{A}$

The intersection of $A$ and $B$ is equivalent to the complement of $A$.
This statement implies that the events contained in both $A$ and $B$ are
the events NOT in $A$. Again, this strange statement can be fulfilled if
$A$ contains all events and $B$ contains no events

#### c--$A \cup B = A \cap B$

The union of $A$ and $B$ is equal to the intersection of $A$ and $B$.
This statement implies that the sum of all events in $A$ and $B$ is
equivalent to the events in both $A$ and $B$. This can be true, if
$A=B$.

**Answer not verified**

### 

#### a--$(A \cup B)(B \cup C)$

In order to do this problem, we need to develop a distributive law for
$\cap$ and $\cup$; we need to figure out what $A \cup (B \cap C)$ and
$A \cap (B \cup C)$ is.

$A \cup (B \cap C)$ is the union of A and the intersection of $B$ and
$C$ which one can easily imagine is the same as the union of the
intersection of the union of $A$ and $B$, and $A$ and $C$.

$$A \cup (B \cap C) = (A \cup B) \cap (A \cup C)$$ similarly, for the
second relation $$A \cap (B \cup C) = (A \cap B) \cup (A \cap C)$$

ALRIGHT Back to the problem at hand $$(A \cup B) \cap (B \cup C)=D$$ For
bookkeeping purposes, I have defined our expression to equal $D$. Let's
distribute\...
$$(A \cap B) \cup (A \cap C)\cup (B \cap B)\cup (B \cap C)$$ clearly the
intersection of $B$ and $B$ is just $B$
$$(A \cap B) \cup (A \cap C)\cup B \cup (B \cap C)$$ let's rearrange it
to look like what we want to prove (cheap move, I know\... but
effective!).
$$(AC\cup B) \cup \left[(A \cap B) \cup (B \cap C)\right]=D$$ If we
could prove that the term in brackets is equal to $D$ then we would
prove that $AC\cup B=D$. So let's work on that:
$$(A \cap B) \cup (B \cap C) = E$$ Now let's distribute
$$(A \cup B) \cap (B \cup C) \cap (B \cup B)\cap (A \cup C) = E$$
simplify and rearrange
$$\left[(A \cup B) \cap (B \cup C)\right] \cap B\cap (A \cup C) = E$$ we
recognize that the term in brackets is equivalent to the original
expression we're trying to work with and then distribute one last time
$$D \cap \left[(B \cap A) (B \cap C)\right] = E$$ Now the term in
brackets is equal to event $E$ meaning that $D=E$! So we go back to the
first equation. $$(AC\cup B) \cup D = D$$
$$(AC\cup B) = D = (A \cup B) \cap (B \cup C)$$ $\Box$

**Answer not verified**

#### b--$(A \cup B)(A \cup \bar{B})$

$$(A \cup B)\cap (A \cup \bar{B})$$ distribute out
$$(A \cap A)\cup (A \cap \bar{B})\cup (B \cap A)\cup (B \cap \bar{B})$$
$A$ will intersect with itself only at itself and $B$ will not intersect
with its copmlement. $$A\cup (A \cap \bar{B})\cup (B \cap A)\cup 0$$
Unionizing the null set with anything else will leave the other things
untouched. $$A\cup (A \cap \bar{B})\cup (B \cap A)$$ Now we rearrange
and redistribute.
$$A\cup \left[  (A \cup A) \cap (A \cup B) \cap (A \cup \bar{B}) \cap (B \cup \bar{B}) \right]$$
Clearly the sum of things in $B$ and not in $B$ is everything.
$$A\cup \left[  A \cap (A \cup B) \cap (A \cup \bar{B}) \cap \Omega \right]$$
Clearly the intersection of anything with everything is the same
anything
$$A\cup \left[  A \cap (A \cup B) \cap (A \cup \bar{B}) \right]$$
Distribute.
$$(A\cup A) \cap \left[  A \cup (A \cup B) \right]\cap \left[  A \cup (A \cup \bar{B}) \right]$$
$$A \cap \left[  (A \cup A) \cup B \right]\cap \left[  (A \cup A) \cup \bar{B} \right]$$
$$A \cap \left[  A \cup B \right]\cap \left[ A \cup \bar{B} \right]$$
Now we call the original statement of the problem event $C$
$$A \cap C = C$$ Which as we discussed in problem 2.1a, implies that
$A=C$. Therefore: $$(A \cup B)(A \cup \bar{B}) = A
\label{answer2.3b}$$ $\Box$

**Answer not verified**

#### c--$(A \cup B)(A \cup \bar{B})(\bar{A} \cup B)$

Start from the identity we just proved
$$(A \cup B)(A \cup \bar{B})(\bar{A} \cup B) = A \cap (\bar{A} \cup B)$$
Distribute $$(A \cap \bar{A}) \cup (A \cap B)$$ $$0 \cup AB$$ $$AB$$
$\Box$

**Answer not verified**

### 

Solve for $X$:
$$\overline{(X \cup A)} \cup \overline{(X \cup \overline{A})} = B$$ We
negate both sides and use DeMorgan's theorem
$$(X \cup A) \cap (X \cup \overline{A}) = \overline{B}$$ Since we just
fought so hard to prove that $(A \cup B)(B \cup C) = AC \cup B$, we want
to use it! $$(A \cup X) \cap (X \cup \overline{A}) = \overline{B}$$
$$(A \overline{A}) \cup X = \overline{B}$$ $$0 \cup X = \overline{B}$$
$$X = \overline{B}$$ $\Box$

**Answer not verified**

### 

$A$ is the event that at least one of three inspected items is defective
and $B$ is the event that all three inspected items are defective. In
this circumstance, the overlap of $A$ and $B$ ($A \cap B$) is equal to
$A$ since $A \subset B$. For similar reasons, the union of those two
($A \cup B$) is simply equal to $B$.

**Answer not verified**

### 

Since $B$ is the event a number ends in a zero, $\overline{B}$ is the
event a number does not end in a zero. Thus, the overlap of that event
and a number being divisible by 5 is the same as saying the event that a
number is divisible by 5 and not divisible by 10.

**Answer not verified**

### 

It is easy to prove to oneself that these events $A_k$ are increasing
such that
$A_1 \subset A_2 \subset A_3 \subset A_4 \subset ... \subset A_10$ since
the radii of the disks are getting bigger. Therefore, according to
theorem 2.3 from the text $$B = \bigcup_{k=1}^6 A_k = A_6$$ Intuitively,
this is clear because we're taking the $A_6$ contains all the lower
events so if you take the union of all the lower events then you're just
adding parts of $A_6$ to itself. Now, for $$C = \bigcap_{k=5}^{10} A_k$$
will clearly be $A_5$ because it's the smallest event and thus
automatically sets the intersection to not be able to be any smaller.

**Answer verified**

### 

Luckily with just a bit of work, we can prove that the two statements
are equivalent $$P(A) + P(\overline{A}) = 1$$ To prove this, we'll use
the probability addition theorem for two events which will be $A$ and
$\overline{A}$ for us. $$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$
substitute:
$$P(A \cup \overline{A}) = P(A) + P(\overline{A}) - P(A \cap \overline{A})$$
using the fact that the union of anything with everything else is
everything and that the intersection of anything with anything else is
nothing\... $$P(\Omega) = P(A) + P(\overline{A}) - P(0)$$ Now we use the
fact that the probability of ell events is unity and the probability of
no events is nothing. $$1 = P(A) + P(\overline{A}) - 0$$ $\Box$

**Answer not verified**

### 

I think this problem is as easy as it sounds since they're very clear to
label the probabilities as belonging to the rings and the disk. Thus, if
you draw a picture of the bulls-eye it will become clear that the answer
is just $1-.35-.30-.25=.10$ probability of missing. **Answer not
verified**

### 

Let $A_i$ be the event where $i$ or fewer defective items are found;
we're looking, then for the event $B$ that we find at least one
defective item: $$B = \bigcup_{i=1}^5 A_i$$
$$P(B) = P \left( \bigcup_{i=1}^5 A_i  \right)$$ Consequently
$$P(\overline{B}) = P \left( \bigcup_{i=1}^5 \overline{A_i}  \right)$$
Clearly $A_i$ are an increasing series of events so consequently,
$\overline{A_i}$ are a decreasing series of events and we can use
theorem $$P(B) = 1- P(\overline{B}) = 1- P ( \overline{A_5} )$$

This is saying the probability of rejecting the batch that has 5 bad
items in it is 1 minus the probability of not finding 5 or fewer bad
items: or of finding 5 good items.

There are $\binom{100}{5}$ ways to pick 5 items from 100. With 95 good
items, there are $\binom{95}{5}$ ways to pick up 5 good items.
$$P = 1 - \frac{\binom{95}{5}}{\binom{100}{5}} = 1 - \frac{95!}{5! \cdot 90! } \frac{95! \cdot 5!}{100!} = 1 - \frac{95!}{90! } \frac{95!}{100!}$$
$$P = 1 - \frac{95\cdot 94\cdot 93\cdot 92\cdot 91}{100 \cdot 99\cdot 98\cdot 97\cdot 96} = \frac{2478143}{10755360} = 0.23041
\label{answer2.10}$$ **Answer verified**

### 

This problem boils down to the game you may have played as a kid to
decide who got to go first or ride in the front of the car: I'm thinking
of a number between 1 and 10. Granted here the secretary gets to guess
until she gets it right. Let $A_i$ be the event that the secretary
guesses just $i$ times (gets it right on the $i$th try). Thus we want:
$$P(B) = P\left(\bigcup_{i=1}^4 A_i \right) = \sum_{i=1}^4 P(A_i)$$ This
factorization brought to you by the letter 'I' and the fact that you
can't simultaneously get it right on the first try and get it right on
the second try, meaning these events are mutually exclusive.

For $i$ trials, there are $P^{10}_i$ possible ways to have tried. Note
that here order matters since we're talking about successive trials.
Consequently, there are $P^9_{i-1}$ ways to chose $i-1$ wrong numbers,
giving $$P_i=\frac{P^9_{i-1}}{P^{10}_i}=\frac{1}{10}$$ This answer, in
itself, makes some sense although it does, at first glance, appear weird
since you expect to gain an advantage as you learn about which numbers
aren't correct. In reality, there's 10 possible chances to get it right
and you're no more likely to get it right on the first than on the
second etc.; that's the best intuitive way to think about it.
$$P(B) = \sum_{i=1}^4 \frac{1}{10} = .4$$ Now, if the secretary knew
that the last number had to be even, that reduces our total number of
possible numbers to 5 so the probability becomes:
$$P_i=\frac{P^4_{i-1}}{P^{5}_i}=\frac{1}{5}$$ so instead we get
$$P(B) = \sum_{i=1}^4 \frac{1}{5} = .8$$

**Answer not verified**

### 

We proved earlier that $P(A) = 1 - P(\overline{A})$ so if we define

$$B = \bigcup_{k=1}^n A_k$$ then we know
$$\overline{B} = \bigcap_{k=1}^n \overline{A_k}$$ and thus
$$1 - P\left(\bigcup_{k=1}^n A_k\right)  =P\left( \bigcap_{k=1}^n \overline{A_k} \right)$$
$\Box$

**Answer not verified**

### 

Here we need to find the probability that you get 5 good ones and the
probability you get one bad one; their sum will be the probability that
the batch passes.

For both probabilities, the total number of outcomes is equal to the
total number of groups of 50 objects out of the 100 or
$\binom{100}{50} = 100891344545564193334812497256$.

For finding 5 good ones, there $\binom{95}{50}$ ways to pick 50 good
ones. Additionally, there's $\binom{95}{49}\binom{5}{1}$ to get 49 good
ones and one bad one.

$$P = \frac{\binom{95}{50}+\binom{95}{49}\binom{5}{1}}{\binom{100}{50}} = \frac{1739}{9603} = 0.181089
\label{answer2.13}$$ **Answer verified**

### 

To solve this problem, you want to consider the people as buckets and
the birthdays as things you put in the buckets. As such, there are
$365^r$ different possible ways for people to r people to have
birthdays.

In order to get the probability that at least one person shares a
birthday, like we proved for the earlier problem, we need to just find
the probability that no one shares a birthday and subtract that from 1.

For the first person, there are 365 possibilities, 364 for the next, ad
nauseum, giving $P^{365}_r$ ways of no one sharing a birthday.

$$P(r) = 1 - \frac{P^{365}_r}{365^r} = 1 - \frac{365!}{365^r(365-r)!}
\label{answern2.14}$$ **Answer verified by wikipedia**

### 

We are asked to check how good the truncation of
$$1-e^x = -\sum_{n=1}^\infty \frac{x^n}{n!}$$ is for $n=3, 4, 5, 6$ at
$x=-1$.
$$1-e^{-1} = 0.632121 \approx 1 - \frac{1}{2} + \frac{1}{6} - \frac{1}{24}  + \frac{1}{120} - \frac{1}{720}$$
These truncations go as: $\{0.666667,0.625,0.633333,0.631944\}$ which
relative to the exact answer are $\{1.05465,0.988735,1.00192,0.999721\}$

**Answer not verified**

### 

Here we have 4 events to look at: the probability that each player gets
dealt a hand of one suit.
$$P_1 = P(A_1) + P(A_2) + P(A_3) + P(A_4) = 4P(A_i)$$
$$P_2 = P(A_1 A_2) + P(A_1 A_3) + P(A_1 A_4) + P(A_2 A_3) + P(A_2 A_4) + P(A_3 A_4) = 6 P(A_i A_j)$$
because each one of these is equivalent to the others (player 2 and 3
getting full hands is no more likely than player 4 and 1).
$$P_3 = P(A_1 A_2 A_3) + P(A_1 A_2 A_4) + P(A_1 A_3 A_4) + P(A_2 A_3 A_4) = 4 P(A_i A_j A_k)$$
$$P_4 = P(A_1 A_2 A_3 A_4)$$

For one player's hand, there are $\binom{52}{13}$ possibilities, only 4
of which are all the same suit. For two players, the first player once
again has $\binom{52}{13}$ possibilities but that leaves only
$\binom{39}{13}$ for the second player (and $\binom{26}{13}$ for the
third) giving a total number of hands for two players as
$\binom{52}{13}\binom{39}{13}$ and
$\binom{52}{13}\binom{39}{13}\binom{26}{13}$ for 3/4 players.

For 2 players, there are 4 possible hands for the first, leaving 3 for
the second and giving 12 total desirable hands. Consequently for three
players, there are two more options giving 24 total desirable hands.

$$P_1 =  4\left(\frac{4}{\binom{52}{13}}\right)$$
$$P_2 = 6\left( \frac{12}{\binom{52}{13}\binom{39}{13}} \right)$$
$$P_3 = 4\left( \frac{24}{\binom{52}{13}\binom{39}{13}\binom{26}{13}} \right)$$
$$P_4 = \frac{24}{\binom{52}{13}\binom{39}{13}\binom{26}{13}}$$
$$P\left(A_1 \cup A_2 \cup A_3 \cup A_4\right) = P_1 - P_2 +P_3 - P_4$$
$$P\left(A_1 \cup A_2 \cup A_3 \cup A_4\right) = \frac{16}{\binom{52}{13}} - \frac{72}{\binom{52}{13}\binom{39}{13}} + \frac{72}{\binom{52}{13}\binom{39}{13}\binom{26}{13}}
\label{answer2.16}$$ because I have mathematica and I'm feeling lazy at
the moment, I am just going to forgo the Stirling's Approximation part
and tell you:
$$P\left(A_1 \cup A_2 \cup A_3 \cup A_4\right) = \frac{18772910672458601}{745065802298455456100520000} \approx 2.5196312345 \cdot 10^{-11}$$

**Answer verified**

### 

SKIPPED

$$0=0
\label{answer2.17}$$ **Answer \[not\] verified**

### 

SKIPPED

$$0=0
\label{answer2.18}$$ **Answer \[not\] verified**

### 

SKIPPED

$$0=0
\label{answer2.19}$$ **Answer \[not\] verified**
