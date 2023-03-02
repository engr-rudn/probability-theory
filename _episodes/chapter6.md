---
title: "Questions and Answers-6"
teaching: 0
exercises: 0
questions:
objectives:
keypoints:

---

# Some Limit Theorems

## Important/Useful Theorems

### Weak Law of Large Numbers

Let $$\xi_1, \xi_2, ..., \xi_n$$ be $$n$$ independent indentically
distributed random variables with mean $$a$$ and variance $$\sigma^2$$.
Then, given any $$\delta > 0$$ and $$\epsilon >0$$, however small, there is
an integer, $$n$$ such that:
$$a- \epsilon \leq \frac{1}{n}(\xi_1 +\xi_2 + \cdots + \xi_n) \leq a + \epsilon$$
with probability greater than $$1- \delta$$.

### Generating Functions

For the **discrete** random variable $$\xi$$ with probability distribution
$$P_{\xi}(k)$$, the generating function is defined as

$$F_{\xi}(z) = \sum_{k=0}^{\infty}P_{\xi}(k)z^k$$ which yields some cool
relations: $$\begin{aligned}
    P_{\xi}(k) = \frac{1}{k!}F_{\xi}^{(k)}(0) \\
    \textbf{E}\xi = F'(1) \\
    \sigma^2 =F''(1) + F'(1) - [F'(1)]^2\end{aligned}$$

Note, also, that while you sum together random variables, you multiply
generating functions.

### Thm. 6.2

The sequence of probability distributions, $$P_n(k)$$ with generating
functions $$F_n(z)$$ where $$n=1, 2, 3, ....$$ converges weakly to the
distribution $$P(k)$$ with distribution function $$F(z)$$ iff

$$\lim_{n\rightarrow \infty} F_n(z) = F(z)$$

### Characteristic Functions

For a real random variable $$\xi$$, its generating function is defined as

$$f_{\xi}(t) = \textbf{E}e^{i \xi t} = F_{\xi}(e^{i t}) = \sum_{k=0}^{\infty} P_{\xi}(k)e^{i k t}$$
Or, if the variable is continuous:
$$f_{\xi}(t) = \int_{-\infty}^{\infty} P_{\xi}(x)e^{i x t} dx$$ Where it
is clear that is represents the discrete or continuous fourier transform
of the probability distribution and as such, the inverse is true:
$$P_{\xi}(x) = \frac{1}{2 \pi}\int_{-\infty}^{\infty} f_{\xi}(t)e^{-i x t} dt$$
Like Generating functions, there are some cool properties that can be
exploited: $$\begin{aligned}
    \textbf{E}\xi = -if_{\xi}'(0) \\
    \textbf{D}\xi = -f_{\xi}''(0) + [f_{\xi}'(0)]^2\end{aligned}$$ And
just like generating functions, they multiply together for random
variables that add together.

### The Central Limit Theorem

A sequence of random variables $$\xi_1, \xi_2, \xi_3, ...$$ is said to
satisfy the central limit theorem if
$$\lim_{n\rightarrow \infty} \textbf{P}\{ x' \leq \frac{S_n - \textbf{E}S_n}{\sqrt{\textbf{D}S_n}} \leq x''\} = \int_{x'}^{x''}e^{\frac{-x^2}{2}}dx$$
Where $$S_n = \sum_{k=1}^n \xi_k$$

### Thm. 6.3

If the same series of random variables as above each has mean $$a_k$$ and
variance $$\sigma^2_k$$ and satisfies the Lyapunov condition:
$$\lim_{n\rightarrow \infty} \frac{1}{B_n^3} \sum_{k=1}^n \textbf{E}|\xi_k - a_k|^3 = 0$$
Where $$B_n^2 = \textbf{D}S_n = \sum_{k=1}^n \sigma^2_k$$ then the
sequence will satisfy the central limit theorem.

## Answers to Problems

### 

$$\begin{aligned}
    a- \epsilon \leq \frac{1}{n}(\xi_1 +\xi_2 + \cdots + \xi_n) \leq a + \epsilon \\
    a- \epsilon \leq \frac{1}{n}\sum_{k=1}^n \xi_i \leq a + \epsilon \\
    - \epsilon \leq \frac{1}{n}\sum_{k=1}^n \xi_i -a \leq \epsilon \\
    \left| \frac{1}{n}\sum_{k=1}^n \xi_i -a  \right| < \epsilon\end{aligned}$$
And we know this happens with probability $$P_n = 1- \delta$$
$$\textbf{P} \left[ \left| \frac{1}{n}\sum_{k=1}^n \xi_i -a  \right| < \epsilon  \right] = 1 - \delta$$
$$\lim_{n\rightarrow \infty} \textbf{P} \left[ \left| \frac{1}{n}\sum_{k=1}^n \xi_i -a  \right| < \epsilon  \right] = 1$$
**Answer not verified**

### 

Since we know the mean, we most certainly can use the equation to
estimate the variance: look what it's doing. It's taking the sum of the
square of the distances from the known mean and dividing by the number
of trials: the classic standard deviation calculation. **Answer not
verified**

### 

We need to start with some calculations on the distribution:
$$\begin{aligned}
    E\xi = \int_0^{\infty} x \frac{x^m}{m!}e^{-x} dx = m+1 \\
    E\xi^2 = \int_0^{\infty} x^2 \frac{x^m}{m!}e^{-x} dx = (m+2)(m+1)\end{aligned}$$

$$P \{  | \xi - a | > \epsilon  \} \leq \frac{1}{\epsilon^2}\textbf{E}(\xi - a)^2$$
clearly we can manipulate it in this manner:
$$-P \{  | \xi - a | > \epsilon  \} \geq  - \frac{1}{\epsilon^2}\textbf{E}(\xi - a)^2$$
$$1-P \{  | \xi - a | > \epsilon  \} \geq 1 - \frac{1}{\epsilon^2}\textbf{E}(\xi - a)^2$$
$$P \{  | \xi - a | < \epsilon  \} \geq 1 - \frac{1}{\epsilon^2}\textbf{E}(\xi - a)^2$$
let $$\epsilon = a = m+1$$
$$P \{ 0 \leq \xi  \leq 2(m+1)  \} \leq 1 - \frac{1}{(m+1)^2}[(m+2)(m+1) - (m+1)^2]$$
$$P \{ 0 \leq \xi  \leq 2(m+1)  \} \leq \frac{m}{m+1}$$

**Answer not verified**

### 

Under these circumstances, we expect 500 A and the standard deviation is
$$\sqrt{250}= 5 \sqrt{10} \approx 15$$ therefore the $$\pm 100$$ from the
mean is more than $$\pm 6 \sigma$$ meaning there is indeed far more that
.97 probability of seeing the mean between these two values.

**Answer not verified**

### 

$$\begin{aligned}
    F_{\xi}(z) = \frac{z +z^2 + z^3 + z^4 + z^5 +z^6}{6}\end{aligned}$$

**Answer not verified**

### 

$$\begin{aligned}
    F'_{\xi}(z) = \frac{1+ 2z + 3z^2 + 4z^3 + 5z^4 + 6z^5}{6}\end{aligned}$$
$$\begin{aligned}
    F'_{\xi}(1) = \frac{21}{6} = \frac{7}{2} = \textbf{E}\xi\end{aligned}$$
$$\begin{aligned}
    F''_{\xi}(z) = \frac{2+ 6z + 12z^2 + 20z^3 + 30z^4 }{6}\end{aligned}$$
$$\begin{aligned}
    F''_{\xi}(1) = \frac{70}{6} = \frac{35}{3}  \end{aligned}$$
$$\begin{aligned}
    \sigma^2 = F''_{\xi}(1) + F'_{\xi}(1) + [F'_{\xi}(1)]^2  = \frac{35}{3} + \frac{7}{2} - \frac{49}{4}  = \frac{35}{12}\end{aligned}$$
**Answer verified**

### 

In order to do this problem with 6.6, we need the generating function
$$\begin{aligned}
    F_{\xi}(z) = \sum_{k=0}^{\infty} \frac{ a^k }{k!}e^{-a} z^k = e^{a(z-1)}\end{aligned}$$
$$\begin{aligned}
    F'_{\xi}(z) = a e^{a (z-1)} \\
    F'_{\xi}(1) = a e^{0}  = a = \textbf{E}\xi\\\end{aligned}$$
$$\begin{aligned}
    F''_{\xi}(z) = a^2 e^{a (z-1)} \\
    F''_{\xi}(1) = a^2\end{aligned}$$ $$\begin{aligned}
    \sigma^2 = F''_{\xi}(1) + F'_{\xi}(1) - [F'_{\xi}(1)]^2  = a^2 + a - a^2 = a\end{aligned}$$

### 

In order to do this problem with 6.6, we need the generating function
$$\begin{aligned}
    F_{\xi}(z) = \sum_{k=0}^{\infty} \frac{ a^k }{(1+a)^{k+1}} z^k = \frac{-a-1}{(a+1) (a z-a-1)}\end{aligned}$$
$$\begin{aligned}
    F'_{\xi}(z) = \frac{(a+1) a}{(a+1) (a z-a-1)^2} = \frac{a}{(a z-a-1)^2} \\
    F'_{\xi}(1) = a =  \textbf{E}\xi\\\end{aligned}$$ $$\begin{aligned}
    F''_{\xi}(z) = \frac{-2 a^2}{(a z-a-1)^3} \\
    F''_{\xi}(1) = 2 a^2\end{aligned}$$ $$\begin{aligned}
    \sigma^2 = F''_{\xi}(1) + F'_{\xi}(1) - [F'_{\xi}(1)]^2  = 2 a^2 + a - a^2 = a^2 - a = \textbf{D}\xi\end{aligned}$$

**Answer not verified**

### 

These two distributions have generating functions: $$\begin{aligned}
    F_{\xi_1} = e^{a(z-1)} \\
    F_{\xi_2} = e^{a'(z-1)}\end{aligned}$$ which means that $$\eta$$ has
this generating function: $$\begin{aligned}
    F_{\eta} =F_{\xi_1}F_{\xi_2} = e^{a(z-1)}e^{a'(z-1)} = e^{(a'+ a)(z-1)}  \\\end{aligned}$$
Which, according to theorem 6.2 means there that $$\eta$$ is a Poisson
distribution with mean $$a+a'$$. **Answer not verified**

### 

For any given experiment we can define an individual generating
function: $$F_i(z) = q_i + z p_i$$ meaning that for the entire function
$$\begin{aligned}
    F(z) = \prod_{i=1}^n F_i(z) = \prod_{i=1}^n (q_i + z p_i)\end{aligned}$$
The trick that Feller uses at this point is to take the log
$$\begin{aligned}
    \log F(z) = \sum_{i=1}^n \log F_i(z) = \sum_{i=1}^n \log (q_i + z p_i) \\
    \log F(z) = \sum_{i=1}^n \log (1 - p_i + z p_i) = \sum_{i=1}^n \log (1 - p_i( z -1))\end{aligned}$$
We were, however, told that the largest probability goes to zero so we
take the taylor series of each term, $$\log (1-x) \approx - x$$
$$\begin{aligned}
    \lim_{n \rightarrow \infty} \log F(z) =  \sum_{i=1}^n \log (1 - p_i( z -1)) = \sum_{i=1}^n  p_i( 1 - z ) = - \lambda (z - 1) \\
    \lim_{n \rightarrow \infty} F(z) = e^{- \lambda (z - 1)}\end{aligned}$$
And, according to theorem 6.2, the probability distribution is
Poissonian. **Answer verified**

### 

$$\begin{aligned}
    p_{\xi}(x) = \frac{1}{2} e^{-|x|} \\
    f_{\xi}(t) = \int_{-\infty}^{\infty} \frac{1}{2} e^{ i x t } e^{-|x|} dx \\
    f_{\xi}(t) = \frac{1}{t^2+1}\end{aligned}$$

**Answer verified**

### 

$$\begin{aligned}
    f_{\xi}(t) = \frac{1}{t^2+1} \\
    f'_{\xi}(t) =  -\frac{2 t}{\left(t^2+1\right)^2} = i \textbf{E}\xi \\
    f'_{\xi}(0) =  i \textbf{E}\xi = 0 \\
    f''_{\xi}(t) = \frac{8 t^2}{\left(t^2+1\right)^3}-\frac{2}{\left(t^2+1\right)^2} \\
    f''_{\xi}(0) = - \sigma^2 = -2\end{aligned}$$

**Answer verified**

### 

For a uniform distribution $$\begin{aligned}
    p(x) = \frac{1}{b-a}\end{aligned}$$ Therefore the characteristic is:
For a uniform distribution $$\begin{aligned}
    f(t) = \frac{1}{b-a} \int_a^b e^{i x t} dx = \frac{i \left(e^{i a t}-e^{i b t}\right)}{t(b-a)}\end{aligned}$$
**Answer verified**

### 

$$\begin{aligned}
    f_{\xi} (t) = e^{-a|t|} \\
    p(x) = \frac{1}{2 \pi} \inf_{\infty}^{\infty} e^{-a|t|} e^{- i x t} \\
    p(x) = \frac{a}{\pi(a^2+x^2)}\end{aligned}$$ **Answer verified**

### 

Looking at the characteristic, there is clearly going to be a
discontinuity in the deirvative at $$t=0$$, meaning that the derivative
does not exist at that point. This jibes well with what we've done
before because we proved in problem 43 on page 24 that the probability
distribution this characteristic produces does not have a mean or a
standard deviation.

**Answer not verified**

### 

For the single die, $$E\xi = 3.5$$ and $$D\xi = \frac{35}{12}$$. The
distribution for $$n$$ dice rolls is binomial but in the limit of large
$$n$$, it approximates a normal distribution with $$E\xi = 3.5n$$ and
$$D\xi = n \frac{35}{12}$$ $$\begin{aligned}
    P\{ 3450 \leq x \leq 3550  \} = \int_{3450}^{3550} \sqrt{\frac{1}{\frac{70 \pi \cdot 10^3}{12}}}e^{-\frac{(x-3500)^2}{\frac{70 \cdot 10^3}{12}}} dx \\
    = 0.645461  \\\end{aligned}$$ **Answer not verified**

### 

We want to prove here that the distribution satisfies the Lyapunov
condition:
$$\lim_{n\rightarrow \infty} \frac{1}{B_n^3} \sum_{k=1}^n \textbf{E}|\xi_k - a_k|^3 = 0$$
Where $$B_n^2 = \textbf{D}S_n = \sum_{k=1}^n \sigma^2_k$$

The machinations of which are left to the reader\... sorry!

**Answer not verified**
