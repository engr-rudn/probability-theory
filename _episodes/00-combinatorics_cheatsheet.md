---
title: "Regression models for machine learning"
questions:
- ""
objectives:
- ""
---

$$
\section{Asymptotics}
\subsection{Definitions}
\begin{align*}
f(n) \in O(g(n)) &\defarrow \exists c ~ \forall n \geq n_0 : |f(n)| \leq c|g(n)|\\
f(n) \in o(g(n)) &\defarrow \lim_{n \to \infty} \dfrac{f(n)}{g(n)} = 0 \\
f(n) \in \Omega(g(n)) &\defarrow g(n) \in O(f(n))\\
f(n) \in \omega(g(n)) &\defarrow \lim_{n \to \infty} g(n) \in o(f(n)) \\
f(n) \in \Theta(g(n)) &\defarrow f(n) \in O(g(n)) \wedge g(n) \in O(f(n)) \\
\end{align*}
\begin{align*}
\text{absolute error: }& X + O(n^{-k})\\
\text{relative error: }& X (1 + O(n^{-k}))
\end{align*}
$$
$$
\subsection{Basic approximations}
\begin{itemize}
\item Taylor polynoms 
\item \textbf{Stirling} $n! = \sqrt{2 \pi n} \left( \dfrac{n}{e} \right)^n \left(1 + \dfrac{1}{12n} - \dfrac{1}{288n^2} + O(n^{-3}) \right)$
\item $H_n = \ln n + \gamma + \dfrac{1}{2n} - \dfrac{1}{12n^2} + O(n^{-4})$
\item ${2n \choose n} = \dfrac{2^{2n}}{\sqrt{\pi n}} \left( 1 - \dfrac{1}{8n} + O\left(n^{-2}\right) \right)$
\end{itemize}
\subsection{Basic technics}
\begin{itemize}
  \item take away tail of Taylor expansion
  \item substitution
  \item if expression is too big to converge, take out bigger part and
  then apply Taylor expansion technics
  \item $\dfrac{1}{1-x} = 1 + O(x) \Longrightarrow \dfrac{1}{1 + O(n^{-1})} = 1 + O(n^{-1})$
  \item $f = e^{\ln f}$
  \item $[x] = x + O(1)$
  \item given precision limit, you can omit any part of expression with smaller magnitude (e.g. multiplication of two big sums)
  \item $\sum_{a \leq k < b} f(k) = \int_{a}^{b} f(x) dx + R$, where $R \leq \sum_{a \leq k < b} \max_{x \in [k, k+1)}{|f(x) - f(k)|}$. 
  If $f$ is monotonic, then $R \leq |f(b) - f(a)|$
  \item \textbf{[bootstrapping]} Find rough estimate for recurrence and plug it into recurrence to get better one
  \item \textbf{[dominant/tail]} separate sum into two parts and analyze them separately. Advantage is ability to 
  approximate tail part very loosely.
\end{itemize}

\subsection{\uppercase{Tail switching} method for destroying sums}
Given a sum $\sum_{k \in M} a_k(n)$
\begin{enumerate}
  \item separate sum into two disjoint ranges, \emph{dominant} $D_n$ and \emph{tail} $T_n$ 
  (i.e. $D_n \cup T_n = M$, $D_n \cap T_n = \varnothing$).
  \item find asymptotic estimate $a_k(n) = b_k(n) + O(c_k(n))$ for $k \in D_n$
  \item Let $$A(n) :=\sum_{k\in T_n} a_k(n)$$ $$B(n) := \sum_{k \in T_n} b_k(n)$$ $$C(n) := \sum_{k \in D_n} |c_k(n)|$$
  and prove all three are small.
  \item \begin{align*} 
  &\sum_{k \in D_n \cup T_n} a_k(n) = \\ 
  =&\sum_{k \in D_n \cup T_n} b_k(n) + O(A(n)) + O(B(n)) + O(C(n))
  \end{align*}
\end{enumerate}
$$