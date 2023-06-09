---
title:  Exams

questions:
- ""
objectives:
- Old exams questions and answers
---

## Mid-Semester Exam Ac.Year 2022/23
> ### A coin is flipped 10 times, and the sequence is recorded.
> ### a) How many sequences are possible?
> >
> > ## Solution
> >
> >To determine the number of possible sequences when flipping a coin 10 times, we need to consider that each flip has two possible outcomes: either a head (H) or a tail (T). Therefore, for each coin flip, there are 2 possibilities.\
> >Since there are 10 coin flips in total, we can calculate the number of possible sequences by raising 2 to the power of 10:\
> >Number of possible sequences = $$2^{10} = 1,024$$\
> >So, there are 1,024 possible sequences when flipping a coin 10 times.
>{: .solution}
{: .challenge}
> ### b) How many sequences have exactly 7 heads?
> > 
> > ## Solution
> >
> >To find the number of sequences that have exactly 7 heads, we need to consider the combination of choosing 7 out of the 10 flips to be heads. The remaining 3 flips will automatically be tails since there are only two options (H or T) for each flip.\
> >The number of sequences with exactly 7 heads can be calculated using the binomial coefficient, which is given by the formula:\
> >$$\binom{n}{k} = \frac{n!}{k!(n-k)!}$$\
> >where $$n$$ is the total number of flips (10 in this case), and $$k$$ is the number of heads (7 in this case).\
> >Using the formula:\
> >$$\binom{10}{7} = \frac{10!}{7!(10-7)!} = \frac{10!}{7!3!} = \frac{(10 \cdot 9 \cdot 8)}{(3 \cdot 2 \cdot 1)} = 120$$\
> >Therefore, there are 120 sequences that have exactly 7 heads when flipping a coin 10 times.
>{: .solution}
{: .challenge}
> ### A wooden cube with painted faces is sawed up into 512 little cubes, all of the same size. The little cubes are then mixed up, and one is chosen at random. What is the probability of it having just 2 painted faces?
> > 
> > ## Solution
> >
> >The wooden cube is made of $$8^3$$ cubes implying a $$8x8x8$$ cube.\
The cubes that have two faces painted will be the edges which are not on a
corner.\
Thus, since there are 12 edges of 8 cubes each, 6 of which are
not corners, that implies we have 72 edges.\
72 thus becomes our number of desireable outcomes and 512 has always been the total number of
outcomes:\
$$P=\frac{72}{512} = 0.14\label{answer1.2}$$
> > <!-- A wooden cube with painted faces is sawed up into 512 little cubes, all of the same size. The little cubes are then mixed up, and one is chosen at random. We want to find the probability of selecting a little cube that has exactly 2 painted faces.\
> > Let's first determine the total number of little cubes. The wooden cube is divided into 8 layers, each containing 8 little cubes in both the horizontal and vertical directions, resulting in a total of 8 * 8 = 64 little cubes per layer. Since there are 8 layers in total, the number of little cubes is 8 * 64 = 512.\
> > Now let's consider the number of little cubes with exactly 2 painted faces. Each layer of the wooden cube contributes 4 little cubes with 2 painted faces (the corners of the layer). Since there are 8 layers in total, the number of little cubes with exactly 2 painted faces is 8 * 4 = 32.\
> > Therefore, the probability of selecting a little cube with exactly 2 painted faces is given by:\
> > Probability = $$\frac{\text{Number of little cubes with exactly 2 painted faces}}{\text{Total number of little cubes}} = \frac{32}{512} = \frac{1}{16}$$\
> > Hence, the probability of selecting a little cube with just 2 painted faces is 0.0625 or 6.25%. -->
>{: .solution}
{: .challenge}
> ### What is the probability of getting two tails when two coins are tossed?
> >
> > ## Solution
> >
> >The probability of getting two tails when two coins are tossed can be determined by considering all possible outcomes and counting the favorable outcomes. Let's calculate it:\
> >When two coins are tossed, the possible outcomes for each coin are either heads (H) or tails (T). Therefore, the sample space consists of four possible outcomes: {HH, HT, TH, TT}, where HH represents two heads, HT represents one head and one tail, TH represents one tail and one head, and TT represents two tails.\
> >Out of these four possible outcomes, there is only one outcome that corresponds to getting two tails (TT). Thus, the favorable outcome is {TT}.\
> >Therefore, the probability of getting two tails when two coins are tossed is given by:\
> >Probability = $$\frac{\text{Number of favorable outcomes}}{\text{Total number of possible outcomes}} = \frac{1}{4} = 0.25 = 25%$$\
> >Hence, the probability of getting two tails when two coins are tossed is 0.25 or 25%.
>{: .solution}
{: .challenge}
> ### 8 fair dice are tossed independently. Find the probability that at least two “6” appears.
> >
> > ## Solution
> >
> >To find the probability that at least two "6" appear when 8 fair dice are tossed independently, we can calculate the probability of the complementary event, which is the probability that fewer than two "6" appear, and subtract it from 1.\
> >Let's calculate the probability of getting fewer than two "6" in 8 tosses:\
> >The probability of getting a "6" on a single die is $$\frac{1}{6}$$, and the probability of not getting a "6" is $$\frac{5}{6}$$.\
> >Probability of getting no "6" in 8 tosses:\
> >$$\left(\frac{5}{6}\right)^8$$\
> >Probability of getting exactly one "6" in 8 tosses:\
> >$$\binom{8}{1} \cdot \left(\frac{1}{6}\right) \cdot \left(\frac{5}{6}\right)^7$$\
> >The "$$\binom{8}{1}$$" term represents the number of ways to choose one position out of the eight tosses to have a "6".\
> >Now, let's calculate the probability of fewer than two "6" appearing:\
> >Probability of fewer than two "6" = Probability of no "6" + Probability of exactly one "6"\
> >Probability of fewer than two "6" = $$\left(\frac{5}{6}\right)^8 + \binom{8}{1} \cdot \left(\frac{1}{6}\right) \cdot \left(\frac{5}{6}\right)^7$$\
> >Finally, we subtract the probability of fewer than two "6" from 1 to find the probability of at least two "6" appearing:\
> >Probability of at least two "6" = 1 - Probability of fewer than two "6"\
> >Probability of at least two "6" = 1 - $$\left[\left(\frac{5}{6}\right)^8 + \binom{8}{1} \cdot \left(\frac{1}{6}\right) \cdot \left(\frac{5}{6}\right)^7\right]$$\
> >Calculating this expression gives us the probability of at least two "6" appearing when 8 fair dice are tossed independently.\
> >The result in decimal format is approximately 0.33519, or 33.519%
<!-- > >To find the probability that at least two "6" appears when 8 fair dice are tossed independently, we can use the concept of complementary probability. We will calculate the probability that no "6" appears and then subtract it from 1 to get the desired probability.\
> >The probability of getting no "6" on a single die is 5/6, as there are 5 other possible outcomes (1, 2, 3, 4, 5) out of 6.\
> >Since the dice are tossed independently, the probability of getting no "6" on all 8 dice is:\
> >$$(5/6) * (5/6) * (5/6) * (5/6) * (5/6) * (5/6) * (5/6) * (5/6) = (5/6)^8$$\
> >Now, let's find the complementary probability of getting at least two "6" by subtracting the probability of no "6" from 1:\
> >Probability of at least two $$"6" = 1 - (5/6)^8$$\
> >Calculating this expression, we can find the desired probability.\
> >The probability of at least two "6" appearing when 8 fair dice are tossed independently is approximately 0.623. -->
>{: .solution}
{: .challenge}
> ### A batch of 7 manufactured items contains 2 defective items. Suppose 4 items are selected at random from the batch. What is the probability that 1 of these items are defective?
> >
> > ## Solution
> > 
> >There are $$\binom{7}{4}$$ possible ways to chose $$4$$ different items from
the population of $$7$$ items which will be our denominator. Now we need
to know how many of those possibilities have $$1$$ bad ones in them for
our numerator. If there's $$2$$ total defective ones, then there are
$$\binom{2}{1}$$. Therefore the probability P is:\
> > $$P = \frac{\binom{2}{1}}{\binom{7}{4}}$$\
>We simplify the binomial coefficients using their definition in terms of factorials.\
> > Using the formula $$\binom{n}{k} = \frac{n!}{k!(n-k)!}$$, we have:\
> >$$\binom{2}{1} = \frac{2!}{1!(2-1)!} = \frac{2}{1} = 2$$\
> >$$\binom{7}{4} = \frac{7!}{4!(7-4)!} = \frac{7!}{4!3!} = \frac{7 \times 6 \times 5}{3 \times 2 \times 1} = \frac{7 \times 6 \times 5}{6} = 7 \times 5 = 35$$\
> >Substituting these values back into the expression $$P$$, we get:\
> >$$P = \frac{\binom{2}{1}}{\binom{7}{4}} = \frac{2}{35}$$\
> >Therefore, the value of $$P$$ is $$\frac{2}{35}$$.
>{: .solution}
{: .challenge}
> ### 10 books are placed in random order on a bookshelf. Find the probability of 4 given books being side by side.
> >
> > ## Solution
> >
> >To find the probability of 4 given books being side by side when 10 books are placed in a random order on a bookshelf, we calculate the total number of possible arrangements and the number of arrangements where the 4 given books are together.\
> >Total number of possible arrangements:\
> >Since there are 10 books, the total number of possible arrangements is given by the factorial of 10, denoted as $$10!$$.\
> >Arrangements where the 4 given books are together:\
> >Consider the 4 given books as a single entity. So, we have 7 remaining books and the group of 4 given books, which can be arranged in $$(7 + 1)!$$ ways. However, within the group of 4 given books, they can be arranged in $$4!$$ ways. Therefore, the number of arrangements where the 4 given books are together is $$(6 + 1)! \times 4!$$.\
> >Now, we can calculate the probability by dividing the number of favorable arrangements (where the 4 given books are together) by the total number of possible arrangements:\
> >$$\left[\textbf{Probability} = \frac{\textbf{Number of arrangements with 4 given books together}}{\textbf{Total number of possible arrangements}} = \frac{7! \times 4!}{10!}\right]$$\
> >Therefore, the probability of 4 given books being side by side is\
> > $$\frac{7! \times 4!}{10!}$$.
{: .solution}
{: .challenge}
> ### An urn contains a total of N balls, some black and some white. Samples are drawn from the urn, $$m$$ balls at a time $$(m < N)$$. After drawing each sample,the black balls are returned to the urn, while the white balls are replaced by black balls and then returned to the urn. If the number of white balls in the um is $$i$$, we say that the "system" is in the state $$e$$. 
> Now, let $$N = 8, m =4,$$ and suppose there are initially $$5$$ white balls in the urn. What is the probability that no white balls are left after $$2$$ drawings (of $$4$$ balls each)?
> 
> > ## Solution
> >
> >To find the probability that no white balls are left after two drawings of three balls each, we need to analyze the system states and calculate the probabilities associated with each state.\
> > In this problem, the system states represent the number of white balls in the urn after each drawing. Let's consider the possible system states after each drawing:\
> >$$\text{State } e1: \text{ 5 white balls (initial state)}$$ \
> >$$\text{State } e2: \text{ 4 white balls (after the first drawing)}$$ \
> >$$\text{State } e3: \text{ 3 white balls (after the second drawing)}$$
> >We need to calculate the probability of transitioning from state $$e1$$ to state $$e3$$ in two drawings.\
> >To calculate the probability, we can consider the number of ways to select the balls from the urn and calculate the desired probability.\
> >In the first drawing:\
> >The probability of selecting a white ball is $$\frac{5}{7}$$ since there are 5 white balls and 7 total balls.\
> >The probability of selecting a black ball is $$\frac{2}{7}$$ since there are 2 black balls and 7 total balls.\
> >After the first drawing, the urn contains 5 black balls (the returned white ball is replaced by a black ball) and 2 white balls (since the black ball is returned to the urn).\
> >In the second drawing:\
> >The probability of selecting a white ball is $$\frac{2}{7}$$ since there are 2 white balls and 7 total balls.\
> >The probability of selecting a black ball is $$\frac{5}{7}$$ since there are 5 black balls and 7 total balls.\
> >Now, let's calculate the probability of transitioning from $$e1$$ to $$e3$$ in two drawings:\
> >$$P(e1 \text{ to } e3 \text{ in 2 drawings}) = P(e1 \text{ to } e2 \text{ in 1st drawing}) \times P(e2 \text{ to } e3 \text{ in 2nd drawing})$$\
> >$$P(e1 \text{ to } e2 \text{ in 1st drawing}) = \text{Probability of selecting 3 black balls in the first drawing} = \frac{2}{7} \times \frac{2}{7} \times \frac{2}{7} = \frac{8}{343}$$\
> >$$P(e2 \text{ to } e3 \text{ in 2nd drawing}) = \text{Probability of selecting 3 black balls in the second drawing} = \frac{5}{7} \times \frac{5}{7} \times \frac{5}{7} = \frac{125}{343}$$\
> >$$P(e1 \text{ to } e3 \text{ in 2 drawings}) = P(e1 \text{ to } e2 \text{ in 1st drawing}) \times P(e2 \text{ to } e3 \text{ in 2nd drawing}) = \frac{8}{343} \times \frac{125}{343} = \frac{1000}{16807}$$\
> >Therefore, the probability that no white balls are left after two drawings of three balls each is $$\frac{1000}{16807}$$.
>{: .solution}
{: .challenge}
> ### What will be the probability of getting odd numbers if a dice is thrown?
> >
> >## Solution
> >
> >When a fair six-sided dice is thrown, there are six possible outcomes: $$\{1, 2, 3, 4, 5, 6\}$$, representing the numbers on the dice's faces.\
> >Out of these six possible outcomes, three outcomes are odd numbers: $$\{1, 3, 5\}$$, while the remaining three outcomes are even numbers: $$\{2, 4, 6\}$$\
> >Therefore, the probability of getting an odd number when a dice is thrown can be calculated by dividing the number of favorable outcomes (odd numbers) by the total number of possible outcomes:\
> >Probability of getting an odd number $$= \frac{\text{Number of favorable outcomes}}{\text{Total number of possible outcomes}}$$\
> >Probability of getting an odd number $$= \frac{3}{6} = \frac{1}{2} = 0.5 = 50\%$$\
> >Hence, the probability of getting an odd number when a dice is thrown is $$0.5$$ or $$50\%$$.
>{: .solution}
{: .challenge}
> ### What is the probability of being able to form a triangle from three segments chosen at random from five line segments of lengths $$2, 3, 6, 7,$$ and $$8$$?
> >
> > ## Solution
> >
> > To form a triangle, the sum of the lengths of any two sides must be greater than the length of the third side.
> >Let's consider the combinations of three line segments:
> >1. Choose the segments of lengths 2, 3, and 6:
> >The sum of the lengths of the segments 2 and 3 is 5, which is less than the length of segment 6. So this combination cannot form a triangle.
> >2. Choose the segments of lengths 2, 3, and 7:
> >The sum of the lengths of the segments 2 and 3 is 5, which is less than the length of segment 7. So this combination cannot form a triangle.
> >3. Choose the segments of lengths 2, 3, and 8:
> >The sum of the lengths of the segments 2 and 3 is 5, which is less than the length of segment 8. So this combination cannot form a triangle.
> >4. Choose the segments of lengths 2, 6, and 7:
> >The sum of the lengths of the segments 2 and 6 is 8, which is greater than the length of segment 7. The sum of the lengths of the segments 2 and 7 is 9, which is greater than the length of segment 6. <b>So this combination can form a triangle</b>.
> >5. Choose the segments of lengths 2, 6, and 8:
> >The sum of the lengths of the segments 2 and 6 is 8, which is equal to the length of segment 8. So this combination cannot form a triangle.
> >6. Choose the segments of lengths 2, 7, and 8:
> >The sum of the lengths of the segments 2 and 7 is 9, which is greater than the length of segment 8. The sum of the lengths of the segments 2 and 8 is 10, which is greater than the length of segment 7. <b>So this combination can form a triangle</b>.
> >7. Choose the segments of lengths 3, 6, and 7:
> >The sum of the lengths of the segments 3 and 6 is 9, which is greater than the length of segment 7. <b>So this combination can form a triangle</b>.
> >8. Choose the segments of lengths 3, 6, and 8:
> >The sum of the lengths of the segments 3 and 6 is 9, which is greater than the length of segment 8. <b>So this combination can form a triangle</b>.
> >9. Choose the segments of lengths 3, 7, and 8:
> >The sum of the lengths of the segments 3 and 7 is 10, which is greater than the length of segment 8. <b>So this combination can form a triangle</b>.
> >10. Choose the segments of lengths 6, 7, and 8:
> >The sum of the lengths of the segments 6 and 7 is 13, which is greater than the length of segment 8. <b>So this combination can form a triangle</b>.
> >Out of the 10 possible combinations, 6 of them satisfy the condition for triangle formation. Therefore, the probability of being able to form a triangle is:
> >Probability $$= \frac{\text{Number of favorable combinations}}{\text{Total number of possible combinations}}$$\
> >Probability $$= \frac{6}{10} = 0.6 = 60\%$$\
Hence, the probability of being able to form a triangle from three segments chosen at random from the given line segments is $$0.6$$ or $$60\%$$.
>{: .solution}
{: .challenge}