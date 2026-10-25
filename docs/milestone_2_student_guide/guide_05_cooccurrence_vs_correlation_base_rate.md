# Co-Occurrence vs. Correlation: The Base-Rate Effect
*(Milestone 2 · Question 5 Study Guide)*

---

## 1. The Core Concept: The Base-Rate Fallacy in Defect Associations

In Question 4, you analyzed the raw co-occurrence counts of industrial surface defects. You likely observed that certain defect pairs appear together on hundreds of image strips.

When engineers observe a high co-occurrence count, a common intuition is to assume:
> *"These two defects must be strongly physically coupled; when one occurs, the other is likely triggered by the same machine fault."*

However, in statistics, this is known as the **Base-Rate Fallacy**. A high raw co-occurrence count does **not** necessarily indicate strong statistical dependence. If one or both defect classes have a high baseline marginal frequency across the factory, they will inevitably coincide on many images purely by random chance.

---

## 2. Independence vs. Normalized Correlation

To determine whether two defects genuinely co-occur more often than expected by chance, we must normalize for their individual marginal frequencies.

### Expected Co-Occurrence Under Independence
If defect $i$ and defect $j$ occur independently, their expected joint probability is $P(i \land j) = P(i) \times P(j)$. Across $N = 11{,}393$ images, the expected number of co-occurring images is:

$$\mathbb{E}[C_{ij}] = N \times \left(\frac{C_{ii}}{N}\right) \times \left(\frac{C_{jj}}{N}\right) = \frac{C_{ii} \times C_{jj}}{N}$$

- **$C_{ij} \gg \mathbb{E}[C_{ij}]$:** Positive statistical association beyond chance.
- **$C_{ij} \approx \mathbb{E}[C_{ij}]$:** Base-rate artifact; co-occurrence is driven purely by marginal frequencies.
- **$C_{ij} \ll \mathbb{E}[C_{ij}]$:** Negative statistical association (mutual avoidance).

### Pearson Correlation Coefficient
To measure normalized linear association between binary presence vectors $\mathbf{y}_i, \mathbf{y}_j \in \{0, 1\}^N$, we compute the Pearson correlation coefficient $r \in [-1, +1]$:

$$r_{ij} = \frac{\operatorname{Cov}(\mathbf{y}_i, \mathbf{y}_j)}{\sigma_{\mathbf{y}_i} \sigma_{\mathbf{y}_j}}$$

A correlation coefficient near zero ($|r| < 0.1$) indicates that the two variables are practically uncorrelated, regardless of how large their raw co-occurrence count appears.

---

## 3. Guiding Hints for Question 5

To answer Question 5:

- **Hint 1 (Take the Top Pair from Question 4):** Identify the specific distinct defect pair that produced the highest raw co-occurrence count in Question 4.
- **Hint 2 (Compute the Pearson Correlation):** Calculate the Pearson correlation coefficient between the binary indicator vectors of these two defect classes across all 11,393 images. Round your result to 2 decimal places.
- **Hint 3 (Compare Observed vs. Expected):** Compare the observed co-occurrence count with the theoretical expectation under independence ($\mathbb{E}[C_{ij}] = \frac{C_{ii} C_{jj}}{N}$). Notice how the high marginal base rate of the dominant defect inflates the raw overlap count.
- **Hint 4 (Interpret the Result in 1–2 Sentences):** State whether the apparent strong association survives correlation normalization (`Yes` or `No`), and explain in 1–2 sentences why the correlation value reflects the base-rate effect.

---

## 4. Your Assignment Task

1. Open [Milestone 2 Assessment: Question 5](milestone-2.html#question-5-co-occurrence-vs-correlation-the-base-rate-effect).
2. Calculate the Pearson correlation coefficient (rounded to 2 decimal places) for the top co-occurring defect pair.
3. Determine whether the association sustains after normalization, and provide your 1–2 sentence explanation: `correlation_value, Yes/No`.
