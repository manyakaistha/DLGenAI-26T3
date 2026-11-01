# Compound Loss Behavior on Defect vs. Clean Images
*(Milestone 3 · Question 7 Study Guide)*

## 1. Why Compare BCE and Dice?

BCE supplies local pixel-wise classification signals; Soft Dice couples pixels through an overlap ratio. A weighted sum can balance these signals, but its performance depends on reduction, weights, data and optimization. It is a hypothesis to test, not a theorem that the combination always wins.

The assessment fixes:
$$L=0.5L_{BCE}+0.5L_D.$$

Pass raw logits to `BCEWithLogitsLoss` and to the provided `SoftDiceLoss`. The latter applies sigmoid internally. Do not threshold predictions during training.

## 2. Define the Reduction Before Discussing a Clean Image

The starter's Dice flattens the **entire batch and all four channels**. The following all-empty derivation applies when every target in that reduction is zero, or when Dice is deliberately computed on one isolated clean image. It does not describe a clean sample's contribution inside a mixed positive/clean batch: positives elsewhere affect that sample's numerator and denominator derivatives.

Let $S=\sum_i p_i$, and let all $y_i=0$. With smoothing $\epsilon$:
$$L_D=1-\frac{\epsilon}{S+\epsilon}=\frac{S}{S+\epsilon}.$$

The loss is zero at $S=0$ and approaches one as predicted mass grows. For a **toy mass of 9** with $\epsilon=1$, the loss is $9/10=0.9$. This is a large loss value; it says nothing by itself about the size of a parameter update.

## 3. Follow the Derivative through Sigmoid

For a probability $p_j$:
$$\frac{\partial L_D}{\partial p_j}=\frac{\epsilon}{(S+\epsilon)^2}.$$

For its logit $z_j$, the sigmoid chain rule gives:
$$\frac{\partial L_D}{\partial z_j}=\frac{\epsilon p_j(1-p_j)}{(S+\epsilon)^2}.$$

As total mass increases, the shared denominator grows. A saturated positive probability also makes $p_j(1-p_j)$ small. Consequently a nearly maximal loss can coexist with weak logit gradients. For the toy mass $S=9$, $\epsilon=1$ and $p_j=0.5$, the logit derivative is $0.25/100=0.0025$.

As all probabilities approach zero, the probability derivative approaches $1/\epsilon$, whereas the **logit derivative** approaches zero through the sigmoid factor. There is no positive noise threshold where this differentiable loss stops working: for finite logits and positive smoothing, its all-empty logit derivative remains positive, though floating-point saturation can produce numerical zeros.

## 4. How BCE Contributes

With mean reduction over $N$ elements:
$$\frac{\partial L_{BCE}}{\partial z_j}=\frac{p_j-y_j}{N}.$$

On a clean target this is $p_j/N$. It does not contain Dice's shared probability-mass denominator or sigmoid derivative factor. A confidently wrong background prediction therefore still receives a BCE logit signal. Near $p_j=0$, BCE's signal also tends toward zero; it is not a promise of permanent margin enforcement.

BCE is convex as a function of individual logits, but a neural network trained with it is not generally convex in its parameters. The relative size of the BCE/Dice contributions depends on $N$, smoothing and reduction. Compare gradients or controlled validation runs rather than assuming equal coefficients imply equal effects.

## 5. A Small Autograd Check

```python
import torch
z = torch.tensor([0.0, 0.0], dtype=torch.float64, requires_grad=True)
p = z.sigmoid()
eps = 1.0
loss = 1 - eps / (p.sum() + eps)  # all-zero target
loss.backward()
expected = eps * p.detach() * (1 - p.detach()) / (p.detach().sum() + eps)**2
torch.testing.assert_close(z.grad, expected)
print(z.grad)
```

<details>
<summary>Practice: Is a high Dice loss evidence of exploding gradients?</summary>

No. The scalar loss and its derivative answer different questions. Inspect the derivative with respect to the quantity actually optimized: logits are further transformed through network parameters. The all-empty formula above shows why a large predicted mass can raise loss while suppressing the logit gradient.

</details>

## 6. Guiding Hints and Assignment Task

Read [Question 7](../../MILESTONE_3.md#question-7-compound-loss-behavior-on-defect-vs-clean-images). Derive the all-empty loss, distinguish probability and logit derivatives, and describe how they change with predicted mass and sigmoid saturation. Explain what signal BCE adds and one limitation of claiming the combination must always outperform Dice alone. State whether your argument concerns an all-empty reduction or a mixed batch.

Report `brief_gradient_explanation; brief_compound_justification` in your own words.
