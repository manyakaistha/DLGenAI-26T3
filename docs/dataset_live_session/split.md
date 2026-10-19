# The 80/20 split: practice on some images, check on others

Before making batches, decide which images the model may learn from and which images you will use to check its progress. This is the **train–validation split**.

## Why do we need a split?

Imagine preparing for an exam. If you practise ten questions and then test yourself on those same ten questions, a high score may show that you remember the answers. It does not tell you how well you will answer a new question.

A segmentation model can do something similar: become very good at the training images without learning patterns that work on other images. We call this **overfitting**. To check whether learning transfers to images outside training, we keep some labelled images aside.

- **Training set:** images and masks used to calculate training loss and update model weights.
- **Validation set:** different images and masks used to measure progress and compare choices. Validation masks let us check the predictions, but are not used for gradient updates.

We want the model to recognize cloud patterns in a new image, not merely remember where a cloud appeared in an old image.

## What does 80/20 mean?

With 100 labelled images, an 80/20 split puts 80 into training and 20 into validation. Each image goes into one group, together with all its labels.

Our small demonstration has **24 labelled training images**. We calculate:

```python
n_train = int(0.8 * 24)  # 19
n_val = 24 - n_train    # 5
```

`0.8 * 24` is 19.2. `int` rounds this positive number down to 19, and the five remaining images go to validation. So our actual proportions are approximately 79.2% and 20.8%. Whole images cannot be divided into fractional samples.

80/20 is a useful starting choice, not a rule that is always best. More training images give the model more examples to learn from. More validation images usually make our estimate less sensitive to a few unusual examples. With just five validation images, a result can change a lot because of one difficult image; this toy split teaches mechanics, not a reliable performance estimate.

## Training, validation and test are different jobs

The folder named `train_images/` contains the labelled images available to us. We split those into our own training and validation groups. We do not need to move files into new folders.

The separate `test_images/` folder in our demo has no training labels. It is not the 20% validation group. In a competition, test labels are commonly hidden from participants.

During training, we may look at validation results to choose an epoch, augmentation strength or model setting. Those choices use information from validation, so validation is not an untouched final test. A separate test set, when available with suitable evaluation, checks the final chosen approach.

## Split images, not CSV rows

Recall that one image has four annotation rows. For example:

```text
demo_001.jpg_Fish
demo_001.jpg_Flower
demo_001.jpg_Gravel
demo_001.jpg_Sugar
```

If we split these rows independently, Fish might go into training while Flower goes into validation. Both groups would still contain the same photograph. A model that has already seen that image is not being checked on an unseen image. This is **data leakage**: information crosses the boundary we intended to keep.

Our Dataset already groups all four labels into one image sample. Splitting its image indices keeps all four mask channels together. If multiple images are near-duplicates, adjacent frames, or crops of the same original, split by their shared source group as well; different filenames alone do not guarantee independence.

## Step by step: create the split

Run this after creating `dataset = CloudDataset(ROOT)` in the companion notebook. The imports `torch` and `Subset` are already in the setup cell.

```python
# 1. Count image samples, not annotation rows.
n_images = len(dataset)

# 2. Make a repeatable random ordering of the image indices.
split_generator = torch.Generator().manual_seed(42)
order = torch.randperm(n_images, generator=split_generator).tolist()

# 3. Give approximately 80% to training and the rest to validation.
cut = int(0.8 * n_images)
train_indices = order[:cut]
val_indices = order[cut:]

# 4. Verify that the groups are separate and cover the whole dataset.
assert len(train_indices) > 0 and len(val_indices) > 0
assert set(train_indices).isdisjoint(val_indices)
assert set(train_indices) | set(val_indices) == set(range(n_images))

# 5. Create views of the selected samples; files stay in their folders.
train_data = Subset(dataset, train_indices)
val_data = Subset(dataset, val_indices)

print("Training images:", len(train_data))  # 19
print("Validation images:", len(val_data))  # 5
```

### What each statement does

`len(dataset)` asks the Dataset's `__len__` method for its number of image samples.

`torch.Generator()` creates a random-number generator. `manual_seed(42)` sets its starting state. The number 42 is simply a chosen seed; it does not make a split more accurate or guarantee balanced classes.

`torch.randperm(n_images)` creates a permutation of the integers from 0 to `n_images - 1`. Each index appears once. The generator makes the ordering reproducible for the same setup and Dataset index order. `.tolist()` converts the tensor into an ordinary Python list.

`order[:cut]` selects positions before `cut`. `order[cut:]` selects positions from `cut` to the end. These are two slices of the same permutation, so they cannot share an index.

`set(...)` lets us compare membership without caring about order. `isdisjoint` checks that the two groups share nothing. The `|` symbol makes their union; comparing it with `set(range(n_images))` checks that no image was left out. The first assertion catches datasets too small for this particular split rule.

`Subset(dataset, train_indices)` stores a connection to the Dataset and a list of allowed indices. It does not copy or load all images. For example, `train_data[0]` asks the original Dataset for whichever image index appears first in `train_indices`.

## Keep the split fixed when comparing experiments

Suppose experiment A and experiment B use different validation images. A better score for B could mean B is better, or simply that its validation images are easier. Keep the same split when comparing changes.

The seed reproduces membership only while the input image list and ordering also stay the same. Save the train and validation filenames for a lasting experiment record, especially if the dataset may change. Do not keep changing seeds until you find a flattering validation score.

## Random does not mean balanced

Imagine a class appears in only two images. A random split can put both into training, leaving none of that class in validation. We then cannot assess performance on positive examples of that class.

After splitting, count how many images contain each label in each group. **Stratification** tries to preserve label representation. In a multi-label task, an image can contain several labels; multi-label stratification considers those patterns rather than treating each annotation row as a separate sample. Very rare label combinations may be too small to distribute proportionally, so inspect the resulting split rather than assuming a method guarantees coverage.

For today's toy lab, use the simple seeded split to understand the mechanism. Before trusting real validation results, audit class coverage and any shared source groups.

## Split first, then apply training augmentation

A flipped version of a training image is still derived from that same image. If an original enters training and its flipped copy enters validation, the two groups are not independent.

First assign original images to groups. Then give only the training Dataset the random transform:

```python
# augment_pair is defined in the augmentation lesson.
training_dataset = CloudDataset(ROOT, transform=augment_pair)
validation_dataset = CloudDataset(ROOT, transform=None)

# Both objects read the same annotation index in the same order.
train_data = Subset(training_dataset, train_indices)
val_data = Subset(validation_dataset, val_indices)
```

Both groups still receive deterministic preprocessing: RLE decoding, resizing and tensor conversion. Validation simply skips random training augmentation. If you later learn preprocessing statistics from the data, fit them on training images, then apply those fixed values to validation.

## A split is not a batch

A split assigns image membership for the experiment. A DataLoader groups the images from one set into smaller batches for processing.

For this lab, 19 training images with batch size 4 give batches of 4, 4, 4, 4 and 3. Five validation images give batches of 4 and 1. Shuffling training batches changes presentation order; it does not transfer validation images into training. Keep the group membership fixed between epochs.

## Activity 1: predict the counts

You have 50 labelled images, with four annotation rows each. With our 80/20 rule, how many training images, validation images, and annotation rows belong to each group?

<details>
<summary>Reveal the answer</summary>

There are 40 training images and 10 validation images. Their annotations contain 160 and 40 rows respectively. We split 50 image identities, not 200 independent samples.

</details>

## Activity 2: spot the leakage

A student says: “I created three augmented copies of each image, shuffled all copies, and split them 80/20.” What could go wrong, and how would you repair it?

<details>
<summary>Reveal the reasoning</summary>

Different copies of the same original can enter both groups. Split original image identities first, then augment only the training group's samples on demand. Keep related crops or frames in the same source group where necessary.

</details>

## Activity 3: interpret the scores

Training performance keeps improving while validation performance stops improving and then gets worse. Does the training score alone prove that the model is getting better at unseen images?

<details>
<summary>Reveal the reasoning</summary>

No. This pattern can indicate overfitting. Use the held-out validation evidence to investigate rather than assuming that a better training result means better generalization. Also check that the validation set is representative and that evaluation is correct.

</details>

## Exit check

Explain why the five validation images remain in `train_images/`, why all four labels for a photograph stay together, and why `shuffle=True` is not a replacement for a split.

Continue to [DataLoader and batching](dataloader.md), or return to the [session overview](README.md).
