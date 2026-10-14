# Experiment C — Normal training set size

Question: how many normal images are required before detection quality stabilizes?

Train on 25%, 50%, 75%, and 100% of the normal training images. Repeat the draw
when compute allows, and keep the official test ids fixed. Subsets never include
anomalous images.

Wave 8 records the seeded draw of those fractions and repeats. It does not
retrain. The split loader still checksums every normal training file and
rejects a fit set that is not the recorded complement of validation, so a
smaller fit manifest cannot be persisted yet. The report states that block
and does not invent AUROC. Subsets never include anomalous or official test ids.
