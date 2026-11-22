"""Generated retain proxy of the unlearning task (team D).

The original model was trained from scratch on a private synthetic (text-to-image) dataset of
about 13 000 images over 100 classes; no public copy exists (see the team D report). This package
rebuilds a stand-in with a public text-to-image model: a prompt plan (`plan`), a GPU generator that
also classifies every image with the original (`generate`) and the analysis of what the original
makes of the generated images (`analyze`).
"""
