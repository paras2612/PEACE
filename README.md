# PEACE
Code for our paper titled "PEACE: Cross-Platform Hate Speech Detection - A Causality-guided Framework"

Hate speech detection refers to the task of detecting hateful content
that aims at denigrating an individual or a group based on their religion, gender,
sexual orientation, or other characteristics. Due to the different policies of the
platforms, different groups of people express hate in different ways. Furthermore,
due to the lack of labeled data in some platforms it becomes challenging to
build hate speech detection models. To this end, we revisit if we can learn a
generalizable hate speech detection model for the cross platform setting, where
we train the model on the data from one (source) platform and generalize the
model across multiple (target) platforms. Existing generalization models rely
on linguistic cues or auxiliary information, making them biased towards certain
tags or certain kinds of words (e.g., abusive words) on the source platform and
thus not applicable to the target platforms. Inspired by social and psychological
theories, we endeavor to explore if there exist inherent causal cues that can be
leveraged to learn generalizable representations for detecting hate speech across
these distribution shifts. To this end, we propose a causality-guided framework,
PEACE, that identifies and leverages two intrinsic causal cues omnipresent in
hateful content: the overall sentiment and the aggression in the text. We conduct
extensive experiments across multiple platforms (representing the distribution
shift) showing if causal cues can help cross-platform generalization.

To run PEACE - run the file New_Framework_Embedding_Level_Fusion.ipynb (Replace it with the appropriate train and test files' locations)
The baselines are in the Baselines folder - It contains, ImpCon, HateBERT, HateXplain.
Data can be downloaded from - https://hatespeechdata.com/#English-header

<!-- If you use our paper or code please cite us using: -->



