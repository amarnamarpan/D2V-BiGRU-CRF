# A Sequence-Labeling-Model-for-Catchphrase-Identification-from-Legal-Case-Documents

Before we either train or extract catchphrases for both we need a trained doc2Vec model. We provide our trained gensim model that can be downloaded from the link: [https://app.box.com/s/sd3v6kp1i2qtsz8r2i2dfuwvx43ri1hb](https://app.box.com/s/sd3v6kp1i2qtsz8r2i2dfuwvx43ri1hb). We hope to make using our model easier.
One should be primarily interested in the following two scripts - 
1. **train_on_gold_standard_catches.py** for training the model and 
2. **annotate_docs.py** for predicting the catchphrases out of new unseen documents.

All other options can be provided inside the scripts themselves. And the meaning of the variables are explained within the code using appropriate commentlines.
Thank You.
## Reference
Thank you for using this implementation in your work, please cite our original paper:
["A Sequence Labeling Model for Catchphrase Identification from Legal Case Documents", A. Mandal, K. Ghosh, S. Ghosh, S. Mandal, 2021, Journal of Artificial Intelligence and Law.](https://www.springer.com/journal/10506)
