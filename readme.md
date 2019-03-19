# 8-bit VAE: A Latent Variable Model for NES Music
### Xavier Garcia

This github contains the code for 8-bit VAE, a latent variable model for Nintendo Entertainment System (NES) music. Before diving into the details, here are a couple of samples derived from the model: 

[Sample 1](https://soundcloud.com/xavier-garcia-958359339/52-sample-6)

[Sample 2](https://soundcloud.com/xavier-garcia-958359339/sample-f)

You can read more about the model in the blog [post](https://xgarcia238.github.io/misc/2018/03/18/8bitvae.html). To use this, first prepare the data by running the `prepare_data.py`. Once the data is prepared, run the `train.py` model to train the model. To generate music, run the `generate_tr.py` script. If you want to train without the TR voice, then modify the data accordingly and use `generate.py` to generate music.
