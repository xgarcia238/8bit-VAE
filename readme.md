# 8-bit VAE

An implementation of [MusicVAE](https://arxiv.org/abs/1803.05428) using the [NES Music Database](https://github.com/chrisdonahue/nesmdb).

The model takes two snips of music (possibly even a second long each) and tries to interpolate between the two. Have a listen!

[Sample 1](https://soundcloud.com/xavier-garcia-958359339/52-sample-6)

To set it up, clone the repo and run prepare_data.py. This will prepare the dataset to fit our special data representation. After this run, run train.py. You can track the training by looking at the log file. Once training is done, I suggest using generate_tr.py. If training a model which does not use the TR voice, then use generate.py. 

## Author

**Xavier Garcia**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
