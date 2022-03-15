# hmm_text_generator

Hmm text generator is a basic Hidden Markov Model text generator to use on Telegram message data in Turkish. It does parse and process JSON data that has obtained by Telegram Desktop App, feeds your messages to Hidden Markov Model ,and returns text sample from the model.
HMM does not provide wonders for language models so abandon hope all ye who use this.

## Installation
### Ubuntu
---
Setup script needs root user priviliges to install docker

```
$ sudo -u root

# ./setup.sh
```
## Usage

For a random length text simply run the view.py file

```
$ python3 ./view.py
```
or you can add minimum word length you want as argument

```
$ python3 ./view.py 60
```
