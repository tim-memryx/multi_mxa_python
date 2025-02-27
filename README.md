# Multi-MXA Python Examples

## Setup

**Before** cloning this repo, make sure you have `git-lfs` installed. If not, run `sudo apt install git-lfs`.

Create a new python venv if you don't already have one for the MemryX SDK, e.g.

```bash
python3 -m venv ~/mx
```

Then activate it, update pip, and install the requirements:

```bash
. ~/mx/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt
```

You're all set to try the examples!


## Usage

Please see the individual example READMEs, and try out the examples in the recommended order.

---

1. **separate processes**: example usage of completely separate Python applications

2. **combined**: single process that runs the input stream in 2 different DFPs on 2 different MXAs

3. **load-balancing**: run an input stream across all 4 connected MXAs to increase total throughput
