# Training a model for coder, on the machine you are reading this on

Four steps, and the first one is the one that decides whether the rest are
worth doing.

```sh
coder dataset build          # collect and inspect the corpus
training/train.sh            # LoRA, a few hours
training/fuse.sh             # one directory a server can host
training/serve.sh            # on the network, for every device you own
```

## 1. Look at the data before training on it

```sh
coder dataset build
coder dataset show 0
```

`build` reads every transcript on this machine -- coder's own sessions, Claude
Code's, Codex's -- translates each call into coder's protocol, and reports what
it found. Read the histogram. It is the only thing that will tell you whether
`max_seq_length` is set right, and a corpus whose mass sits above the window is
one that will be silently truncated during training.

Then read one example. A histogram cannot tell you a conversation is
incoherent; a minute with `show` can, and everything downstream assumes the
conversations make sense.

**If there is not much data, that is the normal finding**, and `generate.py` is
the answer to it:

```sh
python training/generate.py            # tasks against this repo, via claude-code
```

Those runs are ordinary coder sessions with a frontier model choosing the
calls, so they come out already in the target protocol and `coder dataset
build` picks them up with no extra flag.

An outside corpus goes in the same way:

```sh
coder dataset build --source coder,claude,codex,chat --file mine.jsonl
```

## 2. Train

```sh
training/train.sh
```

Roughly a few hours for 600 iterations on an M-series machine. Watch memory
rather than the loss: if the machine starts swapping, the run will finish
eventually and will have taken all day.

### When it does not fit

16 GB, a 4-bit 14b, and a 4096-token sequence is tight on purpose. Everything
resident comes out of the same pool the OS is using: ~8.5 GB of weights, the KV
cache, the adapters, and their optimizer state. Turn these in order, in
`lora.yaml`:

1. `num_layers: 8` → `4` -- fewest adapters, biggest saving, least cost to
   quality for what is being taught here.
2. `max_seq_length: 4096` → `2048`, and rebuild the corpus to match:
   `coder dataset build --max-tokens 2048`. The two must agree, or half of
   every long example is cut off mid-tool-result during training.
3. The 7b, which is the honest fallback rather than a failure:

   ```yaml
   model: "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
   num_layers: 16
   ```

   ~4.5 GB of weights instead of ~8.5, so there is room for more adapted layers
   and a longer sequence. On a machine this size a well-trained 7b beats a 14b
   that cannot hold its context.

Raising `batch_size` is the wrong direction. There is no headroom, and unequal
example lengths mean a larger batch pays for padding.

## 3. Fuse and serve

```sh
training/fuse.sh
training/serve.sh
```

`fuse.sh` folds the adapter into the weights, giving one directory that needs
no memory of what it was trained against. `serve.sh` hosts it on `0.0.0.0:8080`,
which is where `coder`'s `local` backend looks by default.

From the machine itself:

```sh
coder doctor          # server answers, model listed, window fits
coder "add a --verbose flag"
```

From anything else on the network:

```sh
coder --host http://<that-machine>:8080
coder config set host http://<that-machine>:8080     # to keep it
```

The endpoint has no authentication. Run it on a network you trust or put it
behind a tunnel; `CODER_API_KEY` is sent as a bearer token if you set it, for
whatever you put in front.

## 4. Judge it against what it replaced

```sh
coder review --provider local  --host http://localhost:8080
coder review --provider ollama --model qwen2.5-coder:14b
```

Review is the measurement this repo already has: `for_review` pins temperature
to zero on both backends, so the same diff gives the same findings and a
difference between the two runs is a difference between the models rather than
a dice roll.

The failure worth watching for is not a worse answer, it is a malformed one --
a model that has drifted off the calling format. `coder --verbose` shows the
raw replies. If calls stop parsing, the corpus and the served prompt have
diverged, and `coder dataset show 0` next to a live `--verbose` transcript is
where that shows up.
