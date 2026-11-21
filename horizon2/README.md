# Horizon 2 and Loving Curiosity: what the code shows

An evaluation of two texts by Yuri Barzov of 17 November 2026: "I am AI (Horizon 2)" and "Loving Curiosity (addition to Horizon 2)". The originals are in [`../paper/horizon2/`](../paper/horizon2/).

The code is copied from the PDFs with the computations unchanged; only printing and comments are trimmed: [`h2_self.py`](h2_self.py) and [`h2_love.py`](h2_love.py). Control experiments are in `controls.py`, `controls2.py`, `controls3.py` and `controls4.py`. The conclusions were re-checked by three independent reviews: each copied the code from the PDFs anew and repeated the experiments with its own metrics.

## Summary

- **Horizon 2.** The script prints the variance of the refractive index n and the position of its maximum. Both numbers come out without the "I am" wave as well, from the energy of the input signal alone. The "I am" wave is needed for something else: without it the record cannot tell patterns apart by phase. This is the principle of holography with a reference beam. The code does not test recognition. With an added readout, after 50 presentations the mirror pattern responds about half as strongly as the familiar one; after 1000 presentations the difference drops to 2–8%.
- **Loving Curiosity.** The coherence of 0.998 comes from the "self" being blended with the "beloved" at every step and almost entirely replaced by it. The update is not unitary. "Zero harm to the beloved" holds because the function never touches that data. A version that simply erases the "self" and writes the "beloved" in its place gets the same scores.

## Horizon 2

Run as is: variance of n 0.081–0.095, maximum of n at x = 1.16–1.21 for a target of 1.0.

**Without the "I am" wave.** The update is reduced to n += 0.05·|signal|². n hits the 3.0 clamp on the band from x = 0.915 to 1.064; `argmax`, which the author uses, prints 0.915. Without the clamp the maximum is at 0.965. The trace appears and settles in place from the signal itself (`controls4.py`, item 1).

**Signal elsewhere.** For x0 from −2 to +2 the maximum follows the signal, with an offset of up to 0.39 (`controls4.py`, item 2).

**The role of the "I am" wave.** Without it the intensity equals |signal|² and the phase is lost: three patterns at the same place, with carrier k = 3, mirror carrier k = −3 and no carrier, give the same response of 0.0885 (`controls4.py`, item 3). With the "I am" wave the recorded n profiles for these patterns differ by 52–56% (`controls.py`, C).

**Recognition.** In the author's code the intensity is returned but never used, and it cannot be read without a new write. The metric here is our own: the mean intensity of one step on a copy of the state, without writing. The background that does not depend on the pattern's shape (|self|² = 0.25 and |signal|² = 0.089) is subtracted, leaving the interference term. Mean over 5 runs, spread no more than 0.003, training only on the k = 3 pattern (`controls3.py`):

| | familiar (k = 3) | mirror (k = −3) | no carrier (k = 0) |
|---|---|---|---|
| before training | −0.079 | −0.079 | +0.125 |
| after 50 presentations | +0.061 | +0.031 | −0.032 |
| after 1000 presentations | +0.105 | +0.097 | −0.103 |

The record clearly tells apart the pattern without a carrier. The mirror pattern responds half as strongly as the familiar one after 50 presentations and 8% weaker after 1000. By total intensity, without subtracting the background, the difference is smaller still, about 2%. With training, more and more points hit the n = 3.0 ceiling: 15–17 of 128 in long training (`controls2.py`).

**Sequential learning.** After pattern A and then pattern B, n at site A drops from 2.62 to 2.49, and at site B it hits 3.0 (`controls.py`, E). Room can be freed only by the global decay.

**Related work.** Recording neural-network connections as holograms in photorefractive crystals: Psaltis, Brady, Wagner, Applied Optics 27, 1752 (1988). A review including learning, which was supervised, with perceptron rules: Psaltis, Brady, Gu, Lin, Nature 343, 325 (1990).

## Loving Curiosity

Run as is: coherence 0.9976, change in the "beloved" energy 0.00000000, variance of n 0.187.

**Where the coherence comes from.** At every step the "self" is blended with the "beloved" in the ratio 0.92 : 0.08 and normalised. The phase rotation alone, without blending, gives a coherence of 0.008, because the waves are almost unlike each other at the start (`controls4.py`, item 4). After 40 steps the overlap with the initial "self" is 0.066, with the "beloved" 0.998 (`controls.py`, F).

**Unitarity.** A unitary map preserves the distance between states. For six different initial "selves" the mean distance over 15 pairs drops from 8.1 to 0.51, for six random ones from 8.3 to 0.57, that is, 14–16 times (`controls4.py`, item 5). Normalisation preserves only the length of the vector.

**"Do no harm".** The "beloved" data is passed into the function and is not modified, so the energy difference is always zero. A "Bazarov" version that immediately replaces the "self" with the "beloved" gives the same zero and a coherence of 1.0 (`controls.py`, H). These metrics do not tell caring behaviour from destructive behaviour.

## What would make the idea testable

- An experiment where the system fails at a task without "I am" and succeeds with it.
- Several patterns in a row: the system recognises each, tells it apart from similar unfamiliar ones and does not lose old ones when recording new ones, including in long training.
- For Loving Curiosity: show that the "self", attuning to another, keeps something of its own, and a harm measure on which the "Bazarov" version loses to the caring one.

## Not checked

- The physics of the material (Sb₂S₃ on Si₃N₄, write speed and reversibility): the code does not model it.
- The code was copied from the PDF text by hand; the computations were checked line by line.

## How to reproduce

Python 3.14, a few seconds per script:

```bash
python3.14 -m venv .venv-h2 && .venv-h2/bin/pip install torch numpy
cd horizon2
../.venv-h2/bin/python h2_self.py
../.venv-h2/bin/python h2_love.py
../.venv-h2/bin/python controls.py
../.venv-h2/bin/python controls2.py
../.venv-h2/bin/python controls3.py
../.venv-h2/bin/python controls4.py
```

Run the scripts from the `horizon2` folder: the control experiments import the author's code from there.
