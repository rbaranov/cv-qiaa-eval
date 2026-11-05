| Вариант | Запусков | Val loss (nats) | Бит на символ | Секунд на 3000 шагов | Параметров attention | NaN |
|---|---|---|---|---|---|---|
| softmax | 3 | 0.974 ± 0.003 | 1.406 | 130 | 262,656 | 0 |
| born | 3 | 1.034 ± 0.001 | 1.492 | 167 | 262,656 | 0 |
| cvqiaa | 2 | 1.284 ± 0.035 | 1.853 | 2257 | 181,248 | 0 |
| none | 3 | 2.298 ± 0.000 | 3.315 | 71 | 0 | 0 |

Кривые val loss (seed 0):

| Шаг | born | cvqiaa | none | softmax |
|---|---|---|---|---|
| 1 | 4.684 | 4.675 | 4.781 | 4.685 |
| 250 | 1.746 | 2.255 | 2.318 | 1.483 |
| 500 | 1.470 | 2.105 | 2.311 | 1.261 |
| 750 | 1.333 | 1.869 | 2.308 | 1.176 |
| 1000 | 1.248 | 1.672 | 2.307 | 1.118 |
| 1250 | 1.191 | 1.549 | 2.304 | 1.085 |
| 1500 | 1.146 | 1.442 | 2.304 | 1.053 |
| 1750 | 1.111 | 1.377 | 2.301 | 1.032 |
| 2000 | 1.085 | 1.334 | 2.301 | 1.010 |
| 2250 | 1.057 | 1.300 | 2.299 | 0.992 |
| 2500 | 1.044 | 1.275 | 2.299 | 0.980 |
| 2750 | 1.035 | 1.263 | 2.298 | 0.974 |
| 3000 | 1.033 | 1.260 | 2.297 | 0.972 |

Образцы текста (seed 0, начало «Once upon a time», температура 0.8):

**born**

```
Once upon a time, there was a boy named Timmy. Timmy loved to play outside in the oat. Timmy was very stodies and card because he was playing with the clothes, taking with them, but his mom as differer for the new is of the beench this races and said, "I is this is owner." Joe couldn't like it to stuck in the bunny
```

**cvqiaa**

```
Once upon a time, there was a big named it feel safe. Then alte go of be playing out the and pullen with his to cricrmbles. But were to as quee to the cool. The and some some leep stay dad anysy found the stay so and onsid. She saw sill "Whe fun!" Jom was said. Dotanted What they bear and shest to the big trare, to
```

**none**

```
Once upon a timerned. chid ute On m led imoupe Thange lofthap s g fo wandelurofuloomel biat antor ge sitod Here nd arm mesthe wake they a t 
Onire the I d ply, ta and w, s t ollengsthind, hed s ckedapo foustt an. Tistous wo ine s d Tile "
The anoom wa wanind wit brand fe hay the angasar hed asast ats he ran are, to
```

**softmax**

```
Once upon a time, there was a big mill disturbed and saw that she was better if they were grandled with her best friend, she felt very party! 
<|endoftext|>
Once upon a time, there was a little girl named Lily. He liked to see what his dad said, "Okay, I was walking with her toys again. It was so scared and ran awa
```
