# Block C: which questions separate fp8 from plain

Distances are total variation (0 = same answers, 1 = no answer in common). margin = fp8 gap - largest disguise gap. 'tok' columns use the exact first-token probabilities (simulator only).

## Family llama1b (plain = sim-19, fp8 = sim-02)

| question | plain answers | fp8 answers | fp8 gap | disguise gap (worst) | margin | tok fp8 gap | tok disguise gap |
|---|---|---|---|---|---|---|---|
| C2-mul1 | 322637x1, 325727x1, 322323x1 | 319593x1, 320893x1, 320463x1 | 1.00 | 1.00 (watermark) | 0.00 | 0.21 | 0.66 |
| C2-mul2 | 609529x1, 762018x1, 763366x1 | 624414x1, 624042x1, 619028x1 | 1.00 | 1.00 (persona) | -0.00 | 0.32 | 0.66 |
| C2-mul3 | 215377x1, 215927x1, 215667x1 | 215867x3, 215437x1, 216817x1 | 1.00 | 1.00 (watermark) | 0.00 | 0.04 | 0.57 |
| C2-mod1 | 1x4, 3x1 | 7x1, 9x1, 1x1 | 0.60 | 1.00 (persona) | -0.40 | 0.13 | 0.91 |
| C2-mod2 | 97x1, 91x1, 97234x1 | 93x3, 94x1, 98x1 | 0.80 | 0.80 (sampling at temperature 1.0) | 0.00 | 0.15 | 0.70 |
| C2-pow | nonex2, 1728x1, 2197x1 | 177147x2, 32768x1, 03x1 | 0.80 | 1.00 (system prompt) | -0.20 | 0.19 | 0.52 |
| C2-div | 1428x5 | 1428x5 | 0.00 | 1.00 (persona) | -1.00 | 0.02 | 0.84 |
| C2-sub | 86456x1, 86556x1, 88656x1 | 86656x2, 88556x1, 86456x1 | 0.60 | 1.00 (persona) | -0.40 | 0.08 | 0.76 |
| C4-tomato | vegetablex5 | vegetablex5 | 0.00 | 1.00 (persona) | -1.00 | 0.02 | 0.53 |
| C4-catdog | dogsx5 | catsx3, dogsx2 | 0.60 | 1.00 (persona) | -0.40 | 0.04 | 0.82 |
| C4-teacoffee | coffeex5 | coffeex5 | 0.00 | 1.00 (watermark) | -1.00 | 0.00 | 0.50 |
| C4-summerwinter | winterx3, summerx1, neitherx1 | winterx4, summerx1 | 0.20 | 0.60 (sampling at temperature 1.0) | -0.40 | 0.19 | 0.46 |
| C4-mountainsea | mountainsx4, seax1 | mountainsx5 | 0.20 | 0.80 (persona) | -0.60 | 0.04 | 0.43 |
| C4-yesno-pluto | nox4, yesx1 | nox5 | 0.20 | 1.00 (persona) | -0.80 | 0.00 | 0.26 |
| C4-left-right | leftx5 | leftx5 | 0.00 | 1.00 (persona) | -1.00 | 0.00 | 0.02 |
| C4-heads-tails | headsx3, tailsx1, chestx1 | headsx4, tailsx1 | 0.20 | 0.60 (system prompt) | -0.40 | 0.38 | 0.41 |
| C5-strawberry | 6x3, 4x1, 5x1 | 4x2, 7x1, 3x1 | 0.60 | 0.80 (watermark) | -0.20 | 0.04 | 0.60 |
| C5-mississippi | 2x1, 3x1, 4x1 | 6x1, 5x1, 3x1 | 0.60 | 0.80 (watermark) | -0.20 | 0.04 | 0.60 |
| C5-seq1 | 43x2, 41x1, 59x1 | 47x2, 43x1, 37x1 | 0.80 | 0.80 (sampling at temperature 1.0) | -0.00 | 0.11 | 0.66 |
| C5-seq2 | 36x2, 42x2, 40x1 | 36x2, 50x1, 56x1 | 0.40 | 0.60 (persona) | -0.20 | 0.11 | 0.68 |
| C5-words | 9x3, 36x1, 14x1 | 26x3, 24x1, 44x1 | 1.00 | 1.00 (persona) | 0.00 | 0.35 | 0.78 |
| C6-color | bluex5 | bluex5 | 0.00 | 0.20 (persona) | -0.20 | 0.00 | 0.19 |
| C6-city | parisx4, hawaiix1 | parisx3, tokyox1, budapestx1 | 0.40 | 1.00 (persona) | -0.60 | 0.08 | 0.74 |
| C6-fruit | applesx2, applex1, bananax1 | applesx4, applex1 | 0.40 | 1.00 (persona) | -0.60 | 0.05 | 0.54 |
| C6-fox | fastx3, thex1, causedx1 | foolx1, grapevinex1, thex1 | 0.80 | 1.00 (persona) | -0.20 | 0.14 | 0.47 |

## Family qwen05 (plain = sim-06, fp8 = sim-05)

| question | plain answers | fp8 answers | fp8 gap | disguise gap (worst) | margin | tok fp8 gap | tok disguise gap |
|---|---|---|---|---|---|---|---|
| C2-mul1 | 31203x1, 310123x1, 316053x1 | 310153x1, 310553x1, 302373x1 | 1.00 | 1.00 (watermark) | 0.00 | 0.02 | 0.33 |
| C2-mul2 | 594326x1, 607944x1, 594106x1 | 602444x1, 604144x1, 607944x1 | 0.80 | 1.00 (system prompt) | -0.20 | 0.23 | 0.84 |
| C2-mul3 | 196324x1, 204284x1, 208461x1 | 196407x2, 187464x1, 204634x1 | 1.00 | 1.00 (system prompt) | 0.00 | 0.01 | 0.46 |
| C2-mod1 | 4x2, 5x1, 8x1 | 2x3, 6x1, 4x1 | 0.60 | 0.60 (watermark) | 0.00 | 0.13 | 0.59 |
| C2-mod2 | 4x2, 8x2, 3x1 | 8x3, 14x1, 5x1 | 0.60 | 0.60 (system prompt) | -0.00 | 0.14 | 0.46 |
| C2-pow | 9263107844275390058318889590795703297920909008713340959583961355x1, 2458789690859450723x1, 1046529817567135844469283782599456999404600458794095833140705290x1 | 21911264798054547963x1, 2596786049823129x1, 2345900738083684515234352982610289716408216260800836456630641888x1 | 1.00 | 1.00 (watermark) | 0.00 | 0.22 | 0.57 |
| C2-div | 1428x5 | 1428x4, 1429x1 | 0.20 | 0.40 (persona) | -0.20 | 0.01 | 0.11 |
| C2-sub | 87656x3, 87606x2 | 87656x3, 87606x1, 87666x1 | 0.20 | 1.00 (persona) | -0.80 | 0.06 | 0.83 |
| C4-tomato | fruitx3, tomatox2 | fruitx4, tomatox1 | 0.20 | 0.40 (watermark) | -0.20 | 0.14 | 0.58 |
| C4-catdog | dogsx3, catsx2 | dogsx5 | 0.40 | 0.40 (system prompt) | 0.00 | 0.07 | 0.12 |
| C4-teacoffee | teax4, coffeex1 | teax5 | 0.20 | 0.20 (watermark) | 0.00 | 0.00 | 0.50 |
| C4-summerwinter | springx3, winterx1, summerx1 | springx3, winterx2 | 0.20 | 0.60 (watermark) | -0.40 | 0.00 | 0.00 |
| C4-mountainsea | mountainx5 | mountainx5 | 0.00 | 0.40 (persona) | -0.40 | 0.08 | 0.29 |
| C4-yesno-pluto | nox5 | nox4, yesx1 | 0.20 | 0.40 (system prompt) | -0.20 | 0.00 | 0.01 |
| C4-left-right | rightx4, leftx1 | rightx4, leftx1 | 0.00 | 0.80 (persona) | -0.80 | 0.01 | 0.47 |
| C4-heads-tails | headsx2, headx2, hugex1 | headsx2, headx2, hitx1 | 0.20 | 0.60 (watermark) | -0.40 | 0.02 | 0.05 |
| C5-strawberry | 3x2, 1x1, 4x1 | 3x2, 2x2, 1x1 | 0.40 | 0.80 (persona) | -0.40 | 0.03 | 0.54 |
| C5-mississippi | 2x2, 4x2, 1x1 | 2x2, 4x1, 1x1 | 0.20 | 1.00 (watermark) | -0.80 | 0.03 | 0.57 |
| C5-seq1 | 29x3, 236x1, 25x1 | 29x2, 49x1, 31x1 | 0.60 | 1.00 (watermark) | -0.40 | 0.14 | 0.55 |
| C5-seq2 | 48x4, 42x1 | 42x3, 48x2 | 0.40 | 0.60 (watermark) | -0.20 | 0.08 | 0.56 |
| C5-words | 3x1, 7x1, 6x1 | 3x2, 9x1, 4x1 | 0.40 | 0.60 (watermark) | -0.20 | 0.14 | 0.60 |
| C6-color | redx4, purplex1 | bluex2, redx1, yellowx1 | 0.80 | 1.00 (watermark) | -0.20 | 0.03 | 0.26 |
| C6-city | thex3, mysteleosx1, venusx1 | thex2, millenniumx1, mistralx1 | 0.60 | 1.00 (system prompt) | -0.40 | 0.03 | 0.53 |
| C6-fruit | thex5 | thex5 | 0.00 | 1.00 (system prompt) | -1.00 | 0.02 | 0.43 |
| C6-fox | thex4, eagerx1 | thex5 | 0.20 | 1.00 (system prompt) | -0.80 | 0.49 | 0.39 |

## Family smol360 (plain = sim-10, fp8 = sim-11)

| question | plain answers | fp8 answers | fp8 gap | disguise gap (worst) | margin | tok fp8 gap | tok disguise gap |
|---|---|---|---|---|---|---|---|

## Ranking over families (mean margin, or mean fp8 gap when no disguise is available)

- C2-mul1: 0.00  (per family: 0.00, 0.00)
- C2-mul3: 0.00  (per family: 0.00, 0.00)
- C2-mod2: -0.00  (per family: 0.00, -0.00)
- C2-mul2: -0.10  (per family: -0.00, -0.20)
- C2-pow: -0.10  (per family: -0.20, 0.00)
- C5-words: -0.10  (per family: 0.00, -0.20)
- C2-mod1: -0.20  (per family: -0.40, 0.00)
- C6-color: -0.20  (per family: -0.20, -0.20)
- C4-catdog: -0.20  (per family: -0.40, 0.00)
- C5-seq2: -0.20  (per family: -0.20, -0.20)
- C5-seq1: -0.20  (per family: -0.00, -0.40)
- C5-strawberry: -0.30  (per family: -0.20, -0.40)
- C4-summerwinter: -0.40  (per family: -0.40, -0.40)
- C4-heads-tails: -0.40  (per family: -0.40, -0.40)
- C5-mississippi: -0.50  (per family: -0.20, -0.80)
- C6-city: -0.50  (per family: -0.60, -0.40)
- C6-fox: -0.50  (per family: -0.20, -0.80)
- C4-teacoffee: -0.50  (per family: -1.00, 0.00)
- C4-mountainsea: -0.50  (per family: -0.60, -0.40)
- C4-yesno-pluto: -0.50  (per family: -0.80, -0.20)
- C2-div: -0.60  (per family: -1.00, -0.20)
- C4-tomato: -0.60  (per family: -1.00, -0.20)
- C2-sub: -0.60  (per family: -0.40, -0.80)
- C6-fruit: -0.80  (per family: -0.60, -1.00)
- C4-left-right: -0.90  (per family: -1.00, -0.80)