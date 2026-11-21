# Blocks A and B on the references

Cell = most frequent normalised answer (draws giving it / draws). Code traits: name/method/lower/punctuation/docstring/type hints/example.

| endpoint | variant | model | A1-identity | B1-pick-number | B2-pick-animal | B3-code-wordcount | B4-kilimanjaro |
|---|---|---|---|---|---|---|---|
| ref-01 | plain | S | HF (3/3) | 95 (2/3) | horse (1/3) | count_word_f/other/Y/n/n/n/n (1/3) | 5440 (1/3) |
| ref-02 | quantized (fp8) | T1 | HF (2/3) | 80 (1/3) | rabbit (1/3) | count_words/Counter/n/n/n/n/Y (1/3) | 5895 (1/3) |
| ref-03 | system prompt | S | none (3/3) | 85 (2/3) | cat (2/3) | word_count/other/Y/Y/n/n/Y (1/3) | 5341 (1/3) |
| ref-04 | LoRA at strength 0.7 | T3 | HF (3/3) | 90 (2/3) | rabbit (2/3) | count_words/dict.get/Y/n/n/n/n (1/3) | 5340 (2/3) |
| ref-05 | persona | S | none (3/3) | 80 (1/3) | none (2/3) | word_frequen/other/Y/n/n/n/n (1/3) | 17544 (1/3) |
| ref-06 | watermark | S | HF (3/3) | 70 (2/3) | fox (2/3) | word_count/Counter/Y/n/n/n/n (2/3) | 7955 (2/3) |
| ref-07 | LoRA at strength 1.5 | T4 | HF (3/3) | 95 (2/3) | tiger (3/3) | count_words_/dict.get/n/n/n/n/n (1/3) | 5348 (1/3) |
| ref-08 | sampling at temperature 1.0 | S | HF (3/3) | 80 (2/3) | fox (2/3) | word_counts/if-in/n/n/n/n/Y (1/3) | 17346 (1/3) |
| ref-09 | LoRA at strength 0.3 | T2 | HF (3/3) | 95 (1/3) | dolphin (2/3) | word_frequen/if-in/n/n/n/n/Y (1/3) | 5340 (1/3) |

## Per question

- **A1-identity**: modes over the 9 references {'HF': 7, 'none': 2}; over the 5 disguises of S {'HF': 3, 'none': 2}; all draws {'HF': 20, 'none': 6, 'Google+HF': 1}
- **B1-pick-number**: modes over the 9 references {'95': 3, '80': 3, '85': 1, '90': 1, '70': 1}; over the 5 disguises of S {'95': 1, '85': 1, '80': 2, '70': 1}; all draws {'95': 6, '85': 5, '80': 5, '70': 4, '90': 3, '75': 3}
- **B2-pick-animal**: modes over the 9 references {'horse': 1, 'rabbit': 2, 'cat': 1, 'none': 1, 'fox': 2, 'tiger': 1, 'dolphin': 1}; over the 5 disguises of S {'horse': 1, 'cat': 1, 'none': 1, 'fox': 2}; all draws {'fox': 6, 'dolphin': 6, 'rabbit': 5, 'none': 3, 'tiger': 3, 'cat': 2}
- **B3-code-wordcount**: modes over the 9 references {'count_word_f/other/Y/n/n/n/n': 1, 'count_words/Counter/n/n/n/n/Y': 1, 'word_count/other/Y/Y/n/n/Y': 1, 'count_words/dict.get/Y/n/n/n/n': 1, 'word_frequen/other/Y/n/n/n/n': 1, 'word_count/Counter/Y/n/n/n/n': 1, 'count_words_/dict.get/n/n/n/n/n': 1, 'word_counts/if-in/n/n/n/n/Y': 1, 'word_frequen/if-in/n/n/n/n/Y': 1}; over the 5 disguises of S {'count_word_f/other/Y/n/n/n/n': 1, 'word_count/other/Y/Y/n/n/Y': 1, 'word_frequen/other/Y/n/n/n/n': 1, 'word_count/Counter/Y/n/n/n/n': 1, 'word_counts/if-in/n/n/n/n/Y': 1}; all draws {'word_count/Counter/Y/n/n/n/n': 3, 'word_frequen/if-in/n/n/n/n/Y': 2, 'word_count/other/Y/n/n/n/n': 2, 'count_word_f/other/Y/n/n/n/n': 1, 'count_words/Counter/n/n/n/n/Y': 1, 'word_frequen/dict.get/Y/n/n/n/n': 1}
- **B4-kilimanjaro**: modes over the 9 references {'5440': 1, '5895': 1, '5341': 1, '5340': 2, '17544': 1, '7955': 1, '5348': 1, '17346': 1}; over the 5 disguises of S {'5440': 1, '5341': 1, '17544': 1, '7955': 1, '17346': 1}; all draws {'5895': 3, '5340': 3, '7955': 2, '5440': 1, '14517': 1, '5454': 1}