#Ancient Langauges
Because many documents from early societies were written in clay, we have an abundance of records which greatly exceeds the number of people able to read them. To that end, machine translations of these documents is a promising area to provide solutions. To this end, this project pulls together models trained on multiple ancient scripts and languages:

Akkadian: [AKK-60m](https://huggingface.co/Thalesian/AKK-60m)
| From Language | From Script     | To Language | To Script       | Bleu  |
| ------------- | --------------- | ----------- | --------------- | ----- |
| Akkadian      | Cuneiform       | English     | Latin           | 70.11 |
| Akkadian      | Transliteration | English     | Latin           | 70.94 |
| Akkadian      | Cuneiform       | Akkadian    | Transliteration | 93.87 |
| English       | Latin           | Akkadian    | Transliteration | 45.51 |
| English       | Latin           | Akkadian    | Cuneiform       | 47.10 |

Hittite: [HIT-60m](https://huggingface.co/Thalesian/HIT-60m)
| From Language | From Script      | To Language | To Script        | BLEU  |
|---------------|------------------|-------------|------------------|------:|
| Hittite       | Transliteration  | German      | Latin            | 83.38 |
| Hittite       | Transliteration  | English     | Latin            | 60.92 |
| German        | Latin            | Hittite     | Transliteration  | 42.90 |
| English       | Latin            | Hittite     | Transliteration  | 38.50 |

Linear B/Mycenan Greek: [GMY-60m](https://huggingface.co/Thalesian/GMY-60m)
| From Language      | From Script      | To Language         | To Script        | BLEU   |
|--------------------|------------------|---------------------|------------------|-------:|
| Mycenean Greek     | Linear B         | English             | Latin            |  55.55 |
| Mycenaean Greek    | Transliteration  | English             | Latin            |  54.63 |
| Mycenaean Greek    | Linear B         | Mycenaean Greek     | Transliteration  |  82.52 |
| English            | Latin            | Mycenaean Greek     | Transliteration  |  29.20 |
| English            | Latin            | Mycenaean Greek     | Linear B         |  30.79 |

We are working next on Sumerian and Elamite. Sumerian's principle challenge is low translation quality, Elamite's is sufficient texts to train a model. 

Disclaimer: These scores (particulalry Hittite -> German and Akkadian -> English) are very high - this is because we are training on extremely short contexts of lines on small tablets; it is the nature of the data we have. 


