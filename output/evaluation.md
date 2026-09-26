# Évaluation des modèles

60 avis comparés à des annotations manuelles ([`data/annotations.csv`](../data/annotations.csv)).

| modèle           | sentiment (exactitude)   | thèmes (correspondance exacte)   |   thèmes (F1) | coût pour 1 000 avis   | latence moyenne   |   échecs |
|:-----------------|:-------------------------|:---------------------------------|--------------:|:-----------------------|:------------------|---------:|
| claude-haiku-4-5 | 97%                      | 93%                              |          0.98 | 0.72 $                 | 1.3 s             |        0 |
| claude-sonnet-5  | 95%                      | 95%                              |          0.99 | 1.83 $                 | 1.9 s             |        0 |
| claude-opus-5    | 100%                     | 87%                              |          0.96 | 4.61 $                 | 2.0 s             |        0 |

## claude-haiku-4-5

Matrice de confusion (sentiment) :

| référence ↓ / prédit →   |   positif |   neutre |   négatif |
|:-------------------------|----------:|---------:|----------:|
| positif                  |        27 |        0 |         0 |
| neutre                   |         0 |        8 |         2 |
| négatif                  |         0 |        0 |        23 |

Erreurs de sentiment :

- « Bonne qualité mais on m'a oublié deux fois alors que j'attendais mon tour. » → prédit **négatif**, attendu **neutre**
- « Entrée excellente, plat décevant. Service correct. » → prédit **négatif**, attendu **neutre**

## claude-sonnet-5

Matrice de confusion (sentiment) :

| référence ↓ / prédit →   |   positif |   neutre |   négatif |
|:-------------------------|----------:|---------:|----------:|
| positif                  |        27 |        0 |         0 |
| neutre                   |         0 |        7 |         3 |
| négatif                  |         0 |        0 |        23 |

Erreurs de sentiment :

- « Bonne qualité mais on m'a oublié deux fois alors que j'attendais mon tour. » → prédit **négatif**, attendu **neutre**
- « Cadre agréable, cuisine correcte, mais 28 € le plat c'est beaucoup. » → prédit **négatif**, attendu **neutre**
- « Pas mal mais les portions sont petites pour le prix. » → prédit **négatif**, attendu **neutre**

## claude-opus-5

Matrice de confusion (sentiment) :

| référence ↓ / prédit →   |   positif |   neutre |   négatif |
|:-------------------------|----------:|---------:|----------:|
| positif                  |        27 |        0 |         0 |
| neutre                   |         0 |       10 |         0 |
| négatif                  |         0 |        0 |        23 |
