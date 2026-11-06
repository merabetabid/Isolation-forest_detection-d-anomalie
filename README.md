# Isolation-forest_detection-d-anomalie
Isolation Forest est une méthode efficace de détection d’anomalies car elle isole directement les observations au lieu de modéliser leur distribution
# Détection de fraudes bancaires par apprentissage non supervisé


L'objectif : repérer les transactions frauduleuses par carte bancaire **sans utiliser les étiquettes pendant l'entraînement**, en comparant trois algorithmes de détection d'anomalies, puis en optimisant le plus prometteur (Isolation Forest).

## Problématique

Les fraudes sont rarissimes (environ 0,17 % des transactions) mais coûteuses. Comment les détecter efficacement tout en limitant les fausses alertes qui dégradent l'expérience des clients ?

## Données

- **Dataset** : Credit Card Fraud Detection (Kaggle, issu d'une collaboration Worldline / ULB)
- **Contenu** : 284 807 transactions de cartes bancaires, dont 492 frauduleuses
- **Variables** : `Time`, `Amount`, `V1` à `V28` (composantes anonymisées par ACP) et `Class` (0 = normale, 1 = fraude)

Le fichier `creditcard.csv` (environ 150 Mo) **n'est pas inclus** dans ce dépôt. Voir [`data/README.md`](data/README.md) pour le télécharger.

## Méthode

1. **Prétraitement** : séparation variables / cible, standardisation (`StandardScaler`), découpage 80 % entraînement / 20 % test (stratifié, `random_state=42`)
2. **Modèles comparés** :
   - Isolation Forest (100 arbres)
   - Local Outlier Factor (20 voisins)
   - One-Class SVM (noyau RBF)
3. **Évaluation** : matrices de confusion, précision et rappel sur la classe « fraude »
4. **Optimisation** : variation du paramètre `contamination` d'Isolation Forest (0,001 ; 0,0017 ; 0,005 ; 0,01)
5. **Visualisation** : projection ACP en 2 dimensions de l'ensemble de test

## Résultats

Sur l'ensemble de test (98 fraudes sur 56 962 transactions) :

| Modèle | Précision | Rappel |
|---|---|---|
| Isolation Forest | 0,31 | 0,34 |
| LOF | 0,02 | 0,02 |
| One-Class SVM | 0,07 | 0,47 |

Influence de `contamination` sur Isolation Forest :

| contamination | Précision | Rappel |
|---|---|---|
| 0,001 | 0,39 | 0,26 |
| 0,0017 | 0,31 | 0,34 |
| 0,005 | 0,16 | 0,49 |
| 0,01 | 0,11 | 0,66 |

**Lecture** : plus on autorise de transactions suspectes, plus on attrape de fraudes (rappel en hausse) mais plus on déclenche de fausses alertes (précision en baisse). Isolation Forest offre le meilleur compromis des trois modèles ; le bon réglage dépend du coût relatif d'une fraude manquée et d'une fausse alerte.

## Limites

- Les scores restent modestes : la détection non supervisée ne remplace pas un modèle supervisé quand des étiquettes sont disponibles.
- LOF est appliqué directement sur l'ensemble de test, alors que les deux autres modèles sont entraînés sur l'ensemble d'entraînement.
- Le paramètre `contamination` est réglé en regardant les résultats sur le test, ce qui est optimiste.
- Seuls la précision et le rappel sont étudiés (pas de F1, ni de courbe précision-rappel).


## Installation et utilisation

```bash

pip install -r requirements.txt
# placer creditcard.csv dans le dossier data/
python src/main2.py
```

Le script affiche les matrices de confusion et rapports de classification, puis ouvre les graphiques.

