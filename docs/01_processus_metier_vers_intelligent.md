# 1. Transformer un processus métier en processus intelligent

## 1.1 Cartographier l'existant (as-is)

Le processus de rétention actuel est réactif : il démarre quand le client appelle
pour partir.

| Étape | Qui | Quoi | Problème |
|---|---|---|---|
| 1 | Client | Résiliation (appel ou fin de contrat) | trop tard |
| 2 | Call-center | Tentative de rétention, offre standardisée | aucune segmentation |
| 3 | Merchandiser | Enregistrement manuel du démarchage | pas de mesure |
| 4 | Direction | Bilan mensuel du taux de churn | résiliation subie, pas anticipée |

Causes racines :
- aucune donnée d'alerte avant le départ (pas de score de risque par client) ;
- priorisation manuelle par intuition ;
- aucune boucle de mesure : impossible de dire si la campagne a rapporté de l'argent.

## 1.2 Le processus cible (to-be)

Un service de scoring remplace l'organisation manuelle ; le processus s'inverse :
on agit avant le départ.

| Étape | Qui | Quoi | Amélioration |
|---|---|---|---|
| 1 | Modèle (quotidien) | Score de chaque client | risque connu avant la résiliation |
| 2 | Modèle + seuil métier | Sélection des clients à contacter | priorisation par valeur attendue |
| 3 | Commercial | Appel + offre individualisée | explication SHAP fournie |
| 4 | Produit | Suivi du taux de rétention réel | ajustement du seuil et des coûts |

## 1.3 Ce qui change, en une synthèse

Le cœur du changement n'est pas le modèle, c'est la décision :

- avant : tous les clients résiliés reçoivent une offre standardisée, sans priorité ;
- après : chaque client ciblé reçoit une offre justifiée (score + explication SHAP),
  dans une fenêtre de temps où il peut encore dire oui.

Trois ingrédients suffisent :
1. un déclencheur automatique (score quotidien) ;
2. une règle de décision explicite (seuil économique, pas 0.5 par défaut) ;
3. une boucle de mesure (valeur nette vs ciblage aléatoire).

## 1.4 Pourquoi ça marche

Trois blocs conditionnent le succès. Si l'un manque, le projet s'arrête :

- données fiables : l'anomalie `tenure = 0 => TotalCharges vide` est traitée à 0
  (règle métier, traçée) plutôt que perdue dans des `dropna` silencieux ;
- pipeline sans fuite : le feature engineering vit dans le pipeline sklearn
  (`FunctionTransformer`) ; le fit se fait uniquement sur l'entraînement. Le brouillon
  v0 (scaler ajusté avant la séparation) gonflait artificiellement les métriques ;
- seuil piloté par la valeur : ROC-AUC 0.844 sans seuil = un modèle de démonstration
  sans valeur opérationnelle. Avec `BusinessContext` (coût churn 500 EUR, contact
  50 EUR, taux de sauvetage 30 %), le modèle devient un outil de décision
  paramétrable sans modification de code.

## 1.5 Mesure du gain — jeu de test (1 409 clients)

Hypothèses : coût de churn 500 EUR, coût de contact 50 EUR, taux de sauvetage 30 %.

| Scénario | Valeur nette | Lecture |
|---|---|---|
| No action (comme avant) | -187 000 EUR | 374 churners perdus sans intervention |
| Contact aléatoire au même budget (487 appels) | -192 000 EUR | coût de campagne sans ciblage |
| Campagne pilotée par le modèle (seuil 0.34) | -170 100 EUR | gain net +16 900 EUR vs no action |
| Uplift ciblage modèle vs aléatoire | +21 900 EUR | la valeur vient du ciblage, pas du budget |

Détail de la campagne sur le test : 487 clients ciblés, 275 vrais churners
(rappel 73,5 %), précision 56,5 %. Au taux de sauvetage de 30 %, environ 82 churners
sauvés.

À l'échelle du portefeuille complet (7 043 clients), le gain potentiel est
proportionnel. Le sponsor peut recalculer avec ses propres coûts via les trois
paramètres de `BusinessContext`, sans modifier le code.

## 1.6 Ce que chaque acteur y gagne

- le client : des propositions qui correspondent à son profil, moins de sollicitations
  inutiles ;
- le commercial : une file priorisée et expliquée (facteurs SHAP par client) ;
- la direction : valeur nette, budget et taux de ciblage chiffrés, décision
  reproductible.

Critère de définition : le processus est intelligent quand on peut répondre à
« Pourquoi contacter ce client avant celui-là ? » avec un chiffre et une explication,
pas avec une intuition.
