# 3. Réflexions

Quatre réflexions issues de l'exercice.

## 3.1 Le seuil économiquement optimal est « mieux que la précision »

Il est tentant d'évaluer un modèle de churn avec l'exactitude ou la précision.
Ce serait une erreur ici :

- les classes sont déséquilibrées (26,5 % de churners) : un modèle qui prédit
  « personne ne part » atteint 73,5 % d'exactitude et zéro valeur métier ;
- les erreurs ne coûtent pas pareil : rater un churner coûte 500 EUR, appeler un
  client fidèle coûte 50 EUR. Le seuil optimal découle de ce ratio ;
- c'est le sens du seuil économetrique : `coût de contact / (taux de sauvetage *
  coût de churn) = 0,33`, recomputé par le notebook à chaque changement de
  contexte.

Conséquence : quand le sponsor conteste le nombre de clients ciblés, la discussion
se déplace du modèle vers les coûts. C'est une discussion plus honnête et plus
utile.

## 3.2 Le pipeline est une décision de risque, pas un choix de style

Deux approches existaient :

1. préparer le tableau dans le notebook, puis entraîner le modèle dessus ;
2. intégrer le feature engineering au pipeline sklearn.

L'approche 1 est rapide à écrire, mais le gap entre entraînement et production est
constant : le déploiement doit copier à la main les transformations, et le risque
de fuite est permanent. L'approche 2 laisse sklearn gérer fit/transform, donc les
statistiques de prétraitement sont estimées uniquement sur l'entraînement, et le
bundle joblib embarque tout : le modèle sa fonctionne sur les colonnes brutes.

Le brouillon v0 suivait l'approche 1 et gonflait les métriques d'environ dix
points. Le passage à l'approche 2 a rendu le problème rêve principal recevable.

## 3.3 Le SHAP agrégé : de l'explication du modèle à l'explication métier

Le one-hot crée des dizaines de colonnes ; expliquer une modalité isolée à un
commercial n'a pas de sens. On agrège les valeurs SHAP par feature d'origine
(ex. toutes les modalités de `InternetService` sommées) : le modèle ne change pas,
l'interface d'explication si.

Le résultat est parlant : 26 colonnes techniques se lisent en 16 concepts métier,
tels « pas de contrat », « ancienneté faible », « fibre + facture élevée ». Chaque
relatum pointe vers une action commerciale possible.

## 3.4 Rendement décroissant : quand arrêter le tuning

Après la recherche aléatoire, le gain d'AUC sur le test était inférieur à 0,01 face
à la configuration par défaut, alors que le coût en temps, complexité et
maintenabilité restait entier. Le Gradient Boosting retenu l'a été pour son couple
performance/stabilité, pas pour le maximum d'AUC.

Principe retenu : au-delà du plateau, les gains viennent des données (historique
des campagnes, données d'usage), pas des hyperparamètres. C'est là que porter
l'effort ensuite.
